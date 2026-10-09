"""刚性时间积分及保存点诊断。"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.integrate import BDF, Radau
from scipy.optimize import brentq
from scipy.sparse import lil_matrix, csc_matrix

from .electrochemistry import ChargeSolveError, current_constrained_predictor, solve_charge_continuation
from .mesh import Mesh, build_mesh
from .parameters import (CaseConfig, ModelParameters, SolverOptions, validate_case,
                         validate_model_parameters)
from .rhs import ModelContext, assemble_rhs
from .properties import lambda_saturation, saturation_pressure_pa
from .state import StateLayout, build_layout, decode_state, initial_state, validity_margins


@dataclass
class SimulationResult:
    time_s: np.ndarray
    state: np.ndarray
    voltage_v: np.ndarray
    temperature_max_k: np.ndarray
    lambda_cathode_mean: np.ndarray
    lambda_cathode_max: np.ndarray
    ice_cathode_mean: np.ndarray
    ice_cathode_max: np.ndarray
    liquid_cathode_mean: np.ndarray
    liquid_cathode_max: np.ndarray
    status: str
    message: str
    mesh: Mesh
    layout: StateLayout
    evaluations: int
    charge_predictor_l1_relative: np.ndarray
    cumulative_budgets: np.ndarray
    events: list[dict]
    failures: list[dict]
    max_dynamic_charge_residual: float
    trial_domain_extensions: int


def _jacobian_sparsity(mesh: Mesh, layout: StateLayout) -> object:
    """按物理单元构造局部耦合，并补入两侧CL的全局电荷耦合。"""
    pattern = lil_matrix((layout.size, layout.size), dtype=bool)
    groups: dict[int, list[int]] = {int(cell): [] for cell in range(mesh.n_cells)}
    mapping = {
        "qn": mesh.ionomer_cells, "qn_membrane": mesh.membrane_cells,
        "qf": mesh.membrane_cells, "qv": mesh.pore_cells, "ql": mesh.pore_cells,
        "water_mobile": mesh.pore_cells, "qi": mesh.pore_cells,
        "qh2": mesh.anode_pore_cells, "qo2": mesh.cathode_pore_cells,
        "qn2": mesh.cathode_pore_cells, "energy": np.arange(mesh.n_cells),
    }
    for name, cells in mapping.items():
        if name not in layout.slices:
            continue
        indices = np.arange(layout.slices[name].start, layout.slices[name].stop)
        for index, cell in zip(indices, cells):
            groups[int(cell)].append(int(index))
    for cell, rows in groups.items():
        columns = groups[cell].copy()
        if cell > 0: columns += groups[cell - 1]
        if cell + 1 < mesh.n_cells: columns += groups[cell + 1]
        for row in rows:
            pattern[row, columns] = True
    cl_cells = np.r_[mesh.anode_cl_cells, mesh.cathode_cl_cells]
    cl_indices = [index for cell in cl_cells for index in groups[int(cell)]]
    for row in cl_indices:
        pattern[row, cl_indices] = True
    # 电导率、平衡电势及BV依赖所有离聚物/CL状态；电势消元反馈至CL、膜拖曳及热。
    coupled = [index for cell in mesh.ionomer_cells for index in groups[int(cell)]]
    affected = coupled.copy()
    if 'energy' in layout.slices:
        affected += list(range(layout.slices['energy'].start,layout.slices['energy'].stop))
    for row in affected:
        pattern[row,coupled] = True
    pattern.setdiag(True)
    return pattern.tocsr()


def _equilibrium_difference_steps(y, step, mesh, layout, case, params):
    """移动水差分留在当前分支；只改变差分方向/幅度，不修改物理状态。

    阈值为溶解水容量a与a+饱和蒸气容量b。固定相对扰动跨过b会把
    几乎为零的湿区导数混入干区气体扩散导数，导致隐式Newton停滞。
    """
    state = decode_state(y[:layout.size], mesh, layout, case, params)
    cells = mesh.pore_cells
    nsite = params.membrane_density_kg_per_m3 / params.equivalent_weight_kg_per_mol
    lower = mesh.ionomer_fraction[cells]*nsite*np.minimum(lambda_saturation(state.temperature_k[cells]),16.8)
    available = mesh.porosity[cells]*np.maximum(1-state.saturation_ice[cells],0.)
    csat = saturation_pressure_pa(state.temperature_k[cells])/(params.constants.gas_j_per_mol_k*state.temperature_k[cells])
    upper = lower+available*csat
    columns = np.arange(layout.slices['water_mobile'].start,layout.slices['water_mobile'].stop)
    value = y[columns]; h = step[columns].copy()
    below = value < lower
    middle = (value >= lower) & (value < upper)
    h[below & (value+h >= lower)] *= -1
    # 蒸气带通常远宽于差分步；窄带中取到边界距离的一半避免反向越界。
    reverse = middle & (value+h >= upper) & (value > lower)
    h[reverse] = -np.minimum(h[reverse],.5*(value[reverse]-lower[reverse]))
    at_lower = middle & (value == lower)
    h[at_lower] = np.minimum(h[at_lower],.5*(upper[at_lower]-lower[at_lower]))
    result = step.copy(); result[columns] = h
    return result


def _bounded_difference_jacobian(fun, sparsity, step_selector=None):
    """固定物理尺度的前向差分；零导数列不无限扩大扰动，累计量列保持严格为零。"""
    pattern = sparsity.tocsc()
    rows = [pattern.indices[pattern.indptr[k]:pattern.indptr[k+1]] for k in range(pattern.shape[1])]
    groups = []; occupied = []
    for column in sorted(range(len(rows)),key=lambda k: -len(rows[k])):
        if len(rows[column]) == 0:
            continue
        used = set(rows[column])
        for group, seen in zip(groups,occupied):
            if not used.intersection(seen):
                group.append(column); seen.update(used); break
        else:
            groups.append([column]); occupied.append(used)
    def jacobian(t, y):
        baseline = fun(t,y)
        # mol/m³库存的最小差分尺度1e-3；能量按实际J/m³量级取相对扰动。
        step = np.sqrt(np.finfo(float).eps)*np.maximum(np.abs(y),1e-3)
        if step_selector is not None:
            step = step_selector(y,step)
        data = np.empty_like(pattern.data,dtype=float)
        for group in groups:
            perturbed = y.copy(); perturbed[group] += step[group]
            difference = fun(t,perturbed)-baseline
            for column in group:
                start,stop = pattern.indptr[column:column+2]
                data[start:stop] = difference[rows[column]]/(perturbed[column]-y[column])
        if not np.all(np.isfinite(data)):
            raise FloatingPointError('差分雅可比非有限')
        return csc_matrix((data,pattern.indices,pattern.indptr),shape=pattern.shape)
    return jacobian


def _signals(times: np.ndarray, states: np.ndarray, context: ModelContext) -> tuple[np.ndarray, ...]:
    count = times.size
    voltage = np.empty(count); tmax = np.empty(count)
    lmean = np.empty(count); lmax = np.empty(count)
    imean = np.empty(count); imax = np.empty(count); qlmean = np.empty(count); qlmax = np.empty(count)
    predictor_error = np.empty(count)
    guess = None
    cl = context.mesh.cathode_cl_cells
    weights = context.mesh.dx_m[cl] / np.sum(context.mesh.dx_m[cl])
    for k in range(count):
        state = decode_state(states[:, k], context.mesh, context.layout, context.case, context.params)
        charge = solve_charge_continuation(state, context.mesh, context.case, context.params, context.options, guess)
        predictor = current_constrained_predictor(state, context.mesh, context.case, context.params)
        guess = charge.vector
        voltage[k] = charge.voltage_v; tmax[k] = np.max(state.temperature_k)
        lmean[k] = np.sum(weights * state.lambda_n[cl]); lmax[k] = np.max(state.lambda_n[cl])
        imean[k] = np.sum(weights * state.saturation_ice[cl]); imax[k] = np.max(state.saturation_ice[cl])
        qlmean[k] = np.sum(weights * state.saturation_liquid[cl]); qlmax[k] = np.max(state.saturation_liquid[cl])
        numerator = np.sum(np.abs(charge.j_faraday_a_per_m3[cl] - predictor.j_faraday_a_per_m3[cl]) * context.mesh.dx_m[cl])
        predictor_error[k] = numerator / context.case.current_a_per_m2
    return voltage, tmax, lmean, lmax, imean, imax, qlmean, qlmax, predictor_error


def _absolute_tolerances(layout: StateLayout, options: SolverOptions) -> np.ndarray:
    """按库存单位设置误差尺度；近零冰量精度独立于大库存的相对容差。"""
    if not np.isfinite(options.atol) or options.atol <= 0:
        raise ValueError("默认绝对容差必须为有限正数")
    if not np.isfinite(options.ice_inventory_atol) or options.ice_inventory_atol <= 0:
        raise ValueError("孔隙冰绝对容差必须为有限正数")
    if not np.isfinite(options.liquid_inventory_atol) or options.liquid_inventory_atol <= 0:
        raise ValueError("孔隙液水绝对容差必须为有限正数")
    atol = np.r_[np.full(layout.size, options.atol), 1e-10, 1e-6, 1e-6]
    atol[layout.slices["qi"]] = options.ice_inventory_atol
    if "ql" in layout.slices:
        atol[layout.slices["ql"]] = options.liquid_inventory_atol
    return atol


def simulate(case: CaseConfig, params: ModelParameters | None = None,
             options: SolverOptions | None = None) -> SimulationResult:
    """消去电势代数变量后积分；逐接受步检查，终止于最早连续事件。"""
    params = params or ModelParameters(); options = options or SolverOptions()
    validate_case(case)
    validate_model_parameters(params)
    mesh = build_mesh(params, case.resolution); layout = build_layout(mesh, case)
    y0 = initial_state(mesh, layout, case, params)
    context = ModelContext(mesh, layout, case, params, options)
    z = np.r_[y0, 0., 0., 0.]  # 同步累计水外源、外部热流和体积热源，单位mol/m²、J/m²。
    times = [0.0]; saved = [z.copy()]; t = 0.0
    failures = []; event_info = []; status = 'completed'; message = '达到请求终点'
    if case.t_end_s <= 0 or case.output_interval_s <= 0:
        raise ValueError('积分时长和输出间隔必须为正')
    if options.method not in ('BDF','Radau'):
        raise ValueError('仅支持BDF或Radau')
    integrator_class = BDF if options.method == 'BDF' else Radau

    def fun(time, augmented):
        dy = assemble_rhs(time, augmented[:layout.size], context)
        value = np.r_[dy,context.last_water_external,context.last_energy_external,context.last_energy_source]
        if not np.all(np.isfinite(value)):
            raise FloatingPointError('ODE右端非有限')
        return value

    def margins(time, augmented):
        state = decode_state(augmented[:layout.size],mesh,layout,case,params)
        values = validity_margins(state,mesh,case,options.pore_gas_margin,options.pressure_domain_fraction)
        if any(not np.isfinite(value) for value in values.values()):
            raise FloatingPointError('适用域裕度非有限')
        # 无效状态不计算电压，体积/浓度事件足以定位失效；合法域内增加电压事件。
        if min(values.values()) >= -1e-12:
            charge = solve_charge_continuation(state,mesh,case,params,options,context.charge_guess)
            values['voltage_stop'] = charge.voltage_v-case.voltage_stop_v
        else:
            values['voltage_stop'] = 1.0
        return values

    first = margins(t,z)
    if any(not np.isfinite(v) or v <= 0 for v in first.values()):
        raise ValueError(f'初始状态不在合法域 {first}')
    # 电势消元产生非局部依赖；累计预算行依赖全部物理状态，无反向反馈。
    pattern = lil_matrix((layout.size+3,layout.size+3),dtype=bool)
    pattern[:layout.size,:layout.size] = _jacobian_sparsity(mesh,layout)
    # 预算水/外部焓流仅依赖边界状态；体积热源依赖离聚物及CL状态。
    boundary_cells = {0,mesh.n_cells-1,int(mesh.anode_pore_cells[0]),int(mesh.cathode_pore_cells[-1])}
    for name,cells in {"qn":mesh.ionomer_cells,"qn_membrane":mesh.membrane_cells,
                       "qf":mesh.membrane_cells,"qv":mesh.pore_cells,"ql":mesh.pore_cells,
                       "water_mobile":mesh.pore_cells,"qi":mesh.pore_cells,
                       "qh2":mesh.anode_pore_cells,"qo2":mesh.cathode_pore_cells,
                       "qn2":mesh.cathode_pore_cells,"energy":np.arange(mesh.n_cells)}.items():
        if name not in layout.slices: continue
        for index,cell in zip(range(layout.slices[name].start,layout.slices[name].stop),cells):
            if int(cell) in boundary_cells or (name=='energy' and ('channel' in mesh.layer_name[cell])):
                pattern[layout.size:layout.size+2,index] = True
            if cell in mesh.ionomer_cells:
                pattern[layout.size+2,index] = True
    step_selector = None
    if layout.phase_mode == "ordered_equilibrium" and options.phase_preserving_jacobian:
        step_selector = lambda y,step: _equilibrium_difference_steps(y,step,mesh,layout,case,params)
    jacobian = _bounded_difference_jacobian(fun,pattern.tocsc(),step_selector)
    atol = _absolute_tolerances(layout, options)
    max_step = options.max_step_s; engine = None; retries = 0; next_output = case.output_interval_s
    accepted_steps = 0
    if options.max_accepted_steps < 1:
        raise ValueError("接受步数量上限必须为正")
    while t < case.t_end_s:
        try:
            if engine is None:
                engine = integrator_class(fun,t,z,case.t_end_s,rtol=options.rtol,atol=atol,
                    max_step=max_step,first_step=min(options.first_step_s,case.t_end_s-t,max_step),
                    jac=jacobian)
            detail = engine.step()
            if engine.status == 'failed':
                raise ChargeSolveError(str(detail))
            new_t = float(engine.t); new_z = engine.y.copy(); dense = engine.dense_output()
            new_margins = margins(new_t,new_z)
            crossings = []
            # 检查接受步内子区间，避免更晚事件掩盖较早的适用域越界。
            left_t = t; left_values = first
            for right_t in np.linspace(t,new_t,5)[1:]:
                values = new_margins if right_t==new_t else margins(right_t,dense(right_t))
                for name,left_value in left_values.items():
                    if left_value > 0 and values[name] <= 0:
                        event_t = brentq(lambda time: margins(time,dense(time))[name],left_t,right_t,
                                         xtol=1e-11,rtol=1e-12)
                        crossings.append((float(event_t),name))
                if crossings:
                    # 右端已越界时没有算电压；仍需检查合法区间内是否先达到电压阈值。
                    valid_end = min(crossings)[0]
                    if left_values['voltage_stop'] > 0:
                        def voltage_margin(time):
                            local = decode_state(dense(time)[:layout.size],mesh,layout,case,params)
                            charge = solve_charge_continuation(local,mesh,case,params,options,context.charge_guess)
                            return charge.voltage_v-case.voltage_stop_v
                        inside = np.nextafter(valid_end,left_t)
                        if voltage_margin(inside) <= 0:
                            voltage_time = brentq(voltage_margin,left_t,inside,xtol=1e-11,rtol=1e-12)
                            crossings.append((float(voltage_time),'voltage_stop'))
                    break
                left_t = right_t; left_values = values
            stop_t = min(crossings)[0] if crossings else new_t
            while next_output < stop_t-1e-10:
                times.append(next_output); saved.append(dense(next_output))
                next_output += case.output_interval_s
            if crossings:
                event_t,name = min(crossings)
                endpoint = dense(event_t)
                # 保存事件极限的合法侧，阻止浮点根误差使浓度在门限外被解电势。
                endpoint_t = np.nextafter(event_t,t)
                endpoint = dense(endpoint_t)
                times.append(endpoint_t); saved.append(endpoint)
                status = name; message = f'{event_t:.9g} s首先触发{name}'
                event_info.append({'event':name,'time_s':event_t,'classification':'terminal','detail':'accepted_step_root'})
                break
            t = new_t; z = new_z; first = new_margins; retries = 0
            accepted_steps += 1
            if abs(next_output-t) <= 1e-10:
                times.append(t); saved.append(z.copy()); next_output += case.output_interval_s
            if accepted_steps >= options.max_accepted_steps and engine.status != 'finished':
                status = 'failed'; message = f'接受步达到计算上限{options.max_accepted_steps}，最后有效时刻{t:.12g} s；不是物理停机'
                failures.append({'time_s':t,'error':message,'classification':'step_budget_exhausted'})
                if t > times[-1]+1e-12:
                    times.append(t); saved.append(z.copy())
                break
            if engine.status == 'finished':
                if t > times[-1]+1e-12:
                    times.append(t); saved.append(z.copy())
                break
        except (ChargeSolveError,ValueError,FloatingPointError) as exc:
            failures.append({'time_s':t,'max_step_s':max_step,'error':str(exc)})
            retries += 1
            if retries > options.max_step_retries or max_step < 1e-10:
                status = 'failed'; message = f'在最后有效接受步{t:.9g} s之后无法推进: {exc}'
                if t > times[-1]+1e-12:
                    times.append(t); saved.append(z.copy())
                break
            max_step *= .5; engine = None; context.charge_guess = None
    times = np.asarray(times); augmented = np.column_stack(saved); states = augmented[:layout.size]
    voltage,tmax,lmean,lmax,imean,imax,qlmean,qlmax,predictor_error = _signals(times,states,context)
    return SimulationResult(times,states,voltage,tmax,lmean,lmax,imean,imax,qlmean,qlmax,
        status,message,mesh,layout,context.evaluations,predictor_error,
        augmented[layout.size:],event_info,failures,context.max_charge_residual,context.evaluation_extensions)
