"""恒流条件下的电子/质子电势代数约束。"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .mesh import Mesh
from .parameters import CaseConfig, ModelParameters, SolverOptions
from .properties import proton_conductivity_s_per_m, standard_voltage_v
from .state import DerivedState


class ChargeSolveError(RuntimeError):
    pass


@dataclass
class ChargeSolution:
    phi_s_v: np.ndarray
    phi_m_v: np.ndarray
    j_faraday_a_per_m3: np.ndarray
    eta_v: np.ndarray
    electron_face_current_a_per_m2: np.ndarray
    proton_face_current_a_per_m2: np.ndarray
    voltage_v: float
    residual_relative: float
    iterations: int
    vector: np.ndarray


def _face_conductance(left: int, right: int, conductivity: np.ndarray, mesh: Mesh) -> float:
    return 1.0 / (0.5 * mesh.dx_m[left] / conductivity[left] + 0.5 * mesh.dx_m[right] / conductivity[right])


def solve_charge(state: DerivedState, mesh: Mesh, case: CaseConfig, params: ModelParameters,
                 options: SolverOptions, guess: np.ndarray | None = None,
                 current_override: float | None = None) -> ChargeSolution:
    """稀疏有限体积Newton；面功率形式的残差避免体积缩放病态，线搜索检查实际残差。"""
    e = mesh.electronic_cells; m = mesh.ionomer_cells
    ne = len(e); size = ne + len(m); n = mesh.n_cells
    emap = np.full(n, -1, dtype=int); mmap = np.full(n, -1, dtype=int)
    emap[e] = np.arange(ne); mmap[m] = ne + np.arange(len(m))
    current = case.current_a_per_m2 if current_override is None else current_override
    sigma_m = np.zeros(n)
    sigma_m[m] = mesh.ionomer_fraction[m]**1.5 * proton_conductivity_s_per_m(
        np.maximum(state.lambda_n[m], 0.7), state.temperature_k[m])
    if np.any(sigma_m[m] <= 0):
        raise ChargeSolveError("质子电导率非正")
    ef = np.flatnonzero((emap[:-1] >= 0) & (emap[1:] >= 0))
    mf = m[:-1]
    conduct_e = 1 / (.5*mesh.dx_m[ef]/mesh.electronic_s_per_m[ef]
                     + .5*mesh.dx_m[ef+1]/mesh.electronic_s_per_m[ef+1])
    conduct_m = 1 / (.5*mesh.dx_m[mf]/sigma_m[mf] + .5*mesh.dx_m[mf+1]/sigma_m[mf+1])
    left = np.r_[emap[ef], mmap[mf]]; right = np.r_[emap[ef+1], mmap[mf+1]]
    conduct = np.r_[conduct_e, conduct_m]
    boundary = mesh.electronic_s_per_m[0]/(.5*mesh.dx_m[0])
    cells = np.r_[mesh.anode_cl_cells, mesh.cathode_cl_cells]
    anode = np.arange(len(cells)) < len(mesh.anode_cl_cells)
    es = emap[cells]; im = mmap[cells]; dx = mesh.dx_m[cells]
    c = params.constants; t = state.temperature_k[cells]
    concentration = np.maximum(np.where(anode, state.hydrogen_concentration[cells],
                                        state.oxygen_concentration[cells]), 1e-20)
    pressure = c.gas_j_per_mol_k*t*concentration
    equilibrium = np.where(anode, -c.gas_j_per_mol_k*t/(2*c.faraday_c_per_mol)*np.log(pressure/c.reference_pressure_pa),
        standard_voltage_v(t,params)+c.gas_j_per_mol_k*t/(4*c.faraday_c_per_mol)*np.log(pressure/c.reference_pressure_pa))
    j0 = np.where(anode, params.j0_anode_a_per_m3, params.j0_cathode_a_per_m3) * np.exp(
        -np.where(anode,params.activation_anode_k,params.activation_cathode_k)*(1/t-1/353.15))
    prefactor = state.gas_fraction[cells]*j0*(concentration/params.concentration_ref_mol_per_m3)**np.where(anode,.5,1.)
    if np.any(prefactor < 0) or np.any(~np.isfinite(prefactor)):
        raise ChargeSolveError("动力学前因子非法")
    beta = params.transfer_coefficient*np.where(anode,2.,4.*params.cathode_bv_exponent_multiplier)*c.faraday_c_per_mol/(c.gas_j_per_mol_k*t)
    vector = guess.copy() if guess is not None and len(guess)==size else np.r_[
        np.where(np.char.startswith(mesh.layer_name[e], 'cathode'), 1., -.001),np.full(len(m),-.04)]
    # 分相平移未知量，导电相内部用微小电势差计算面电流，避免1 V附近相减的舍入下限。
    offset = np.zeros(size)
    cathode_nodes = np.flatnonzero(np.char.startswith(mesh.layer_name[e], 'cathode'))
    offset[cathode_nodes] = np.mean(vector[cathode_nodes]); offset[ne:] = np.mean(vector[ne:])
    vector = vector-offset
    shifted_equilibrium = equilibrium-offset[es]+offset[im]
    rows = np.r_[left,left,right,right,0,es,es,im,im]
    cols = np.r_[left,right,left,right,0,es,im,es,im]
    fixed = np.r_[conduct,-conduct,-conduct,conduct,boundary]
    scale = max(abs(current),1.)

    def evaluate(v):
        eta = v[es]-v[im]-shifted_equilibrium
        argument = beta*eta
        if np.max(np.abs(argument)) > 600 or not np.all(np.isfinite(argument)):
            return None
        j = 2*prefactor*np.sinh(argument)
        derivative = 2*prefactor*beta*np.cosh(argument)
        flow = conduct*(v[left]-v[right])
        r = np.zeros(size)
        np.add.at(r,left,flow); np.add.at(r,right,-flow)
        r[0] += boundary*v[0]; r[ne-1] += current
        np.add.at(r,es,j*dx); np.add.at(r,im,-j*dx)
        norm = max(np.max(np.abs(r))/scale,
                   abs(np.sum(j[anode]*dx[anode])-current)/scale,
                   abs(np.sum(j[~anode]*dx[~anode])+current)/scale)
        return r,j,derivative,eta,float(norm)

    for iteration in range(1, options.charge_max_iterations+1):
        # 每次Newton步后重新定中心，使高导电阴极的未知量始终保持微伏量级。
        shift = np.mean(vector[cathode_nodes])
        vector[cathode_nodes] -= shift; offset[cathode_nodes] += shift
        shifted_equilibrium = equilibrium-offset[es]+offset[im]
        evaluated = evaluate(vector)
        if evaluated is None:
            raise ChargeSolveError("BV指数超出数值域，不裁剪反应电流")
        residual,j,derivative,eta,norm = evaluated
        if norm <= options.charge_tolerance:
            break
        d = derivative*dx
        jac = coo_matrix((np.asarray(np.r_[fixed,d,-d,-d,d],dtype=float),(rows,cols)),shape=(size,size)).tocsc()
        step = spsolve(jac,-np.asarray(residual,dtype=float))
        damping = 1.0
        while damping >= 2**-20:
            trial = vector+damping*step
            candidate = evaluate(trial)
            if candidate is not None and (candidate[-1] < norm*(1-1e-4*damping)
                                          or candidate[-1] <= options.charge_tolerance):
                vector = trial
                break
            damping *= .5
        else:
            raise ChargeSolveError(f"电势残差线搜索失败，残差{norm:.3e}")
    else:
        raise ChargeSolveError(f"电势迭代未收敛，残差{norm:.3e}")
    ps = np.full(n,np.nan); pm = np.full(n,np.nan)
    ps[e] = vector[:ne]+offset[:ne]; pm[m] = vector[ne:]+offset[ne:]
    electron = np.zeros(n+1); proton = np.zeros(n+1)
    electron[0] = -boundary*vector[0]; electron[-1] = current
    electron[ef+1] = conduct_e*(vector[emap[ef]]-vector[emap[ef+1]])
    proton[mf+1] = conduct_m*(vector[mmap[mf]]-vector[mmap[mf+1]])
    jf = np.zeros(n); overpotential = np.zeros(n); jf[cells] = j; overpotential[cells] = eta
    voltage = ps[-1]-current*.5*mesh.dx_m[-1]/mesh.electronic_s_per_m[-1]
    return ChargeSolution(ps,pm,jf,overpotential,electron,proton,float(voltage),norm,iteration,(vector+offset).copy())


def solve_charge_continuation(state: DerivedState, mesh: Mesh, case: CaseConfig,
                              params: ModelParameters, options: SolverOptions,
                              guess: np.ndarray | None = None) -> ChargeSolution:
    solution = None
    vector = guess
    if guess is None:
        targets = (0.1, 0.25, 0.5, 1.0)
    else:
        targets = (1.0,)
    for factor in targets:
        solution = solve_charge(state, mesh, case, params, options, vector, factor * case.current_a_per_m2)
        vector = solution.vector
    assert solution is not None
    return solution


def current_constrained_predictor(state: DerivedState, mesh: Mesh, case: CaseConfig,
                                  params: ModelParameters) -> ChargeSolution:
    """历史近似反应分布，仅供误差诊断，不用于时间推进。"""
    c = params.constants; n = mesh.n_cells
    j = np.zeros(n); eta = np.zeros(n)
    for electrode, cells in (("anode", mesh.anode_cl_cells), ("cathode", mesh.cathode_cl_cells)):
        if electrode == "anode":
            concentration = np.maximum(state.hydrogen_concentration[cells], 1e-20)
            j0 = params.j0_anode_a_per_m3 * np.exp(-params.activation_anode_k * (1.0 / state.temperature_k[cells] - 1.0 / 353.15))
            power = 0.5; electrons = 2.0; sign = 1.0
        else:
            concentration = np.maximum(state.oxygen_concentration[cells], 1e-20)
            j0 = params.j0_cathode_a_per_m3 * np.exp(-params.activation_cathode_k * (1.0 / state.temperature_k[cells] - 1.0 / 353.15))
            power = 1.0; electrons = 4.0; sign = -1.0
        kinetic = state.gas_fraction[cells] * j0 * (concentration / params.concentration_ref_mol_per_m3) ** power
        kinetic_integral = np.sum(kinetic * mesh.dx_m[cells])
        if kinetic_integral <= 0.0 or not np.isfinite(kinetic_integral):
            raise ChargeSolveError(f"{electrode}催化层没有正的可用反应孔隙")
        weights = kinetic / kinetic_integral
        j[cells] = sign * case.current_a_per_m2 * weights
        beta = params.transfer_coefficient * electrons * (params.cathode_bv_exponent_multiplier if electrode == "cathode" else 1.0) * c.faraday_c_per_mol / (c.gas_j_per_mol_k * state.temperature_k[cells])
        eta[cells] = np.arcsinh(j[cells] / (2.0 * np.maximum(kinetic, 1e-30))) / beta
    proton = np.zeros(n + 1)
    start, stop = mesh.ionomer_cells[0], mesh.ionomer_cells[-1]
    for cell in range(start, stop + 1):
        proton[cell + 1] = proton[cell] + j[cell] * mesh.dx_m[cell]
    electron = np.zeros(n + 1)
    electron[0] = case.current_a_per_m2
    for cell in range(n):
        electron[cell + 1] = electron[cell] - j[cell] * mesh.dx_m[cell]
        if cell + 1 == mesh.membrane_cells[0]: electron[cell + 1] = 0.0
        if cell == mesh.membrane_cells[-1]: electron[cell + 1] = 0.0
    electron[-1] = case.current_a_per_m2
    phi_s = np.full(n, np.nan); phi_m = np.full(n, np.nan)
    # 预测器电压只供热源日志，正式电压在保存时刻由完整电势约束计算。
    voltage = float(standard_voltage_v(np.mean(state.temperature_k[mesh.cathode_cl_cells]), params)
                    + np.mean(eta[mesh.cathode_cl_cells]) - np.mean(eta[mesh.anode_cl_cells]))
    return ChargeSolution(phi_s, phi_m, j, eta, electron, proton, voltage, 0.0, 0, np.empty(0))
