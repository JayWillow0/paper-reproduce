"""独立物理恒等式与曾经失败的制造状态；不以自引用期望值替代验证。"""
from dataclasses import replace
import numpy as np
import pytest
from pemfc_coldstart.parameters import ModelParameters, CaseConfig, SolverOptions
from pemfc_coldstart.mesh import build_mesh
from pemfc_coldstart.state import (build_layout, initial_state, decode_state, validity_margins,
                                  _water_enthalpy_energy)
from pemfc_coldstart.energy import compute_heat_sources, advective_enthalpy_transport
from pemfc_coldstart.properties import relative_enthalpies_j_per_mol, lambda_saturation, saturation_pressure_pa
from pemfc_coldstart.electrochemistry import solve_charge_continuation, current_constrained_predictor
from pemfc_coldstart.rhs import ModelContext, assemble_rhs
from pemfc_coldstart.solver import simulate, _jacobian_sparsity


def fixture(thermal='isothermal',phase='non_equilibrium',resolution='coarse'):
    p=ModelParameters(); c=CaseConfig(resolution=resolution,thermal_mode=thermal,phase_mode=phase,t_end_s=.02,output_interval_s=.01)
    m=build_mesh(p,resolution); l=build_layout(m,c); y=initial_state(m,l,c,p)
    return p,c,m,l,y,decode_state(y,m,l,c,p)


@pytest.mark.parametrize('resolution',['coarse','base','fine'])
def test_heat_power_equals_terminal_electrical_loss(resolution):
    p,c,m,l,y,s=fixture(resolution=resolution)
    q=solve_charge_continuation(s,m,c,p,SolverOptions())
    h=compute_heat_sources(s,q,m,p)
    cl=np.r_[m.anode_cl_cells,m.cathode_cl_cells]
    u=q.phi_s_v[cl]-q.phi_m_v[cl]-q.eta_v[cl]
    reversible_power=-np.sum(q.j_faraday_a_per_m3[cl]*u*m.dx_m[cl])
    losses=np.sum((h.activation+h.ohmic)*m.dx_m)
    assert losses==pytest.approx(reversible_power-c.current_a_per_m2*q.voltage_v,rel=1e-8,abs=1e-7)
    assert np.sum(h.ohmic*m.dx_m)>60 # 基准独立数量级，防止面积功率误作体积源。


def test_reservoir_enthalpy_is_external_not_channel_heating():
    p,c,m,l,y,s=fixture('coupled')
    cells=m.cathode_pore_cells; f=np.zeros(len(cells)+1); f[-1]=.003
    zeros=(m.ionomer_cells,np.zeros(len(m.ionomer_cells)+1))
    liquid=(m.pore_cells,np.zeros(len(m.pore_cells)+1))
    transport=advective_enthalpy_transport(s,m,c,p,{'vapor_cathode':(cells,f)},zeros,liquid)
    h=relative_enthalpies_j_per_mol(s.temperature_k,p)['v']
    expected=-.003*h[cells[-1]]
    assert transport.boundary_in_w_per_m2==pytest.approx(expected)
    assert np.sum(transport.source_w_per_m3*m.dx_m)==pytest.approx(expected)
    channel=np.flatnonzero(np.char.find(m.layer_name,'channel')>=0)
    assert np.all(transport.source_w_per_m3[channel]==0)
    # 入流使用邻接流道温度而不是环境常量。
    s.temperature_k[cells[-1]+1]+=5; f[-1]=-.003
    transport=advective_enthalpy_transport(s,m,c,p,{'vapor_cathode':(cells,f)},zeros,liquid)
    expected=.003*relative_enthalpies_j_per_mol(s.temperature_k,p)['v'][cells[-1]+1]
    assert transport.boundary_in_w_per_m2==pytest.approx(expected)


def test_rhs_reaction_is_full_charge_not_normalized_predictor():
    p,c,m,l,y,s=fixture()
    context=ModelContext(m,l,c,p,SolverOptions()); dy=assemble_rhs(0,y,context)
    full=solve_charge_continuation(s,m,c,p,SolverOptions())
    predictor=current_constrained_predictor(s,m,c,p)
    assert context.last_charge.residual_relative<1e-9
    assert np.allclose(context.last_charge.j_faraday_a_per_m3,full.j_faraday_a_per_m3,rtol=1e-9,atol=1e-2)
    # 初态氧均匀，无扩散；逐单元氧消耗必须等于完整局部法拉第源。
    oxygen=l.view(dy,'qo2'); clpos=np.isin(m.cathode_pore_cells,m.cathode_cl_cells)
    assert np.allclose(oxygen[clpos],full.j_faraday_a_per_m3[m.cathode_cl_cells]/(4*p.constants.faraday_c_per_mol),rtol=1e-9)
    assert np.max(abs(full.j_faraday_a_per_m3-predictor.j_faraday_a_per_m3))>1e7
    # 转移系数变化现在应能改变电流空间分布和动态，而非只改保存点电压。
    other=assemble_rhs(0,y,ModelContext(m,l,c,replace(p,transfer_coefficient=.35),SolverOptions()))
    assert np.max(abs(l.view(other,'qo2')-oxygen))>1


def test_ordered_high_saturation_has_unit_activity():
    p,c,m,l,y,s=fixture(phase='ordered_equilibrium')
    cell=m.cathode_cl_cells[0]; pos=np.flatnonzero(m.pore_cells==cell)[0]; eps=m.porosity[cell]
    T=c.temperature_initial_k
    qn=m.ionomer_fraction[cell]*p.membrane_density_kg_per_m3/p.equivalent_weight_kg_per_mol*lambda_saturation(T)
    csat=saturation_pressure_pa(T)/(p.constants.gas_j_per_mol_k*T)
    l.view(y,'water_mobile')[pos]=qn+.98*eps*p.liquid_density_kg_per_m3/p.constants.water_kg_per_mol+.02*eps*csat
    state=decode_state(y,m,l,c,p)
    assert state.water_activity[cell]==pytest.approx(1.,abs=1e-8)
    assert state.geometric_gas_fraction[cell]==pytest.approx(.02)
    assert state.gas_fraction[cell]<state.geometric_gas_fraction[cell]


def test_ordered_temperature_and_partition_satisfy_stored_energy():
    p,c,m,l,y,s=fixture('coupled','ordered_equilibrium')
    l.view(y,'water_mobile')[:]+=200
    T=np.full(m.n_cells,257.)
    at=decode_state(y,m,l,c,p,temperature_override=T)
    energy=_water_enthalpy_energy(m,T,at.qn,at.qf,at.qv,at.ql,at.qi,at.qh2,at.qo2,at.qn2,p)
    l.view(y,'energy')[:]=energy
    actual=decode_state(y,m,l,c,p)
    assert np.allclose(actual.temperature_k,T,rtol=0,atol=1e-9)
    reconstructed=_water_enthalpy_energy(m,actual.temperature_k,actual.qn,actual.qf,actual.qv,actual.ql,actual.qi,
                                        actual.qh2,actual.qo2,actual.qn2,p)
    assert np.allclose(reconstructed,energy,rtol=1e-11,atol=1e-5)


def test_pressure_invalid_before_blockage_is_not_overridden():
    p,c,m,l,y,s=fixture()
    l.view(y,'qn2')[:]*=1.1
    state=decode_state(y,m,l,c,p)
    margins=validity_margins(state,m,c)
    assert margins['pressure_out_of_domain']<0
    assert margins['pore_near_blocked']>0
    assert state.domain_violation=='pressure_out_of_domain'


def test_earliest_pressure_event_is_located_and_not_saved_after(monkeypatch):
    import pemfc_coldstart.solver as module
    p,c,m,l,y,s=fixture()
    # 制造均匀氮气增加，其他状态不变。t=0.05时总压正好增加5%。
    c=replace(c,t_end_s=.1,output_interval_s=.04)
    rate=m.porosity[m.cathode_pore_cells]*c.pressure_pa/(p.constants.gas_j_per_mol_k*c.temperature_initial_k)
    def manufactured(t,y,context):
        dy=np.zeros_like(y); context.layout.view(dy,'qn2')[:]=rate
        return dy
    monkeypatch.setattr(module,'assemble_rhs',manufactured)
    result=module.simulate(c,p,SolverOptions(max_step_s=.03))
    assert result.status=='pressure_out_of_domain'
    assert result.events[0]['time_s']==pytest.approx(.05,abs=1e-8)
    assert result.time_s[-1]<=.05+1e-8
    assert np.all(np.isfinite(result.voltage_v))


def test_thermal_run_budget_and_solver_agreement():
    p,c,m,l,y,s=fixture('coupled')
    b=simulate(c,p,SolverOptions(max_step_s=.005))
    r=simulate(c,p,SolverOptions(method='Radau',max_step_s=.005))
    assert b.status==r.status=='completed'
    assert abs(b.voltage_v[-1]-r.voltage_v[-1])<1e-5
    assert b.max_dynamic_charge_residual<=1e-9
    energy=np.sum(l.view(b.state[:,-1],'energy')*m.dx_m)-np.sum(l.view(b.state[:,0],'energy')*m.dx_m)
    assert energy==pytest.approx(sum(b.cumulative_budgets[1:,-1]),abs=1e-6)


def test_near_blockage_event_keeps_finite_positive_gas_volume(monkeypatch):
    import pemfc_coldstart.solver as module
    p,c,m,l,y,s=fixture()
    c=replace(c,t_end_s=1.1,output_interval_s=.2)
    cell=m.anode_pore_cells[0]; pore_pos=np.flatnonzero(m.pore_cells==cell)[0]
    ice_rate=m.porosity[cell]*p.ice_density_kg_per_m3/p.constants.water_kg_per_mol
    hydrogen_rate=l.view(y,'qh2')[0]
    def manufactured(t,y,context):
        dy=np.zeros_like(y)
        context.layout.view(dy,'qi')[pore_pos]=ice_rate
        context.layout.view(dy,'qh2')[0]=-hydrogen_rate
        return dy
    monkeypatch.setattr(module,'assemble_rhs',manufactured)
    r=module.simulate(c,p,SolverOptions(max_step_s=.1))
    assert r.status=='pore_near_blocked'
    assert r.events[0]['time_s']==pytest.approx(1-1e-6,abs=1e-8)
    last=decode_state(r.state[:,-1],m,l,c,p)
    assert last.geometric_gas_fraction[cell]>0
    assert last.hydrogen_concentration[cell]>0
    assert np.all(np.isfinite(r.voltage_v))


def test_algebraic_failure_keeps_last_accepted_state(monkeypatch):
    import pemfc_coldstart.solver as module
    from pemfc_coldstart.electrochemistry import ChargeSolveError
    p,c,m,l,y,s=fixture()
    c=replace(c,t_end_s=.1,output_interval_s=.001)
    def manufactured(t,y,context):
        if t>.02: raise ChargeSolveError('manufactured failure')
        return np.zeros_like(y)
    monkeypatch.setattr(module,'assemble_rhs',manufactured)
    r=module.simulate(c,p,SolverOptions(max_step_s=.005,max_step_retries=3))
    assert r.status=='failed'
    assert .005<r.time_s[-1]<=.02
    assert len(r.time_s)>1 and len(r.failures)>0


def test_eliminated_charge_jacobian_has_membrane_nonlocal_dependency():
    p,c,m,l,y,s=fixture('coupled')
    pattern=_jacobian_sparsity(m,l)
    row=l.slices['energy'].start+m.membrane_cells[0]
    column=l.slices['qo2'].start+len(m.cathode_cl_cells)-1
    assert pattern[row,column]


def test_bounded_jacobian_constant_columns_never_grow_or_overflow():
    from scipy.sparse import csr_matrix
    from pemfc_coldstart.solver import _bounded_difference_jacobian
    matrix=np.array([[-2.,3.,0.],[0.,-7.,0.],[0.,0.,0.]])
    y=np.array([.2,1e-5,0.])
    jac=_bounded_difference_jacobian(lambda t,y: matrix@y,csr_matrix(np.ones((3,3))))
    with np.errstate(all='raise'):
        for _ in range(400):
            actual=jac(0,y).toarray()
    # 小扰动作用于大基线时，差分误差界由浮点消减决定，不用无量纲固定阈值冒充精度。
    step=np.sqrt(np.finfo(float).eps)*np.maximum(abs(y),1e-3)
    roundoff_bound=8*np.finfo(float).eps*max(abs(matrix@y))/step
    assert np.all(abs(actual-matrix)<=roundoff_bound[None,:])


def test_step_budget_is_numerical_failure_not_physical_event():
    p,c,m,l,y,s=fixture()
    r=simulate(c,p,SolverOptions(max_accepted_steps=2))
    assert r.status=='failed'
    assert 0<r.time_s[-1]<c.t_end_s
    assert r.failures[-1]['classification']=='step_budget_exhausted'
    assert r.events==[]
    assert len(r.time_s)>1
