"""相分配单侧导数与避免跨支点的独立回归测试。"""
from dataclasses import replace
import numpy as np
import pytest
from scipy.sparse import eye
from pemfc_coldstart.parameters import CaseConfig,ModelParameters
from pemfc_coldstart.mesh import build_mesh
from pemfc_coldstart.state import build_layout,initial_state,decode_state
from pemfc_coldstart.properties import lambda_saturation,saturation_pressure_pa
from pemfc_coldstart.solver import _equilibrium_difference_steps,_bounded_difference_jacobian

@pytest.mark.parametrize('boundary,side',[('dissolved',-1),('dissolved',1),('vapor',-1),('vapor',1)])
def test_equilibrium_derivative_stays_in_current_phase(boundary,side):
 p=ModelParameters();c=CaseConfig(phase_mode='ordered_equilibrium',thermal_mode='isothermal',resolution='coarse')
 m=build_mesh(p,c.resolution);l=build_layout(m,c);y=initial_state(m,l,c,p)
 cell=m.cathode_cl_cells[3];pos=np.flatnonzero(m.pore_cells==cell)[0];column=l.slices['water_mobile'].start+pos
 nsite=p.membrane_density_kg_per_m3/p.equivalent_weight_kg_per_mol
 a=m.ionomer_fraction[cell]*nsite*lambda_saturation(c.temperature_initial_k)
 csat=saturation_pressure_pa(c.temperature_initial_k)/(p.constants.gas_j_per_mol_k*c.temperature_initial_k)
 b=a+m.porosity[cell]*csat
 y[column]=(a if boundary=='dissolved' else b)+side*1e-6
 before=y.copy();step=np.sqrt(np.finfo(float).eps)*np.maximum(abs(y),1e-3)
 h=_equilibrium_difference_steps(y,step,m,l,c,p)[column]
 def vapor(w):
  v=y.copy();v[column]=w
  return decode_state(v,m,l,c,p).qv[cell]
 derivative=(vapor(y[column]+h)-vapor(y[column]))/h
 if boundary=='dissolved':expected=0. if side<0 else 1.
 else:expected=1. if side<0 else -p.constants.water_kg_per_mol*csat/(p.liquid_density_kg_per_m3-p.constants.water_kg_per_mol*csat)
 assert derivative==pytest.approx(expected,abs=2e-5)
 assert np.array_equal(y,before)
 if boundary=='vapor' and side<0:
  wrong=(vapor(y[column]+step[column])-vapor(y[column]))/step[column]
  assert abs(wrong-expected)>.5

def test_difference_selector_signed_step_is_used():
 fun=lambda t,y: np.minimum(y,1.)
 y=np.array([1.-1e-9])
 j=_bounded_difference_jacobian(fun,eye(1),lambda y,h:-h)(0,y)
 assert j[0,0]==pytest.approx(1.,abs=1e-8)


@pytest.mark.parametrize('temperature',[253.15,270.])
def test_thermal_selected_branch_difference_is_locally_consistent(temperature):
 from pemfc_coldstart.state import _water_enthalpy_energy
 p=ModelParameters();c=CaseConfig(phase_mode='ordered_equilibrium',thermal_mode='coupled',resolution='coarse',temperature_initial_k=temperature)
 m=build_mesh(p,c.resolution);l=build_layout(m,c);y=initial_state(m,l,c,p)
 cell=m.cathode_cl_cells[3];pos=np.flatnonzero(m.pore_cells==cell)[0];column=l.slices['water_mobile'].start+pos
 nsite=p.membrane_density_kg_per_m3/p.equivalent_weight_kg_per_mol
 lower=m.ionomer_fraction[cell]*nsite*min(float(lambda_saturation(temperature)),16.8)
 csat=saturation_pressure_pa(temperature)/(p.constants.gas_j_per_mol_k*temperature)
 y[column]=lower+m.porosity[cell]*csat-1e-6
 state=decode_state(y,m,l,c,p,temperature_override=np.full(m.n_cells,temperature))
 l.view(y,'energy')[:]=_water_enthalpy_energy(m,state.temperature_k,state.qn,state.qf,state.qv,state.ql,state.qi,state.qh2,state.qo2,state.qn2,p)
 step=np.sqrt(np.finfo(float).eps)*np.maximum(abs(y),1e-3)
 h=_equilibrium_difference_steps(y,step,m,l,c,p)[column]
 # 热平衡反算可能选择另一能量根；验证实际选中分支的导数，不预设差分符号。
 base=decode_state(y,m,l,c,p)
 derivatives=[]
 for delta in [h,h/2]:
  trial=y.copy();trial[column]+=delta;s=decode_state(trial,m,l,c,p)
  assert (s.qv[cell]>0)==(base.qv[cell]>0)
  assert (s.ql[cell]>0)==(base.ql[cell]>0)
  derivatives.append((s.qv[cell]-base.qv[cell])/delta)
 assert derivatives[0]==pytest.approx(derivatives[1],rel=2e-4,abs=2e-5)
