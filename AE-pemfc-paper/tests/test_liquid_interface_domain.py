"""物理域内的制造状态验证界面闭合失败不会伪装成零通量。"""
import numpy as np
import pytest
from pemfc_coldstart.parameters import ModelParameters,CaseConfig
from pemfc_coldstart.mesh import build_mesh
from pemfc_coldstart.state import build_layout,initial_state,decode_state,validity_margins
from pemfc_coldstart.transport import liquid_flux,LiquidInterfaceClosureError

def manufactured_interface(ice):
 p=ModelParameters();c=CaseConfig(thermal_mode='isothermal',resolution='coarse')
 m=build_mesh(p,c.resolution);l=build_layout(m,c);y=initial_state(m,l,c,p)
 a=int(m.cathode_cl_cells[-1]);b=a+1;ia=int(np.flatnonzero(m.pore_cells==a)[0]);ib=ia+1
 l.view(y,'ql')[ia]=.05*m.porosity[a]*p.liquid_density_kg_per_m3/p.constants.water_kg_per_mol
 l.view(y,'qi')[ib]=ice*m.porosity[b]*p.ice_density_kg_per_m3/p.constants.water_kg_per_mol
 # 气体库存跟随真实空孔体积，保持给定气相总压；不用压力越界触发本测试。
 for cell,saturation in [(a,.05),(b,ice)]:
  j=int(np.flatnonzero(m.cathode_pore_cells==cell)[0])
  for key in ['qo2','qn2']:l.view(y,key)[j]*=1-saturation
 return decode_state(y,m,l,c,p),m,c,p,ia

def test_missing_pressure_root_is_explicit_failure():
 s,m,c,p,i=manufactured_interface(.99)
 assert min(validity_margins(s,m,c,1e-6).values())>0
 with pytest.raises(LiquidInterfaceClosureError,match='共同压力无根'):
  liquid_flux(s,m,p)

def test_open_downstream_interface_has_outgoing_flux():
 s,m,c,p,i=manufactured_interface(0.)
 cells,flux=liquid_flux(s,m,p)
 assert flux[i+1]>0
 assert np.all(np.isfinite(flux))
