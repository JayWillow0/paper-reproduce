"""冰库存误差尺度与BDF高阶历史的反例。"""
import numpy as np
import pytest
from dataclasses import replace
from scipy.integrate import BDF
from pemfc_coldstart.parameters import CaseConfig,ModelParameters,SolverOptions
from pemfc_coldstart.mesh import build_mesh
from pemfc_coldstart.state import build_layout
from pemfc_coldstart.solver import _absolute_tolerances

def test_ice_inventory_uses_physical_error_scale():
 m=build_mesh(ModelParameters(),'coarse');l=build_layout(m,CaseConfig())
 a=_absolute_tolerances(l,SolverOptions())
 assert np.all(a[l.slices['qi']]==1e-12)
 assert np.all(a[l.slices['ql']]==1e-12)
 assert np.all(a[l.slices['energy']]==1e-8)
 assert np.array_equal(a[-3:],[1e-10,1e-6,1e-6])

@pytest.mark.parametrize('value',[0.,-1.,float('nan'),float('inf')])
def test_invalid_ice_error_scale_rejected(value):
 m=build_mesh(ModelParameters(),'coarse');l=build_layout(m,CaseConfig())
 with pytest.raises(ValueError):_absolute_tolerances(l,replace(SolverOptions(),ice_inventory_atol=value))

@pytest.mark.parametrize('value',[0.,-1.,float('nan'),float('inf')])
def test_invalid_liquid_error_scale_rejected(value):
 m=build_mesh(ModelParameters(),'coarse');l=build_layout(m,CaseConfig())
 with pytest.raises(ValueError):_absolute_tolerances(l,replace(SolverOptions(),liquid_inventory_atol=value))
