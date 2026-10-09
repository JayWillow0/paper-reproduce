"""未经热力学验证的组合必须在积分前明确拒绝；历史解码仍可用于诊断。"""
import pytest
from pemfc_coldstart.parameters import CaseConfig,validate_case
import pemfc_coldstart.solver as solver

@pytest.mark.parametrize('phase,thermal',[('ordered_equilibrium','isothermal'),('non_equilibrium','isothermal'),('non_equilibrium','coupled')])
def test_supported_model_combinations(phase,thermal):
 validate_case(CaseConfig(phase_mode=phase,thermal_mode=thermal))

def test_unsupported_combination_rejected_before_mesh(monkeypatch):
 def unexpected_mesh(*args,**kwargs):
  raise AssertionError('Unsupported case reached mesh construction')
 monkeypatch.setattr(solver,'build_mesh',unexpected_mesh)
 with pytest.raises(ValueError,match='ordered_equilibrium \\+ coupled'):
  solver.simulate(CaseConfig(phase_mode='ordered_equilibrium',thermal_mode='coupled'))
