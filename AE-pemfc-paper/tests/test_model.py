import numpy as np
import pytest
from dataclasses import replace

from pemfc_coldstart.electrochemistry import solve_charge_continuation
from pemfc_coldstart.mesh import build_mesh
from pemfc_coldstart.parameters import (CaseConfig, ModelParameters, SolverOptions,
                                        validate_model_parameters)
from pemfc_coldstart.phase_change import compute_phase_rates, water_sources
from pemfc_coldstart.properties import (lambda_saturation, membrane_diffusivity_m2_per_s,
                                        saturation_pressure_pa, smooth_pore_availability)
from pemfc_coldstart.rhs import ModelContext, assemble_rhs
from pemfc_coldstart.state import build_layout, decode_state, initial_state


def fixture(phase_mode="non_equilibrium", thermal_mode="isothermal"):
    params = ModelParameters()
    case = CaseConfig(resolution="coarse", phase_mode=phase_mode, thermal_mode=thermal_mode, t_end_s=0.01)
    mesh = build_mesh(params, case.resolution)
    layout = build_layout(mesh, case)
    vector = initial_state(mesh, layout, case, params)
    state = decode_state(vector, mesh, layout, case, params)
    return params, case, mesh, layout, vector, state


def test_mesh_and_state_counts():
    params = ModelParameters()
    base = build_mesh(params, "base")
    assert base.n_cells == 108
    non = build_layout(base, CaseConfig(thermal_mode="isothermal"))
    eq = build_layout(base, CaseConfig(phase_mode="ordered_equilibrium", thermal_mode="isothermal"))
    assert non.size == 422
    assert eq.size == 306


def test_paper_property_corrections_at_minus_20c():
    assert np.isclose(saturation_pressure_pa(253.15), 157.69774835421103, rtol=1e-12)
    assert np.isclose(lambda_saturation(253.15), 7.305862128393798, rtol=1e-12)


def test_membrane_diffusivity_follows_equation_11():
    # 式(11)各支在T=303.15 K（f_T=1）的原文系数，λ≤2支无温度因子。
    assert np.isclose(membrane_diffusivity_m2_per_s(1.5, 303.15), 2.69266e-10, rtol=1e-12, atol=1e-22)
    assert np.isclose(membrane_diffusivity_m2_per_s(2.5, 303.15), 1.91e-10, rtol=1e-12, atol=1e-22)
    assert np.isclose(membrane_diffusivity_m2_per_s(3.5, 303.15), 2.2969270e-10, rtol=1e-12, atol=1e-22)
    assert np.isclose(membrane_diffusivity_m2_per_s(5.0, 303.15), 1.489125e-10, rtol=1e-12, atol=1e-22)
    # λ=2支点跳变：左侧常数支，右侧0.87×10⁻¹⁰·f_T。
    assert np.isclose(membrane_diffusivity_m2_per_s(2.0, 303.15), 2.69266e-10, rtol=1e-12, atol=1e-22)
    assert np.isclose(membrane_diffusivity_m2_per_s(2.0 + 1e-9, 303.15), 0.87e-10, rtol=1e-6, atol=1e-22)
    # λ≤2支不随温度缩放；λ>4支在253.15 K按f_T=0.20717缩放。
    assert np.isclose(membrane_diffusivity_m2_per_s(1.5, 253.15), 2.69266e-10, rtol=1e-12, atol=1e-22)
    f_t = np.exp(2416.0 * (1.0 / 303.15 - 1.0 / 253.15))
    assert np.isclose(membrane_diffusivity_m2_per_s(3.4, 253.15),
                      1.0e-10 * (2.9514 * (4 - 3.4) + 1.642454 * (3.4 - 3)) * f_t, rtol=1e-12, atol=1e-22)


def test_membrane_diffusivity_c2_regularization_is_positive_and_local():
    temperature = 253.15
    width = 0.10
    points = np.linspace(1.8, 2.2, 401)
    literal = membrane_diffusivity_m2_per_s(points, temperature, "paper_literal", width)
    smooth = membrane_diffusivity_m2_per_s(points, temperature, "paper_c2_regularized", width)
    assert np.all(np.isfinite(smooth))
    assert np.all(smooth > 0.0)
    outside = (points <= 2.0 - width) | (points >= 2.0 + width)
    assert np.array_equal(smooth[outside], literal[outside])
    assert literal[200] != smooth[200]

    # 五次smootherstep在两端的一、二阶导数为零，连接后继承原分支导数。
    h = 1.0e-4
    for point in (2.0 - width, 2.0 + width):
        values = membrane_diffusivity_m2_per_s(
            point + h * np.arange(-2, 3), temperature,
            "paper_c2_regularized", width,
        )
        slope_left = (values[2] - values[1]) / h
        slope_right = (values[3] - values[2]) / h
        curvature_left = (values[2] - 2.0 * values[1] + values[0]) / h**2
        curvature_right = (values[4] - 2.0 * values[3] + values[2]) / h**2
        assert abs(slope_left - slope_right) < 2.0e-14
        assert abs(curvature_left - curvature_right) < 2.0e-10


@pytest.mark.parametrize("mode", ["unknown", "springer_cubic"])
def test_membrane_diffusivity_rejects_unknown_mode(mode):
    with pytest.raises(ValueError):
        membrane_diffusivity_m2_per_s(2.0, 253.15, mode)


@pytest.mark.parametrize("width", [0.0, -0.1, 1.0, 2.0])
def test_model_parameters_reject_invalid_membrane_transition_width(width):
    with pytest.raises(ValueError):
        validate_model_parameters(replace(
            ModelParameters(), membrane_diffusivity_transition_half_width=width,
        ))


def test_pore_availability_is_bounded_and_c1_at_blockage():
    onset = 0.95
    saturation = np.array([0.0, onset, 0.97, 0.99, 1.0, 1.02])
    available = smooth_pore_availability(saturation, onset)
    assert np.all(available >= 0.0)
    assert np.all(np.diff(available) <= 0.0)
    assert np.isclose(available[0], 1.0)
    assert np.isclose(available[1], 1.0 - onset)
    assert available[-2] == 0.0 and available[-1] == 0.0
    epsilon = 1.0e-6
    left_slope = (smooth_pore_availability(onset, onset)
                  - smooth_pore_availability(onset - epsilon, onset)) / epsilon
    right_slope = (smooth_pore_availability(onset + epsilon, onset)
                   - smooth_pore_availability(onset, onset)) / epsilon
    blocked_slope = (smooth_pore_availability(1.0, onset)
                     - smooth_pore_availability(1.0 - epsilon, onset)) / epsilon
    assert np.isclose(left_slope, -1.0, atol=1.0e-5)
    assert np.isclose(right_slope, -1.0, atol=1.0e-4)
    assert np.isclose(blocked_slope, 0.0, atol=1.0e-3)


def test_phase_matrix_conserves_water():
    params, _, mesh, _, _, state = fixture()
    sources = water_sources(compute_phase_rates(state, mesh, params))
    assert np.max(np.abs(sum(sources))) < 1e-12


def test_charge_constraint_reaches_imposed_current():
    params, case, mesh, _, _, state = fixture()
    charge = solve_charge_continuation(state, mesh, case, params, SolverOptions())
    assert charge.residual_relative < 1e-8
    assert np.isclose(charge.electron_face_current_a_per_m2[-1], case.current_a_per_m2)
    assert np.all(charge.j_faraday_a_per_m3[mesh.anode_cl_cells] > 0)
    assert np.all(charge.j_faraday_a_per_m3[mesh.cathode_cl_cells] < 0)


def test_full_rhs_faraday_water_balance_at_initial_time():
    params, case, mesh, layout, vector, _ = fixture()
    context = ModelContext(mesh, layout, case, params, SolverOptions())
    derivative = assemble_rhs(0.0, vector, context)
    total = np.sum(layout.view(derivative, "qn") * mesh.dx_m[mesh.ionomer_cells])
    total += np.sum(layout.view(derivative, "qf") * mesh.dx_m[mesh.membrane_cells])
    for name in ("qv", "ql", "qi"):
        total += np.sum(layout.view(derivative, name) * mesh.dx_m[mesh.pore_cells])
    expected = case.current_a_per_m2 / (2.0 * params.constants.faraday_c_per_mol)
    assert np.isclose(total, expected, rtol=1e-10, atol=1e-14)


def test_ordered_equilibrium_initial_water_is_preserved():
    params, case, mesh, layout, vector, state = fixture("ordered_equilibrium")
    total_state = np.sum(mesh.dx_m * (state.qn + state.qv + state.ql + state.qi + state.qf))
    mobile = np.sum(layout.view(vector, "water_mobile") * mesh.dx_m[mesh.pore_cells])
    membrane = np.sum(layout.view(vector, "qn_membrane") * mesh.dx_m[mesh.membrane_cells])
    frozen = np.sum(layout.view(vector, "qf") * mesh.dx_m[mesh.membrane_cells])
    assert np.isclose(total_state, mobile + membrane + frozen, rtol=1e-13)


def test_coupled_energy_recovers_initial_temperature():
    params, case, mesh, layout, vector, state = fixture(thermal_mode="coupled")
    assert np.allclose(state.temperature_k, case.temperature_initial_k, atol=1e-11)
