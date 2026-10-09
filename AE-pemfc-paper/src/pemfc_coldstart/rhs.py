"""守恒右端组装。"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .electrochemistry import ChargeSolution, ChargeSolveError, solve_charge_continuation
from .energy import advective_enthalpy_transport, compute_heat_sources
from .mesh import Mesh
from .parameters import CaseConfig, ModelParameters, SolverOptions
from .phase_change import compute_phase_rates, water_sources
from .state import DerivedState, StateLayout, decode_state
from .transport import gas_flux, heat_flux, ionomer_water_flux, liquid_flux


def _divergence(cells: np.ndarray, faces: np.ndarray, mesh: Mesh) -> np.ndarray:
    return (faces[:-1] - faces[1:]) / mesh.dx_m[cells]


@dataclass
class ModelContext:
    mesh: Mesh
    layout: StateLayout
    case: CaseConfig
    params: ModelParameters
    options: SolverOptions
    charge_guess: np.ndarray | None = None
    last_charge: ChargeSolution | None = None
    last_state: DerivedState | None = None
    evaluations: int = 0
    evaluation_extensions: int = 0
    charge_fallbacks: int = 0
    last_water_external: float = 0.0
    last_energy_external: float = 0.0
    last_energy_source: float = 0.0
    max_charge_residual: float = 0.0
    failures: list[str] = field(default_factory=list)


def assemble_rhs(t: float, y: np.ndarray, context: ModelContext) -> np.ndarray:
    del t
    mesh, layout, case, params = context.mesh, context.layout, context.case, context.params
    state = decode_state(y, mesh, layout, case, params)
    if state.domain_violation:
        context.evaluation_extensions += 1
    try:
        charge = solve_charge_continuation(state, mesh, case, params, context.options, context.charge_guess)
    except ChargeSolveError:
        if context.charge_guess is None:
            raise
        context.charge_fallbacks += 1
        charge = solve_charge_continuation(state, mesh, case, params, context.options)
    context.charge_guess = charge.vector
    context.max_charge_residual = max(context.max_charge_residual, charge.residual_relative)
    context.last_charge = charge; context.last_state = state
    context.evaluations += 1
    rates = compute_phase_rates(state, mesh, params)
    sn, sf, sv, sl, si = water_sources(rates)
    dqn = sn.copy(); dqf = sf.copy(); dqv = sv.copy(); dql = sl.copy(); dqi = si.copy()
    dqh2 = np.zeros(mesh.n_cells); dqo2 = np.zeros(mesh.n_cells); dqn2 = np.zeros(mesh.n_cells)
    # 气体和蒸气通量。
    gas_fluxes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for species in ("h2", "o2", "n2", "vapor_anode", "vapor_cathode"):
        cells, faces = gas_flux(state, mesh, case, params, species)
        gas_fluxes[species] = (cells, faces)
        div = _divergence(cells, faces, mesh)
        if species == "h2": dqh2[cells] += div
        elif species == "o2": dqo2[cells] += div
        elif species == "n2": dqn2[cells] += div
        else: dqv[cells] += div
    # 离聚物水与电渗拖曳。
    ion_cells, ion_faces = ionomer_water_flux(state, charge.proton_face_current_a_per_m2, mesh, params)
    dqn[ion_cells] += _divergence(ion_cells, ion_faces, mesh)
    # 毛细液水。
    liq_cells, liq_faces = liquid_flux(state, mesh, params)
    dql[liq_cells] += _divergence(liq_cells, liq_faces, mesh)
    # 法拉第反应计量。
    f = params.constants.faraday_c_per_mol
    ja = charge.j_faraday_a_per_m3[mesh.anode_cl_cells]
    jc = charge.j_faraday_a_per_m3[mesh.cathode_cl_cells]
    dqh2[mesh.anode_cl_cells] -= ja / (2.0 * f)
    dqo2[mesh.cathode_cl_cells] += jc / (4.0 * f)
    water = -jc / (2.0 * f)
    if case.product_mode == "dissolved": dqn[mesh.cathode_cl_cells] += water
    elif case.product_mode == "liquid": dql[mesh.cathode_cl_cells] += water
    else: dqv[mesh.cathode_cl_cells] += water
    dy = np.zeros_like(y)
    if layout.phase_mode == "non_equilibrium":
        layout.view(dy, "qn")[:] = dqn[mesh.ionomer_cells]
        layout.view(dy, "qv")[:] = dqv[mesh.pore_cells]
        layout.view(dy, "ql")[:] = dql[mesh.pore_cells]
    else:
        layout.view(dy, "qn_membrane")[:] = dqn[mesh.membrane_cells]
        # CL的离聚物、气液水共享总库存；非CL只含气液水。
        mobile = dqv[mesh.pore_cells] + dql[mesh.pore_cells]
        cl_mask = np.isin(mesh.pore_cells, np.r_[mesh.anode_cl_cells, mesh.cathode_cl_cells])
        mobile[cl_mask] += dqn[mesh.pore_cells[cl_mask]]
        layout.view(dy, "water_mobile")[:] = mobile
    layout.view(dy, "qf")[:] = dqf[mesh.membrane_cells]
    layout.view(dy, "qi")[:] = dqi[mesh.pore_cells]
    layout.view(dy, "qh2")[:] = dqh2[mesh.anode_pore_cells]
    layout.view(dy, "qo2")[:] = dqo2[mesh.cathode_pore_cells]
    layout.view(dy, "qn2")[:] = dqn2[mesh.cathode_pore_cells]
    context.last_water_external = (gas_fluxes["vapor_anode"][1][0] - gas_fluxes["vapor_cathode"][1][-1]
                                   + liq_faces[0] - liq_faces[-1] + case.current_a_per_m2/(2*f))
    context.last_energy_external = 0.0; context.last_energy_source = 0.0
    if layout.thermal:
        thermal_faces = heat_flux(state, mesh, case, params)
        heat = compute_heat_sources(state, charge, mesh, params)
        advective = advective_enthalpy_transport(state, mesh, case, params, gas_fluxes,
                                            (ion_cells, ion_faces), (liq_cells, liq_faces))
        layout.view(dy, "energy")[:] = ((thermal_faces[:-1] - thermal_faces[1:]) / mesh.dx_m
                                         + advective.source_w_per_m3 + heat.total)
        context.last_energy_external = float(thermal_faces[0]-thermal_faces[-1]+advective.boundary_in_w_per_m2)
        context.last_energy_source = float(np.sum(heat.total*mesh.dx_m))
    return dy
