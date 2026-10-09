"""能量源及热量分解。"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .electrochemistry import ChargeSolution
from .mesh import Mesh
from .parameters import CaseConfig, ModelParameters
from .properties import relative_enthalpies_j_per_mol, standard_voltage_v
from .properties import proton_conductivity_s_per_m
from .state import DerivedState


@dataclass
class HeatSources:
    activation: np.ndarray
    reversible: np.ndarray
    ohmic: np.ndarray
    reaction_reference: np.ndarray

    @property
    def total(self) -> np.ndarray:
        return self.activation + self.reversible + self.ohmic + self.reaction_reference


def _face_dissipation(mesh: Mesh, face_currents: np.ndarray, sigma: np.ndarray,
                      cells: np.ndarray) -> np.ndarray:
    """两个半单元的面功率除以控制体厚度，返回W/m³。"""
    source = np.zeros(mesh.n_cells)
    source[cells] = (face_currents[cells]**2 + face_currents[cells + 1]**2) / (2.0 * sigma[cells])
    return source


@dataclass
class EnthalpyTransport:
    source_w_per_m3: np.ndarray
    boundary_in_w_per_m2: float


def advective_enthalpy_transport(state: DerivedState, mesh: Mesh, case: CaseConfig,
                                 params: ModelParameters,
                                 gas_fluxes: dict[str, tuple[np.ndarray, np.ndarray]],
                                 ionomer_flux: tuple[np.ndarray, np.ndarray],
                                 liquid_flux: tuple[np.ndarray, np.ndarray]) -> EnthalpyTransport:
    """内部焓流成对更新；储库只更新多孔域，不向流道热单元加相反项。"""
    del case
    h = relative_enthalpies_j_per_mol(state.temperature_k, params)
    source = np.zeros(mesh.n_cells)
    boundary = 0.0

    def scatter(cells: np.ndarray, flux: np.ndarray, key: str) -> None:
        nonlocal boundary
        # 跨膜的非连续孔隙索引对应零通量，禁止把两侧孔隙作为相邻物质单元。
        for k in range(1, cells.size):
            left, right = cells[k-1], cells[k]
            if right != left + 1:
                if flux[k] != 0.0:
                    raise ValueError("非相邻物质单元出现焓通量")
                continue
            power = flux[k] * h[key][left if flux[k] >= 0 else right]
            source[left] -= power / mesh.dx_m[left]
            source[right] += power / mesh.dx_m[right]
        for cell, reservoir, value in [(cells[0], cells[0]-1, flux[0]),
                                        (cells[-1], cells[-1]+1, -flux[-1])]:
            if value == 0.0:
                continue
            # 正值为储库入流，储库温度取对应流道热等效层；出流取供体温度。
            upstream = reservoir if value > 0.0 else cell
            power = value * h[key][upstream]
            source[cell] += power / mesh.dx_m[cell]
            boundary += power

    keys = {"h2": "h2", "o2": "o2", "n2": "n2", "vapor_anode": "v", "vapor_cathode": "v"}
    for species, (cells, flux) in gas_fluxes.items():
        scatter(cells, flux, keys[species])
    scatter(*ionomer_flux, "n")
    scatter(*liquid_flux, "l")
    return EnthalpyTransport(source, float(boundary))


def compute_heat_sources(state: DerivedState, charge: ChargeSolution, mesh: Mesh,
                         params: ModelParameters) -> HeatSources:
    n = mesh.n_cells; c = params.constants
    activation = charge.j_faraday_a_per_m3 * charge.eta_v
    reversible = np.zeros(n); ohmic = np.zeros(n); reaction = np.zeros(n)
    dt = 1.0e-3
    for cells, cathode in ((mesh.anode_cl_cells, False), (mesh.cathode_cl_cells, True)):
        t = state.temperature_k[cells]
        if cathode:
            d_u = (standard_voltage_v(t + dt, params) - standard_voltage_v(t - dt, params)) / (2.0 * dt)
            # 固定分压的Nernst氧项温度导数。
            p = np.maximum(c.gas_j_per_mol_k * t * state.oxygen_concentration[cells], 1e-30)
            d_u += c.gas_j_per_mol_k / (4.0 * c.faraday_c_per_mol) * np.log(p / c.reference_pressure_pa)
        else:
            p = np.maximum(c.gas_j_per_mol_k * t * state.hydrogen_concentration[cells], 1e-30)
            d_u = -c.gas_j_per_mol_k / (2.0 * c.faraday_c_per_mol) * np.log(p / c.reference_pressure_pa)
        reversible[cells] = t * charge.j_faraday_a_per_m3[cells] * d_u
    # 欧姆耗散用面电流i_f²R_f并对两个半单元分别记账，积分与离散电势功率一致。
    ohmic += _face_dissipation(mesh, charge.electron_face_current_a_per_m2,
                               mesh.electronic_s_per_m, mesh.electronic_cells)
    proton_sigma = np.zeros(n)
    ionomer_cells = mesh.ionomer_cells
    proton_sigma[ionomer_cells] = mesh.ionomer_fraction[ionomer_cells] ** 1.5 * proton_conductivity_s_per_m(
        np.maximum(state.lambda_n[ionomer_cells], 0.7), state.temperature_k[ionomer_cells])
    ohmic += _face_dissipation(mesh, charge.proton_face_current_a_per_m2,
                               proton_sigma, ionomer_cells)
    h = relative_enthalpies_j_per_mol(state.temperature_k, params)
    ja = charge.j_faraday_a_per_m3[mesh.anode_cl_cells]
    jc = charge.j_faraday_a_per_m3[mesh.cathode_cl_cells]
    reaction[mesh.anode_cl_cells] -= h["h2"][mesh.anode_cl_cells] * ja / (2.0 * c.faraday_c_per_mol)
    reaction[mesh.cathode_cl_cells] -= h["o2"][mesh.cathode_cl_cells] * (-jc) / (4.0 * c.faraday_c_per_mol)
    reaction[mesh.cathode_cl_cells] += h["l"][mesh.cathode_cl_cells] * (-jc) / (2.0 * c.faraday_c_per_mol)
    return HeatSources(activation, reversible, ohmic, reaction)
