"""气体、膜水、液水和热的公共面通量。"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .mesh import Mesh
from .parameters import CaseConfig, ModelParameters
from .properties import (gas_diffusivity_m2_per_s, liquid_viscosity_pa_s,
                         membrane_diffusivity_m2_per_s, surface_tension_n_per_m)
from .state import DerivedState


class LiquidInterfaceClosureError(ValueError):
    """跨材料界面无法同时满足共同液水压力和通量连续。"""


def resistance_flux(values: np.ndarray, coefficient: np.ndarray, cells: np.ndarray,
                    mesh: Mesh, left_value: float | None = None,
                    right_value: float | None = None) -> np.ndarray:
    """在连续单元链上返回n+1个正向面通量。"""
    m = cells.size
    flux = np.zeros(m + 1)
    if left_value is not None:
        flux[0] = -(values[0] - left_value) / (0.5 * mesh.dx_m[cells[0]] / coefficient[0])
    for k in range(m - 1):
        flux[k + 1] = -(values[k + 1] - values[k]) / (
            0.5 * mesh.dx_m[cells[k]] / coefficient[k] + 0.5 * mesh.dx_m[cells[k + 1]] / coefficient[k + 1])
    if right_value is not None:
        flux[-1] = -(right_value - values[-1]) / (0.5 * mesh.dx_m[cells[-1]] / coefficient[-1])
    return flux


def gas_flux(state: DerivedState, mesh: Mesh, case: CaseConfig, params: ModelParameters,
             species: str) -> tuple[np.ndarray, np.ndarray]:
    c = params.constants
    anode_side = species in ("h2", "vapor_anode")
    reservoir_cell = mesh.anode_pore_cells[0]-1 if anode_side else mesh.cathode_pore_cells[-1]+1
    reservoir_total = case.pressure_pa / (c.gas_j_per_mol_k * state.temperature_k[reservoir_cell])
    if species == "h2":
        cells = mesh.anode_pore_cells; values = state.hydrogen_concentration[cells]
        left, right, key = reservoir_total, None, "h2"
    elif species == "vapor_anode":
        cells = mesh.anode_pore_cells; values = state.vapor_concentration[cells]
        # Huo 2019表2死端阳极：H2维持供应、水蒸气零通量；默认仍为干储库汇。
        left, right, key = (None if params.anode_vapor_boundary == "dead_end" else 0.0), None, species
    elif species == "o2":
        cells = mesh.cathode_pore_cells; values = state.oxygen_concentration[cells]
        left, right, key = None, case.cathode_oxygen_fraction * reservoir_total, "o2"
    elif species == "n2":
        cells = mesh.cathode_pore_cells; values = state.nitrogen_concentration[cells]
        left, right, key = None, (1.0 - case.cathode_oxygen_fraction) * reservoir_total, "n2"
    else:
        cells = mesh.cathode_pore_cells; values = state.vapor_concentration[cells]
        left, right, key = None, 0.0, "vapor_cathode"
    d = gas_diffusivity_m2_per_s(key, state.temperature_k[cells], case.pressure_pa)
    deff = d * mesh.porosity[cells] ** 1.5 * np.maximum(state.gas_fraction[cells], 1e-12) ** 1.5
    return cells, resistance_flux(values, deff, cells, mesh, left, right)


def ionomer_water_flux(state: DerivedState, proton_face_current: np.ndarray, mesh: Mesh,
                       params: ModelParameters) -> tuple[np.ndarray, np.ndarray]:
    cells = mesh.ionomer_cells
    local_current = proton_face_current[cells[0]:cells[-1] + 2]
    diffusivity = membrane_diffusivity_m2_per_s(
        state.lambda_n[cells], state.temperature_k[cells],
        params.membrane_diffusivity_mode,
        params.membrane_diffusivity_transition_half_width,
    )
    coeff = (params.membrane_density_kg_per_m3 / params.equivalent_weight_kg_per_mol
             * mesh.ionomer_fraction[cells] ** 1.5 * diffusivity)
    diff = resistance_flux(state.lambda_n[cells], coeff, cells, mesh)
    upwind_lambda = np.where(local_current >= 0.0,
                             np.r_[state.lambda_n[cells[0]], state.lambda_n[cells]],
                             np.r_[state.lambda_n[cells], state.lambda_n[cells[-1]]])
    drag = (2.5 * upwind_lambda / 22.0) * local_current / params.constants.faraday_c_per_mol
    drag[[0, -1]] = 0.0
    return cells, diff + drag


def heat_flux(state: DerivedState, mesh: Mesh, case: CaseConfig, params: ModelParameters) -> np.ndarray:
    k = mesh.thermal_w_per_m_k
    t = state.temperature_k
    flux = np.zeros(mesh.n_cells + 1)
    h = params.heat_transfer_w_per_m2_k
    flux[0] = -(t[0] - case.temperature_environment_k) / (0.5 * mesh.dx_m[0] / k[0] + 1.0 / h)
    for j in range(mesh.n_cells - 1):
        flux[j + 1] = -(t[j + 1] - t[j]) / (0.5 * mesh.dx_m[j] / k[j] + 0.5 * mesh.dx_m[j + 1] / k[j + 1])
    flux[-1] = -(case.temperature_environment_k - t[-1]) / (0.5 * mesh.dx_m[-1] / k[-1] + 1.0 / h)
    return flux


def _leverett(s: float) -> float:
    return 1.42 * s - 2.12 * s * s + 1.26 * s**3


def liquid_flux(state: DerivedState, mesh: Mesh, params: ModelParameters) -> tuple[np.ndarray, np.ndarray]:
    """逐多孔区计算液水通量。异材界面使用共同液水压力。"""
    p = mesh.pore_cells
    flux = np.zeros(p.size + 1)
    sl = state.saturation_liquid[p]; si = state.saturation_ice[p]
    t = state.temperature_k[p]
    sigma = surface_tension_n_per_m(t)
    angle = np.deg2rad(mesh.contact_angle_deg[p])
    amp = sigma * np.cos(angle) * np.sqrt(mesh.porosity[p] / mesh.permeability_m2[p])
    viscosity = liquid_viscosity_pa_s(t)
    exponent = params.relative_permeability_exponent

    def primitive(k: int, s: float) -> float:
        # 指数3时使用解析原函数，其他指数用于敏感性时数值近似。
        if abs(exponent - 3.0) < 1e-12:
            poly = 1.42 * s**4 / 4.0 - 4.24 * s**5 / 5.0 + 3.78 * s**6 / 6.0
            return -mesh.permeability_m2[p[k]] * amp[k] * poly / viscosity[k]
        grid = np.linspace(0.0, s, 32)
        derivative = 1.42 - 4.24 * grid + 3.78 * grid**2
        return float(np.trapezoid(-mesh.permeability_m2[p[k]] * grid**exponent * amp[k] * derivative / viscosity[k], grid))

    for k in range(p.size - 1):
        if p[k + 1] != p[k] + 1:
            continue
        dl = 0.5 * mesh.dx_m[p[k]]; dr = 0.5 * mesh.dx_m[p[k + 1]]
        if mesh.layer_index[p[k]] == mesh.layer_index[p[k + 1]]:
            flux[k + 1] = (primitive(k, sl[k]) - primitive(k + 1, sl[k + 1])) / (dl + dr)
            continue
        max_l = max(0.0, 1.0 - si[k]); max_r = max(0.0, 1.0 - si[k + 1])
        pressure_max = min(-amp[k] * _leverett(max_l), -amp[k + 1] * _leverett(max_r))
        if pressure_max <= 0.0 or max(sl[k], sl[k + 1]) <= 1.0e-7:
            continue
        def saturation_at(index: int, pressure: float, maximum: float) -> float:
            return brentq(lambda value: -amp[index] * _leverett(value) - pressure, 0.0, maximum)
        def mismatch(pressure: float) -> float:
            left = saturation_at(k, pressure, max_l); right = saturation_at(k + 1, pressure, max_r)
            return ((primitive(k + 1, right) - primitive(k + 1, sl[k + 1])) / dr
                    - (primitive(k, sl[k]) - primitive(k, left)) / dl)
        try:
            root = brentq(mismatch, 0.0, pressure_max)
            left = saturation_at(k, root, max_l); right = saturation_at(k + 1, root, max_r)
            flux[k + 1] = (primitive(k, sl[k]) - primitive(k, left)) / dl
        except ValueError as exc:
            # 零通量也是物理假设，不能用它替代无根。由积分器记录并减步重试。
            raise LiquidInterfaceClosureError(
                f"液水界面共同压力无根: cells={p[k]},{p[k+1]}, "
                f"layers={mesh.layer_name[p[k]]},{mesh.layer_name[p[k+1]]}, "
                f"pressure_bracket_pa=(0,{pressure_max:.12g})"
            ) from exc
    # Part2 §3.1：两侧GDL外侧面对干储库p_l=p_0，对应毛细曲线s*=0、Ψ(0)=0，
    # 只允许液水外排（疏水支Ψ(s)>0保证单向），无储库供液。
    flux[0] = -primitive(0, sl[0]) / (0.5 * mesh.dx_m[p[0]])
    flux[-1] = primitive(p.size - 1, sl[-1]) / (0.5 * mesh.dx_m[p[-1]])
    molar = params.liquid_density_kg_per_m3 / params.constants.water_kg_per_mol * flux
    return p, molar
