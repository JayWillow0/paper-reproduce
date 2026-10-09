"""守恒库存布局、初态和派生状态。"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .mesh import Mesh
from .parameters import CaseConfig, ModelParameters
from .properties import (lambda_saturation, relative_enthalpies_j_per_mol,
                         saturation_pressure_pa, smooth_pore_availability)


@dataclass(frozen=True)
class StateLayout:
    slices: dict[str, slice]
    size: int
    phase_mode: str
    thermal: bool

    def view(self, y: np.ndarray, name: str) -> np.ndarray:
        return y[self.slices[name]]


@dataclass
class DerivedState:
    temperature_k: np.ndarray
    qn: np.ndarray
    qf: np.ndarray
    qv: np.ndarray
    ql: np.ndarray
    qi: np.ndarray
    qh2: np.ndarray
    qo2: np.ndarray
    qn2: np.ndarray
    lambda_n: np.ndarray
    lambda_f: np.ndarray
    saturation_liquid: np.ndarray
    saturation_ice: np.ndarray
    gas_fraction: np.ndarray
    vapor_concentration: np.ndarray
    hydrogen_concentration: np.ndarray
    oxygen_concentration: np.ndarray
    nitrogen_concentration: np.ndarray
    water_activity: np.ndarray
    geometric_gas_fraction: np.ndarray
    pressure_pa: np.ndarray
    domain_violation: str | None = None


def build_layout(mesh: Mesh, case: CaseConfig) -> StateLayout:
    names: list[tuple[str, int]]
    if case.phase_mode == "non_equilibrium":
        names = [("qn", mesh.ionomer_cells.size), ("qf", mesh.membrane_cells.size),
                 ("qv", mesh.pore_cells.size), ("ql", mesh.pore_cells.size),
                 ("qi", mesh.pore_cells.size)]
    else:
        names = [("qn_membrane", mesh.membrane_cells.size), ("qf", mesh.membrane_cells.size),
                 ("water_mobile", mesh.pore_cells.size), ("qi", mesh.pore_cells.size)]
    names += [("qh2", mesh.anode_pore_cells.size), ("qo2", mesh.cathode_pore_cells.size),
              ("qn2", mesh.cathode_pore_cells.size)]
    if case.thermal_mode == "coupled":
        names.append(("energy", mesh.n_cells))
    start = 0
    slices: dict[str, slice] = {}
    for name, count in names:
        slices[name] = slice(start, start + count)
        start += count
    return StateLayout(slices, start, case.phase_mode, case.thermal_mode == "coupled")


def _water_enthalpy_energy(mesh: Mesh, t: np.ndarray, qn: np.ndarray, qf: np.ndarray,
                           qv: np.ndarray, ql: np.ndarray, qi: np.ndarray,
                           qh2: np.ndarray, qo2: np.ndarray, qn2: np.ndarray,
                           params: ModelParameters) -> np.ndarray:
    h = relative_enthalpies_j_per_mol(t, params)
    e = mesh.dry_heat_capacity_j_per_m3_k * (t - params.constants.reference_temperature_k)
    e += qn * h["n"] + qf * h["f"] + qv * h["v"] + ql * h["l"] + qi * h["i"]
    e += qh2 * h["h2"] + qo2 * h["o2"] + qn2 * h["n2"]
    return e


def initial_state(mesh: Mesh, layout: StateLayout, case: CaseConfig, params: ModelParameters) -> np.ndarray:
    y = np.zeros(layout.size)
    c = params.constants
    nsite = params.membrane_density_kg_per_m3 / params.equivalent_weight_kg_per_mol
    if layout.phase_mode == "non_equilibrium":
        layout.view(y, "qn")[:] = mesh.ionomer_fraction[mesh.ionomer_cells] * nsite * case.lambda_initial
    else:
        layout.view(y, "qn_membrane")[:] = nsite * case.lambda_initial
        cl_pos = np.isin(mesh.pore_cells, np.r_[mesh.anode_cl_cells, mesh.cathode_cl_cells])
        layout.view(y, "water_mobile")[cl_pos] = mesh.ionomer_fraction[mesh.pore_cells[cl_pos]] * nsite * case.lambda_initial
    ch2 = case.pressure_pa / (c.gas_j_per_mol_k * case.temperature_initial_k)
    layout.view(y, "qh2")[:] = mesh.porosity[mesh.anode_pore_cells] * ch2
    layout.view(y, "qo2")[:] = mesh.porosity[mesh.cathode_pore_cells] * case.cathode_oxygen_fraction * ch2
    layout.view(y, "qn2")[:] = mesh.porosity[mesh.cathode_pore_cells] * (1.0 - case.cathode_oxygen_fraction) * ch2
    if layout.thermal:
        state = decode_state(y, mesh, layout, case, params, temperature_override=np.full(mesh.n_cells, case.temperature_initial_k))
        layout.view(y, "energy")[:] = _water_enthalpy_energy(
            mesh, state.temperature_k, state.qn, state.qf, state.qv, state.ql, state.qi,
            state.qh2, state.qo2, state.qn2, params)
    return y


def _partition_ordered(w: np.ndarray, qi: np.ndarray, pore_cells: np.ndarray,
                       mesh: Mesh, temperature_k: np.ndarray, params: ModelParameters) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = mesh.n_cells
    qn = np.zeros(n); qv = np.zeros(n); ql = np.zeros(n)
    p = pore_cells
    nsite = params.membrane_density_kg_per_m3 / params.equivalent_weight_kg_per_mol
    cap = np.minimum(lambda_saturation(temperature_k[p]), 16.8)
    cl = np.isin(p, np.r_[mesh.anode_cl_cells, mesh.cathode_cl_cells])
    qn_p = np.zeros_like(w)
    qn_p[cl] = np.minimum(w[cl], mesh.ionomer_fraction[p[cl]] * nsite * cap[cl])
    remainder = np.maximum(w - qn_p, 0.0)
    si = params.constants.water_kg_per_mol * qi[p] / (mesh.porosity[p] * params.ice_density_kg_per_m3)
    available = mesh.porosity[p] * np.maximum(1.0 - si, 0.0)
    csat = saturation_pressure_pa(temperature_k[p]) / (params.constants.gas_j_per_mol_k * temperature_k[p])
    vapor_cap = available * csat
    denominator = 1.0 - params.constants.water_kg_per_mol * csat / params.liquid_density_kg_per_m3
    liquid = np.where(remainder > vapor_cap, (remainder - vapor_cap) / denominator, 0.0)
    vapor = remainder - liquid
    qn[p] = qn_p; qv[p] = vapor; ql[p] = liquid
    return qn, qv, ql


def decode_state(y: np.ndarray, mesh: Mesh, layout: StateLayout, case: CaseConfig,
                 params: ModelParameters, temperature_override: np.ndarray | None = None) -> DerivedState:
    n = mesh.n_cells; p = mesh.pore_cells; ion = mesh.ionomer_cells; mem = mesh.membrane_cells
    c = params.constants
    qn = np.zeros(n); qf = np.zeros(n); qv = np.zeros(n); ql = np.zeros(n); qi = np.zeros(n)
    qh2 = np.zeros(n); qo2 = np.zeros(n); qn2 = np.zeros(n)
    qi[p] = layout.view(y, "qi")
    qf[mem] = layout.view(y, "qf")
    qh2[mesh.anode_pore_cells] = layout.view(y, "qh2")
    qo2[mesh.cathode_pore_cells] = layout.view(y, "qo2")
    qn2[mesh.cathode_pore_cells] = layout.view(y, "qn2")
    if temperature_override is not None:
        temperature = temperature_override.copy()
    elif not layout.thermal:
        temperature = np.full(n, case.temperature_initial_k)
    else:
        temperature = np.full(n, case.temperature_initial_k)
    if layout.phase_mode == "non_equilibrium":
        qn[ion] = layout.view(y, "qn")
        qv[p] = layout.view(y, "qv"); ql[p] = layout.view(y, "ql")
    else:
        qn[mem] = layout.view(y, "qn_membrane")
        part_qn, qv, ql = _partition_ordered(layout.view(y, "water_mobile"), qi, p, mesh, temperature, params)
        qn += part_qn
    if layout.thermal and temperature_override is None:
        # 相焓与显热均为温度线性函数，固定相分配时可显式反算。
        energy = layout.view(y, "energy")
        h0 = relative_enthalpies_j_per_mol(np.full(n, c.reference_temperature_k), params)
        numerator = energy - (qf * h0["f"] + qv * h0["v"] + qi * h0["i"])
        capacity = mesh.dry_heat_capacity_j_per_m3_k.copy()
        capacity += c.water_kg_per_mol * (params.liquid_cp_j_per_kg_k * (qn + ql)
                    + params.ice_cp_j_per_kg_k * (qf + qi) + params.vapor_cp_j_per_kg_k * qv)
        capacity += c.hydrogen_kg_per_mol * params.hydrogen_cp_j_per_kg_k * qh2
        capacity += c.oxygen_kg_per_mol * params.oxygen_cp_j_per_kg_k * qo2
        capacity += c.nitrogen_kg_per_mol * params.nitrogen_cp_j_per_kg_k * qn2
        temperature = c.reference_temperature_k + numerator / capacity
        if layout.phase_mode == "ordered_equilibrium":
            # 联合满足相分配与能量库存，逐单元向量化有括区间二分；不裁剪温度。
            target = energy
            def partition_energy(temp):
                part, vapor, liquid = _partition_ordered(layout.view(y, "water_mobile"), qi, p, mesh, temp, params)
                part[mem] = layout.view(y, "qn_membrane")
                value = _water_enthalpy_energy(mesh, temp, part, qf, vapor, liquid, qi, qh2, qo2, qn2, params)
                return value, part, vapor, liquid
            low = np.full(n, 223.15); high = np.full(n, 353.15)
            el = partition_energy(low)[0]; eh = partition_energy(high)[0]
            if np.any(target < el) or np.any(target > eh):
                raise ValueError("平衡相分配的能量在温度适用域内无根")
            for _ in range(48):
                mid = (low + high) * 0.5
                em = partition_energy(mid)[0]
                high = np.where(em >= target, mid, high)
                low = np.where(em < target, mid, low)
            temperature = (low + high) * 0.5
            reconstructed, qn, qv, ql = partition_energy(temperature)
            if not np.allclose(reconstructed, target, rtol=1e-11, atol=1e-4):
                raise ValueError("平衡相分配与能量约束不一致")
    nsite = params.membrane_density_kg_per_m3 / params.equivalent_weight_kg_per_mol
    lam = np.zeros(n); lam_f = np.zeros(n)
    lam[ion] = qn[ion] / (mesh.ionomer_fraction[ion] * nsite)
    lam_f[mem] = qf[mem] / nsite
    sl = np.zeros(n); si = np.zeros(n)
    sl[p] = c.water_kg_per_mol * ql[p] / (mesh.porosity[p] * params.liquid_density_kg_per_m3)
    si[p] = c.water_kg_per_mol * qi[p] / (mesh.porosity[p] * params.ice_density_kg_per_m3)
    total_saturation = sl[p] + si[p]
    gas_fraction = np.ones(n)
    gas_fraction[p] = smooth_pore_availability(
        total_saturation, params.pore_blockage_smoothing_onset)
    geometric = np.ones(n); geometric[p] = 1.0 - total_saturation
    # 仅延拓牛顿试探态的除法；接受步由独立适用域门限拦截，不导出该奇异区电压。
    volume = mesh.porosity[p] * np.maximum(geometric[p], 1.0e-12)
    cv = np.zeros(n); ch2 = np.zeros(n); co2 = np.zeros(n); cn2 = np.zeros(n)
    cv[p] = qv[p] / volume; ch2[p] = qh2[p] / volume; co2[p] = qo2[p] / volume; cn2[p] = qn2[p] / volume
    activity = np.zeros(n); activity[p] = cv[p] * c.gas_j_per_mol_k * temperature[p] / saturation_pressure_pa(temperature[p])
    pressure = (cv + ch2 + co2 + cn2) * c.gas_j_per_mol_k * temperature
    state = DerivedState(temperature, qn, qf, qv, ql, qi, qh2, qo2, qn2, lam, lam_f,
                         sl, si, gas_fraction, cv, ch2, co2, cn2, activity, geometric, pressure)
    state.domain_violation = next((name for name, margin in validity_margins(state, mesh, case).items()
                                   if not np.isfinite(margin) or margin < 0), None)
    return state


def validity_margins(state: DerivedState, mesh: Mesh, case: CaseConfig,
                     pore_gas_margin: float = 1e-6,
                     pressure_domain_fraction: float = 0.05) -> dict[str, float]:
    """连续事件裕度，正值合法；负库存容差为1e-9 mol/m³，不改写库存。"""
    p = mesh.pore_cells; ion = mesh.ionomer_cells
    minimum = min(float(np.min(q)) for q in (state.qn, state.qf, state.qv, state.ql,
                                             state.qi, state.qh2, state.qo2, state.qn2))
    reactant = min(float(np.min(state.hydrogen_concentration[mesh.anode_pore_cells])),
                   float(np.min(state.oxygen_concentration[mesh.cathode_pore_cells])))
    return {
        "negative_inventory": minimum + 1e-9,
        "temperature_below_domain": float(np.min(state.temperature_k) - 223.15),
        "temperature_above_domain": float(353.15 - np.max(state.temperature_k)),
        "lambda_below_domain": float(np.min(state.lambda_n[ion]) - 0.7),
        "lambda_above_domain": float(25.0 - np.max(state.lambda_n[ion])),
        "reactant_depleted": reactant - 1e-10,
        "pressure_out_of_domain": float(pressure_domain_fraction - np.max(np.abs(state.pressure_pa[p] / case.pressure_pa - 1.0))),
        "activity_above_domain": float(3.0 - np.max(state.water_activity[p])),
        "pore_near_blocked": float(np.min(state.geometric_gas_fraction[p]) - pore_gas_margin),
    }
