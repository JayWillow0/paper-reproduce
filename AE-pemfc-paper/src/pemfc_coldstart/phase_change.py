"""五态水转换速率。每条边只计算一次，再由计量关系分配。"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .mesh import Mesh
from .parameters import ModelParameters
from .properties import lambda_equilibrium, lambda_saturation, saturation_pressure_pa
from .state import DerivedState


@dataclass
class PhaseRates:
    nv: np.ndarray
    nl: np.ndarray
    nf: np.ndarray
    vl: np.ndarray
    vi: np.ndarray
    li: np.ndarray


def _donor(rate: np.ndarray, donor_forward: np.ndarray, donor_reverse: np.ndarray,
           reference: np.ndarray) -> np.ndarray:
    epsilon = 1.0e-8 * np.maximum(reference, 1.0)
    forward = np.maximum(rate, 0.0) * donor_forward / (donor_forward + epsilon)
    reverse = np.maximum(-rate, 0.0) * donor_reverse / (donor_reverse + epsilon)
    return forward - reverse


def compute_phase_rates(state: DerivedState, mesh: Mesh, params: ModelParameters) -> PhaseRates:
    n = mesh.n_cells; p = mesh.pore_cells; mem = mesh.membrane_cells
    nv = np.zeros(n); nl = np.zeros(n); nf = np.zeros(n)
    vl = np.zeros(n); vi = np.zeros(n); li = np.zeros(n)
    nsite = params.membrane_density_kg_per_m3 / params.equivalent_weight_kg_per_mol
    csat = saturation_pressure_pa(state.temperature_k[p]) / (params.constants.gas_j_per_mol_k * state.temperature_k[p])
    below = np.minimum(1.0, np.maximum(params.pore_freeze_k - state.temperature_k[p], 0.0) / params.freeze_smoothing_k)
    above = np.minimum(1.0, np.maximum(state.temperature_k[p] - params.pore_freeze_k, 0.0) / params.freeze_smoothing_k)
    cond = params.kappa_vl_per_s * (state.gas_fraction[p] * np.maximum(state.vapor_concentration[p] - csat, 0.0)
                                   - state.saturation_liquid[p] * np.maximum(csat - state.vapor_concentration[p], 0.0))
    vl[p] = _donor(cond, state.qv[p], state.ql[p], mesh.porosity[p] * np.maximum(csat, 1.0))
    deposition = params.kappa_vi_per_s * state.gas_fraction[p] * np.maximum(state.vapor_concentration[p] - csat, 0.0) * below
    vi[p] = deposition * state.qv[p] / (state.qv[p] + 1.0e-8 * np.maximum(mesh.porosity[p] * csat, 1.0))
    raw_li = params.kappa_li_per_s * (state.ql[p] * below - state.qi[p] * above)
    li[p] = _donor(raw_li, state.ql[p], state.qi[p], np.maximum(state.ql[p] + state.qi[p], 1.0))
    cl = np.r_[mesh.anode_cl_cells, mesh.cathode_cl_cells]
    eq = lambda_equilibrium(state.water_activity[cl])
    raw_nv = params.kappa_nv_per_s * nsite * (state.lambda_n[cl] - eq) * state.gas_fraction[cl]
    nv[cl] = _donor(raw_nv, state.qn[cl], state.qv[cl], nsite * np.maximum(eq, 1.0))
    cap = np.minimum(lambda_saturation(state.temperature_k[cl]), 16.8)
    raw_nl = params.kappa_nl_per_s * nsite * np.maximum(state.lambda_n[cl] - cap, 0.0) * state.gas_fraction[cl]
    nl[cl] = raw_nl * state.qn[cl] / (state.qn[cl] + 1.0e-8 * nsite * np.maximum(cap, 1.0))
    cap_mem = np.where(state.temperature_k[mem] <= 273.15,
                       lambda_saturation(state.temperature_k[mem]), 16.8)
    cold = state.temperature_k[mem] <= 273.15
    # Part2 §5.2：冻结支κ·b_m[λ_n−λ_sat]_+，回融支不超过min(q_f, 含水缺口)，修复原文A14符号。
    freeze = params.kappa_nf_per_s * nsite * np.maximum(state.lambda_n[mem] - cap_mem, 0.0)
    remelt = params.kappa_nf_per_s * np.minimum(
        state.qf[mem], nsite * np.maximum(cap_mem - state.lambda_n[mem], 0.0))
    nf[mem] = np.where(cold, freeze - remelt, -params.kappa_nf_per_s * state.qf[mem])
    return PhaseRates(nv, nl, nf, vl, vi, li)


def water_sources(rates: PhaseRates) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按(n,f,v,l,i)返回成对守恒源。"""
    sn = -rates.nv - rates.nl - rates.nf
    sf = rates.nf
    sv = rates.nv - rates.vl - rates.vi
    sl = rates.nl + rates.vl - rates.li
    si = rates.vi + rates.li
    return sn, sf, sv, sl, si
