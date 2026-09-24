from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


R_GAS = 8.314
C_MIN_STANDARD = 0.01

_KIND_LINEAR = 0
_KIND_SEI = 1
_KIND_ANODE = 2
_KIND_CATHODE = 3


@dataclass(frozen=True, slots=True)
class PreparedReactionModel:
    model: str
    names: tuple[str, ...]
    parameters: Mapping[str, Mapping[str, Any]]
    kind: np.ndarray
    a: np.ndarray
    ea: np.ndarray
    mass: np.ndarray
    heat: np.ndarray
    onset: np.ndarray
    beta: np.ndarray
    sei_index: int
    anode_index: int


def _readonly(values: np.ndarray) -> np.ndarray:
    values.setflags(write=False)
    return values


def prepare_reaction_model(
    model: str,
    reactions: Mapping[str, Mapping[str, Any]],
    active: Sequence[str],
) -> PreparedReactionModel:
    """Pack immutable reaction metadata once for batched hot-path evaluation."""
    names = tuple(active)
    kind = np.asarray(
        [
            _KIND_SEI if name == "SEI" else
            _KIND_ANODE if name == "anode" else
            _KIND_CATHODE if name in {"cathode1", "cathode2"} else
            _KIND_LINEAR
            for name in names
        ],
        dtype=np.int8,
    )
    return PreparedReactionModel(
        model=model,
        names=names,
        parameters=reactions,
        kind=_readonly(kind),
        a=_readonly(np.asarray([reactions[name]["A"] for name in names], dtype=float)),
        ea=_readonly(np.asarray([reactions[name]["Ea"] for name in names], dtype=float)),
        mass=_readonly(np.asarray([reactions[name]["m"] for name in names], dtype=float)),
        heat=_readonly(np.asarray([reactions[name]["dH"] for name in names], dtype=float)),
        onset=_readonly(np.asarray([reactions[name]["T_onset"] for name in names], dtype=float)),
        beta=_readonly(np.asarray([reactions[name]["smooth_beta"] for name in names], dtype=float)),
        sei_index=names.index("SEI") if "SEI" in names else -1,
        anode_index=names.index("anode") if "anode" in names else -1,
    )


def _safe_exp(exponent: float) -> float:
    """Finite scalar exponential for implicit-solver trial states."""
    return math.exp(max(-700.0, min(700.0, float(exponent))))


def reaction_params(model: str = "standard") -> dict[str, dict[str, Any]]:
    if model == "standard":
        return _standard_reactions()
    raise ValueError(f'Unknown reaction model "{model}".')


def _standard_reactions() -> dict[str, dict[str, float]]:
    beta = 1.0
    return {
        "SEI": {"dH": 257.0, "m": 100.58, "c0": 0.15, "T_onset": 323.15, "A": 1.667e15, "Ea": 135080.0, "n1": 1.0, "n2": 0.0, "K_reg": 5.0, "smooth_beta": beta},
        "anode": {"dH": 1714.0, "m": 100.58, "c0": 1.0, "T_onset": 323.15, "A": 0.035, "A_high": 5.0, "T_switch": 533.15, "Ea": 33000.0, "n1": 1.0, "n2": 0.0, "c_SEI0": 1.0, "smooth_beta": beta, "smooth_beta_switch": 1.0},
        "separator": {"dH": -233.2, "m": 17.6, "c0": 1.0, "T_onset": 393.15, "A": 1.5e50, "Ea": 420000.0, "n1": 1.0, "n2": 0.0, "smooth_beta": beta},
        "electrolyte": {"dH": 800.0, "m": 108.0, "c0": 1.0, "T_onset": 413.15, "A": 3e15, "Ea": 170000.0, "n1": 1.0, "n2": 0.0, "smooth_beta": beta},
        "cathode1": {"dH": 77.0, "m": 179.12, "c0": 0.999, "T_onset": 453.15, "A": 1.75e9, "Ea": 114950.0, "n1": 1.0, "n2": 1.0, "smooth_beta": beta},
        "cathode2": {"dH": 84.0, "m": 179.12, "c0": 0.999, "T_onset": 493.15, "A": 1.077e12, "Ea": 158880.0, "n1": 1.0, "n2": 1.0, "smooth_beta": beta},
    }


def validate_reaction_overrides(model: str, overrides: Mapping[str, Mapping[str, Any]] | None) -> None:
    if not overrides:
        return
    defaults = reaction_params(model)
    for reaction, fields in overrides.items():
        if reaction not in defaults:
            raise ValueError(f'Unknown reaction override group "{reaction}".')
        if not isinstance(fields, Mapping):
            raise ValueError(f"reactions.overrides.{reaction} must be a mapping.")
        for field, value in fields.items():
            if field not in defaults[reaction]:
                raise ValueError(f'Unknown reaction override field "{reaction}.{field}".')
            _validate_reaction_field(reaction, field, value)


def _validate_reaction_field(reaction: str, field: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{reaction}.{field} must be a finite numeric scalar.")
    if field in {"A", "A_high", "Ea", "m", "T_onset", "T_switch", "c_SEI0", "smooth_beta", "smooth_beta_switch"} and value <= 0:
        raise ValueError(f"{reaction}.{field} must be > 0.")
    if field == "dH" and abs(value) > 1e5:
        raise ValueError(f"{reaction}.{field} has an abnormal heat scale; expected J/g.")
    if field == "c0" and not 0 <= value <= 1:
        raise ValueError(f"{reaction}.c0 must be in [0, 1].")


def apply_reaction_overrides(model: str, overrides: Mapping[str, Mapping[str, Any]] | None) -> dict[str, dict[str, Any]]:
    validate_reaction_overrides(model, overrides)
    merged = deepcopy(reaction_params(model))
    if overrides:
        for reaction, fields in overrides.items():
            merged[reaction].update(deepcopy(dict(fields)))
    return merged


def smooth_concentration(c_value: float, c_min: float) -> float:
    x = max(float(c_value), 0.0) / c_min
    if x <= 0:
        return 0.0
    if x < 1:
        return x * x * (3.0 - 2.0 * x)
    return 1.0


def smooth_sigmoid(value: float, onset: float, beta: float) -> float:
    z = -beta * (value - onset)
    if z > 700:
        return 0.0
    if z < -700:
        return 1.0
    return 1.0 / (1.0 + math.exp(z))


def _safe_exp_array(exponent: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(np.asarray(exponent, dtype=float), -700.0, 700.0))


def _smooth_concentration_array(values: np.ndarray, c_min: float) -> np.ndarray:
    x = np.maximum(np.asarray(values, dtype=float), 0.0) / c_min
    return np.where(x <= 0.0, 0.0, np.where(x < 1.0, x * x * (3.0 - 2.0 * x), 1.0))


def _smooth_sigmoid_array(values: np.ndarray, onset: np.ndarray, beta: np.ndarray) -> np.ndarray:
    z = np.clip(-beta * (values - onset), -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(z))


def concentration_rates_batch(
    node_temperature_k: np.ndarray,
    core_temperature_k: np.ndarray,
    concentration: np.ndarray,
    prepared: PreparedReactionModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized equivalent of the scalar source-faithful reaction functions."""
    node_temperature = np.asarray(node_temperature_k, dtype=float).reshape(-1)
    core_temperature = np.asarray(core_temperature_k, dtype=float).reshape(-1)
    concentrations = np.asarray(concentration, dtype=float)
    if concentrations.ndim != 2 or concentrations.shape[0] != node_temperature.size:
        raise ValueError("concentration batch must have shape (n_samples, n_concentrations).")
    if core_temperature.shape != node_temperature.shape:
        raise ValueError("core_temperature_k must match node_temperature_k.")

    c = np.maximum(concentrations, 0.0)
    gate = _smooth_concentration_array(c, C_MIN_STANDARD)
    onset = _smooth_sigmoid_array(
        core_temperature[:, None], prepared.onset[None, :], prepared.beta[None, :]
    )
    arrhenius = _safe_exp_array(-prepared.ea[None, :] / (R_GAS * node_temperature[:, None]))
    rates = -prepared.a[None, :] * c * arrhenius * gate * onset
    cathode = prepared.kind == _KIND_CATHODE
    if np.any(cathode):
        rates[:, cathode] = (
            -prepared.a[cathode][None, :]
            * np.maximum(c[:, cathode] * (1.0 - c[:, cathode]), 0.0)
            * arrhenius[:, cathode]
            * gate[:, cathode]
            * onset[:, cathode]
        )

    anode_rate: np.ndarray | None = None
    if prepared.anode_index >= 0:
        index = prepared.anode_index
        p = prepared.parameters["anode"]
        c_anode = c[:, index]
        c_sei = concentrations[:, prepared.sei_index] if prepared.sei_index >= 0 else 0.0
        switch = _smooth_sigmoid_array(
            core_temperature,
            np.asarray(float(p["T_switch"])),
            np.asarray(float(p["smooth_beta_switch"])),
        )
        a_effective = float(p["A"]) + (float(p["A_high"]) - float(p["A"])) * switch
        anode_rate = (
            a_effective
            * c_anode
            * _safe_exp_array(-float(p["Ea"]) / (R_GAS * node_temperature))
            * _safe_exp_array(-np.asarray(c_sei) / float(p["c_SEI0"]))
            * _smooth_concentration_array(c_anode, C_MIN_STANDARD)
        )
        rates[:, index] = -anode_rate * onset[:, index]

    if prepared.sei_index >= 0:
        index = prepared.sei_index
        decomposition = (
            prepared.a[index]
            * c[:, index]
            * arrhenius[:, index]
            * gate[:, index]
            * onset[:, index]
        )
        regeneration = 0.0
        if anode_rate is not None:
            # Source-faithful MATLAB legacy: regeneration uses the ungated
            # base anode rate and SEI heat is based on abs(net dc/dt).
            regeneration = float(prepared.parameters["SEI"]["K_reg"]) * anode_rate
        rates[:, index] = -decomposition + regeneration
    heat_w = np.sum(np.abs(rates) * prepared.mass[None, :] * prepared.heat[None, :], axis=1)
    return rates, heat_w


def reaction_heat_components_batch(
    node_temperature_k: np.ndarray,
    core_temperature_k: np.ndarray,
    concentration: np.ndarray,
    prepared: PreparedReactionModel,
) -> dict[str, np.ndarray]:
    rates, _ = concentration_rates_batch(
        node_temperature_k, core_temperature_k, concentration, prepared
    )
    return {
        name: np.abs(rates[:, index]) * prepared.mass[index] * prepared.heat[index]
        for index, name in enumerate(prepared.names)
    }


def concentration_rates(
    node_temperature_k: float,
    core_temperature_k: float,
    concentration: np.ndarray,
    reactions: Mapping[str, Mapping[str, Any]],
    active: Sequence[str],
) -> tuple[np.ndarray, float]:
    return standard_rates(node_temperature_k, core_temperature_k, concentration, reactions, active)


def reaction_heat_components(
    node_temperature_k: float,
    core_temperature_k: float,
    concentration: np.ndarray,
    reactions: Mapping[str, Mapping[str, Any]],
    active: Sequence[str],
) -> dict[str, float]:
    """Recompute named reaction heat terms without mutating solver state."""
    rates, _ = standard_rates(node_temperature_k, core_temperature_k, concentration, reactions, active)
    return {
        name: abs(float(rates[index])) * float(reactions[name]["m"]) * float(reactions[name]["dH"])
        for index, name in enumerate(active)
    }


def standard_rates(
    node_temperature_k: float,
    core_temperature_k: float,
    concentration: np.ndarray,
    reactions: Mapping[str, Mapping[str, Any]],
    active: Sequence[str],
) -> tuple[np.ndarray, float]:
    rates = np.zeros(len(active), dtype=float)
    heat_w = 0.0
    offsets = {name: index for index, name in enumerate(active)}
    anode_rate: float | None = None
    for index, name in enumerate(active):
        p = reactions[name]
        c_value = max(float(concentration[index]), 0.0)
        gate = smooth_concentration(c_value, C_MIN_STANDARD)
        onset = smooth_sigmoid(core_temperature_k, float(p["T_onset"]), float(p["smooth_beta"]))
        if name == "SEI":
            decomposition = float(p["A"]) * c_value * _safe_exp(-float(p["Ea"]) / (R_GAS * node_temperature_k)) * gate * onset
            regeneration = 0.0
            if "anode" in offsets:
                if anode_rate is None:
                    anode_rate = _anode_rate(node_temperature_k, core_temperature_k, concentration, reactions, offsets)
                # MATLAB legacy equation: SEI regeneration intentionally uses
                # the ungated base anode rate. Do not apply the anode onset
                # factor here without introducing a new, validated model.
                regeneration = float(p["K_reg"]) * anode_rate
            rates[index] = -decomposition + regeneration
        elif name == "anode":
            if anode_rate is None:
                anode_rate = _anode_rate(node_temperature_k, core_temperature_k, concentration, reactions, offsets)
            rates[index] = -anode_rate * onset
        elif name in {"separator", "electrolyte"}:
            rates[index] = -float(p["A"]) * c_value * _safe_exp(-float(p["Ea"]) / (R_GAS * node_temperature_k)) * gate * onset
        elif name in {"cathode1", "cathode2"}:
            rates[index] = -float(p["A"]) * max(c_value * (1.0 - c_value), 0.0) * _safe_exp(-float(p["Ea"]) / (R_GAS * node_temperature_k)) * gate * onset
        # MATLAB legacy heat definition uses abs(net state rate), including
        # SEI regeneration. It is regression-locked for source compatibility.
        heat_w += abs(rates[index]) * float(p["m"]) * float(p["dH"])
    return rates, heat_w


def _anode_rate(
    node_temperature_k: float,
    core_temperature_k: float,
    concentration: np.ndarray,
    reactions: Mapping[str, Mapping[str, Any]],
    offsets: Mapping[str, int],
) -> float:
    p = reactions["anode"]
    c_anode = max(float(concentration[offsets["anode"]]), 0.0)
    c_sei = float(concentration[offsets["SEI"]]) if "SEI" in offsets else 0.0
    switch = smooth_sigmoid(core_temperature_k, float(p["T_switch"]), float(p["smooth_beta_switch"]))
    a_effective = float(p["A"]) + (float(p["A_high"]) - float(p["A"])) * switch
    return a_effective * c_anode * _safe_exp(-float(p["Ea"]) / (R_GAS * node_temperature_k)) * _safe_exp(-c_sei / float(p["c_SEI0"])) * smooth_concentration(c_anode, C_MIN_STANDARD)
