from __future__ import annotations

import math

import numpy as np
import pytest

from moduletr.reactions import (
    R_GAS,
    concentration_rates_batch,
    prepare_reaction_model,
    reaction_params,
    smooth_concentration,
    smooth_sigmoid,
    standard_rates,
)


def test_smooth_functions_boundaries() -> None:
    assert smooth_concentration(-1, 0.01) == 0
    assert smooth_concentration(0, 0.01) == 0
    assert smooth_concentration(0.005, 0.01) == pytest.approx(0.5)
    assert smooth_concentration(0.01, 0.01) == 1
    assert smooth_sigmoid(400, 400, 1) == 0.5
    assert smooth_sigmoid(-1000, 400, 1) == 0
    assert smooth_sigmoid(2000, 400, 1) == 1


def test_standard_model_uses_core_temperature_for_onset() -> None:
    reactions = reaction_params("standard")
    active = ("SEI", "anode")
    c = np.array([0.15, 1.0])
    cold_core, _ = standard_rates(600.0, 300.0, c, reactions, active)
    hot_core, _ = standard_rates(600.0, 600.0, c, reactions, active)
    assert abs(hot_core[1]) > abs(cold_core[1]) * 1e6


def test_standard_sei_legacy_regeneration_and_net_rate_heat_are_locked() -> None:
    reactions = reaction_params("standard")
    active = ("SEI", "anode")
    concentration = np.asarray([0.15, 1.0])
    node_temperature = 330.0
    core_temperature = 320.0
    rates, heat = standard_rates(
        node_temperature, core_temperature, concentration, reactions, active
    )
    anode = reactions["anode"]
    switch = smooth_sigmoid(
        core_temperature, anode["T_switch"], anode["smooth_beta_switch"]
    )
    a_effective = anode["A"] + (anode["A_high"] - anode["A"]) * switch
    anode_base = (
        a_effective
        * concentration[1]
        * math.exp(-anode["Ea"] / (R_GAS * node_temperature))
        * math.exp(-concentration[0] / anode["c_SEI0"])
        * smooth_concentration(concentration[1], 0.01)
    )
    sei = reactions["SEI"]
    decomposition = (
        sei["A"]
        * concentration[0]
        * math.exp(-sei["Ea"] / (R_GAS * node_temperature))
        * smooth_concentration(concentration[0], 0.01)
        * smooth_sigmoid(core_temperature, sei["T_onset"], sei["smooth_beta"])
    )
    expected_sei_rate = -decomposition + sei["K_reg"] * anode_base
    assert rates[0] == pytest.approx(expected_sei_rate, rel=1e-13)
    expected_heat = sum(
        abs(rates[index]) * reactions[name]["m"] * reactions[name]["dH"]
        for index, name in enumerate(active)
    )
    assert heat == pytest.approx(expected_heat, rel=1e-13)


@pytest.mark.parametrize(
    "active",
    [
        ("SEI", "anode", "separator", "electrolyte", "cathode1", "cathode2"),
        ("anode", "SEI", "cathode2"),
        ("separator", "electrolyte"),
    ],
)
def test_standard_batch_matches_scalar_reference(active: tuple[str, ...]) -> None:
    reactions = reaction_params("standard")
    prepared = prepare_reaction_model("standard", reactions, active)
    rng = np.random.default_rng(11)
    node_temperature = rng.uniform(280.0, 900.0, 12)
    core_temperature = rng.uniform(250.0, 950.0, 12)
    concentration = rng.uniform(-0.005, 1.005, (12, len(active)))
    batch_rates, batch_heat = concentration_rates_batch(
        node_temperature, core_temperature, concentration, prepared
    )
    for index in range(len(node_temperature)):
        scalar_rates, scalar_heat = standard_rates(
            float(node_temperature[index]),
            float(core_temperature[index]),
            concentration[index],
            reactions,
            active,
        )
        assert batch_rates[index] == pytest.approx(scalar_rates, rel=1e-12, abs=1e-15)
        assert batch_heat[index] == pytest.approx(scalar_heat, rel=1e-12, abs=1e-15)
