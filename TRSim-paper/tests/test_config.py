from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from moduletr import (
    HeaterPowerProfileConfig,
    ModuleConfig,
    ReactionConfig,
    build_config,
    build_config_from_profile,
    cell_general,
    validate_config,
    with_heat_dissipation,
    with_heater_profile,
    with_short_energy_scale,
    with_thermal_barrier,
    without_thermal_barrier,
)
from moduletr.parameters import get_config_parameter, set_config_parameter


def test_config_json_round_trip() -> None:
    config = build_config(
        n_batteries=3,
        holders="none",
        reaction_overrides={"SEI": {"Ea": 140000.0}},
        solver_options={"MaxStep": 0.5},
    )
    payload = config.to_dict()
    json.dumps(payload)
    restored = ModuleConfig.from_dict(payload)
    assert restored == config
    assert restored.solver.max_step_s == 0.5


def test_numpy_reaction_override_is_json_safe() -> None:
    config = ModuleConfig(
        reactions=ReactionConfig(
            model="standard",
            overrides={
                "SEI": {
                    "Ea": np.float64(135080.0),
                    "T_onset": np.float64(323.15),
                }
            },
        )
    )
    payload = config.to_dict()
    json.dumps(payload)
    assert payload["reactions"]["overrides"]["SEI"]["Ea"] == 135080.0


def test_legacy_parameter_paths() -> None:
    config = build_config()
    updated = set_config_parameter(config, "E_short_total", 5e5)
    updated = set_config_parameter(updated, "reaction_overrides.SEI.Ea", 140000.0)
    assert get_config_parameter(updated, "E_short_total") == 5e5
    assert get_config_parameter(updated, "reaction_overrides.SEI.Ea") == 140000.0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_batteries": True}, "positive integer"),
        ({"n_batteries": 2, "trigger_battery": 3}, "must be in"),
        ({"active_reactions": ["anode"]}, "depends on SEI"),
        ({"active_reactions": ["bogus"]}, "Unknown standard"),
        ({"trigger_type": 2, "Q_heater": 0}, "must be > 0"),
        ({"solver_options": {"BadField": 1}}, "Unknown solver"),
    ],
)
def test_invalid_legacy_inputs_fail(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_config(**kwargs)


def test_explicit_cell_thermal_properties_are_kept() -> None:
    config = build_config(
        M_cell=0.6,
        Cp_cell=1150.0,
    )
    assert config.cell.effective() == pytest.approx((0.6, 1150.0))


def test_unknown_nested_field_fails() -> None:
    payload = ModuleConfig().to_dict()
    payload["thermal"]["typo"] = 1.0
    with pytest.raises(ValueError, match="Unknown thermal"):
        ModuleConfig.from_dict(payload)


def test_profile_build_and_explicit_override_precedence() -> None:
    profile = cell_general()
    profile["config_defaults"] = {"n_batteries": 2, "t_end": 10}
    config = build_config_from_profile(profile, n_batteries=3)
    assert config.topology.n_cells == 3
    assert config.t_end_s == 10


def test_paper_study_config_transforms_are_pure() -> None:
    config = ModuleConfig()
    dissipative = with_heat_dissipation(config, 70.0)
    insulated = with_thermal_barrier(config, 0.001, 0.2)
    no_layer = without_thermal_barrier(config)
    scaled = with_short_energy_scale(config, 0.75)
    assert dissipative.thermal.r_convection == pytest.approx(1.0 / 70.0)
    assert insulated.thermal.r_add == pytest.approx(0.005)
    assert no_layer.thermal.r_add == pytest.approx(1e-15)
    assert scaled.trigger.trigger_energy_j == pytest.approx(3.0e5)
    assert scaled.trigger.spontaneous_energy_j == pytest.approx(2.775e5)
    assert config == ModuleConfig()


def test_adiabatic_release_temperature_requires_adiabatic_boundary() -> None:
    config = ModuleConfig()
    with pytest.raises(ValueError, match="requires"):
        replace(
            config,
            thermal=replace(config.thermal, adiabatic_release_temperature_k=900.0),
        ).validate()


def test_heater_profile_json_round_trip_and_pure_transform() -> None:
    base = ModuleConfig()
    configured = with_heater_profile(
        base,
        [0.0, 5.0, 12.0],
        [0.0, 800.0, 100.0],
        cell=2,
        front_fraction=0.4,
        efficiency=0.75,
    )
    assert configured.trigger.heater_profile == HeaterPowerProfileConfig(
        time_s=(0.0, 5.0, 12.0),
        total_power_w=(0.0, 800.0, 100.0),
        front_fraction=0.4,
        efficiency=0.75,
    )
    assert ModuleConfig.from_dict(configured.to_dict()) == configured.normalized()
    assert base == ModuleConfig()


def test_heater_surface_stop_thresholds_round_trip_and_require_heater() -> None:
    configured = with_heater_profile(
        ModuleConfig(),
        [0.0, 5.0],
        [100.0, 100.0],
    )
    configured = replace(
        configured,
        trigger=replace(
            configured.trigger,
            heater_stop_surface_temperature_k=423.15,
            heater_stop_surface_rate_k_per_s=2.0,
        ),
    )
    restored = ModuleConfig.from_dict(configured.to_dict())
    assert restored.trigger.heater_stop_surface_temperature_k == 423.15
    assert restored.trigger.heater_stop_surface_rate_k_per_s == 2.0

    with pytest.raises(ValueError, match="require trigger.kind='heater'"):
        replace(
            ModuleConfig(),
            trigger=replace(
                ModuleConfig().trigger,
                heater_stop_surface_temperature_k=423.15,
            ),
        ).validate()


@pytest.mark.parametrize(
    "profile, message",
    [
        (HeaterPowerProfileConfig((1.0, 2.0), (0.0, 1.0)), "start at 0"),
        (HeaterPowerProfileConfig((0.0, 0.0), (0.0, 1.0)), "strictly increasing"),
        (HeaterPowerProfileConfig((0.0, 1.0), (0.0, 1.0), efficiency=1.1), "efficiency"),
    ],
)
def test_invalid_heater_profiles_fail(
    profile: HeaterPowerProfileConfig,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        profile.validate()
