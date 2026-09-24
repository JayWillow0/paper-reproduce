from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from .config import (
    CellConfig,
    HolderConfig,
    ModuleConfig,
    ReactionConfig,
    SolverConfig,
    TopologyConfig,
    TriggerConfig,
)


LEGACY_FIELDS = {
    "n_batteries",
    "holders",
    "active_reactions",
    "trigger_type",
    "trigger_battery",
    "heating_stop_mode",
    "t_end",
    "M_cell",
    "Cp_cell",
    "M_holder",
    "Cp_holder",
    "E_short_total",
    "E_spontaneous",
    "Q_heater",
    "T_amb",
    "T_init",
    "reaction_overrides",
    "solver_preset",
    "solver_options",
}


def _name_value_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    if len(args) % 2:
        raise ValueError("Positional compatibility arguments must be name/value pairs.")
    values = dict(kwargs)
    for index in range(0, len(args), 2):
        name = args[index]
        if not isinstance(name, str):
            raise ValueError("Compatibility parameter names must be strings.")
        if name in values:
            raise ValueError(f'Duplicate parameter "{name}".')
        values[name] = args[index + 1]
    return values


def build_config(*args: Any, **kwargs: Any) -> ModuleConfig:
    values = _name_value_args(args, kwargs)
    unknown = set(values) - LEGACY_FIELDS
    if unknown:
        raise ValueError(f"Unknown configuration parameter(s): {', '.join(sorted(unknown))}.")
    active = tuple(values.get("active_reactions", ("SEI", "anode", "separator", "electrolyte", "cathode1", "cathode2")))
    trigger_type = values.get("trigger_type", 1)
    if trigger_type not in (1, 2):
        raise ValueError("trigger_type must be 1 (needle) or 2 (heater).")
    stop_mode = values.get("heating_stop_mode", 2)
    if stop_mode not in (1, 2):
        raise ValueError("heating_stop_mode must be 1 (temperature) or 2 (self-heating rate).")
    solver_options = values.get("solver_options", {})
    if not isinstance(solver_options, Mapping):
        raise ValueError("solver_options must be a mapping.")
    solver_map = {
        "RelTol": "rel_tol",
        "MaxStep": "max_step_s",
        "AbsTol_T": "abs_tol_temperature",
        "AbsTol_c": "abs_tol_concentration",
        "AbsTol_E": "abs_tol_energy",
    }
    unknown_solver = set(solver_options) - set(solver_map)
    if unknown_solver:
        raise ValueError(f"Unknown solver option(s): {', '.join(sorted(unknown_solver))}.")
    solver_kwargs = {solver_map[name]: value for name, value in solver_options.items()}
    config = ModuleConfig(
        topology=TopologyConfig(n_cells=values.get("n_batteries", 5), holders=values.get("holders", "both")),
        cell=CellConfig(
            mass_per_node_kg=values.get("M_cell"),
            specific_heat_j_per_kg_k=values.get("Cp_cell"),
        ),
        holder=HolderConfig(
            mass_kg=values.get("M_holder", 0.474),
            specific_heat_j_per_kg_k=values.get("Cp_holder", 460.0),
        ),
        trigger=TriggerConfig(
            kind="needle" if trigger_type == 1 else "heater",
            cell=values.get("trigger_battery", 1),
            heating_stop_mode="temperature" if stop_mode == 1 else "self_heating_rate",
            trigger_energy_j=values.get("E_short_total", 4e5),
            spontaneous_energy_j=values.get("E_spontaneous", 3.7e5),
            heater_power_w_per_node=values.get("Q_heater", 1000.0),
        ),
        reactions=ReactionConfig(
            model="standard",
            active=active,
            overrides=values.get("reaction_overrides", {}),
        ),
        solver=SolverConfig(preset=values.get("solver_preset", "default"), **solver_kwargs),
        t_end_s=values.get("t_end", 2000.0),
        ambient_temperature_k=values.get("T_amb", 299.15),
        initial_temperature_k=values.get("T_init"),
    ).normalized()
    config.validate()
    return config


def cell_general() -> dict[str, Any]:
    return {
        "name": "cell_general",
        "config_defaults": {},
        "reaction_overrides": {},
        "fit_parameters": [],
        "sensitivity_parameters": [],
        "units": {
            "M_cell": "kg/node",
            "Cp_cell": "J/(kg*K)",
            "E_short_total": "J/cell",
            "E_spontaneous": "J/cell",
            "Q_heater": "W/node",
            "T_amb": "K",
            "T_init": "K",
        },
    }


def build_config_from_profile(profile: Mapping[str, Any], *args: Any, **kwargs: Any) -> ModuleConfig:
    if not isinstance(profile, Mapping):
        raise ValueError("profile must be a mapping.")
    allowed = {"name", "config_defaults", "reaction_overrides", "fit_parameters", "sensitivity_parameters", "units"}
    unknown = set(profile) - allowed
    if unknown:
        raise ValueError(f"Unknown profile field(s): {', '.join(sorted(unknown))}.")
    defaults = dict(profile.get("config_defaults", {}))
    if profile.get("reaction_overrides"):
        defaults["reaction_overrides"] = profile["reaction_overrides"]
    overrides = _name_value_args(args, kwargs)
    defaults.update(overrides)
    return build_config(**defaults)
