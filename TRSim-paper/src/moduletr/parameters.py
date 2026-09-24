from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .config import ModuleConfig


LEGACY_PATHS = {
    "E_short_total": "trigger.trigger_energy_j",
    "E_spontaneous": "trigger.spontaneous_energy_j",
    "M_cell": "cell.mass_per_node_kg",
    "Cp_cell": "cell.specific_heat_j_per_kg_k",
    "M_holder": "holder.mass_kg",
    "Cp_holder": "holder.specific_heat_j_per_kg_k",
    "Q_heater": "trigger.heater_power_w_per_node",
    "T_amb": "ambient_temperature_k",
    "T_init": "initial_temperature_k",
}


def canonical_parameter_path(path: str) -> str:
    if path in LEGACY_PATHS:
        return LEGACY_PATHS[path]
    if path.startswith("reaction_overrides."):
        return "reactions.overrides." + path[len("reaction_overrides.") :]
    return path


def set_config_parameter(config: ModuleConfig, path: str, value: Any) -> ModuleConfig:
    return set_config_parameters(config, [(path, value)])


def set_config_parameters(
    config: ModuleConfig,
    updates: Sequence[tuple[str, Any]],
) -> ModuleConfig:
    """Apply ordered parameter updates with one serialization/validation pass."""
    data = config.to_dict()
    for path, value in updates:
        canonical = canonical_parameter_path(path)
        parts = canonical.split(".")
        cursor: dict[str, Any] = data
        inside_overrides = False
        for part in parts[:-1]:
            if part not in cursor:
                if part == "overrides" or inside_overrides:
                    cursor[part] = {}
                else:
                    raise ValueError(f'Unknown configuration parameter path "{path}".')
            if not isinstance(cursor[part], dict):
                raise ValueError(f'Parameter path "{path}" crosses non-mapping field "{part}".')
            cursor = cursor[part]
            inside_overrides = inside_overrides or part == "overrides"
        leaf = parts[-1]
        if leaf not in cursor and "overrides" not in parts:
            raise ValueError(f'Unknown configuration parameter path "{path}".')
        cursor[leaf] = value
    return ModuleConfig.from_dict(data)


def get_config_parameter(config: ModuleConfig, path: str) -> Any:
    data: Any = config.to_dict()
    for part in canonical_parameter_path(path).split("."):
        if not isinstance(data, dict) or part not in data:
            raise ValueError(f'Unknown configuration parameter path "{path}".')
        data = data[part]
    return data
