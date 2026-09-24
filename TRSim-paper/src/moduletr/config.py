from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any, Mapping

import numpy as np


STANDARD_REACTIONS = ("SEI", "anode", "separator", "electrolyte", "cathode1", "cathode2")


def _json_safe(value: Any) -> Any:
    """Return nested builtins suitable for direct JSON serialization."""
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _strict_fields(data: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown {context} field(s): {', '.join(sorted(unknown))}.")


def _finite(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite numeric scalar.")
    value = float(value)
    if positive and value <= 0:
        raise ValueError(f"{name} must be > 0.")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be >= 0.")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


@dataclass(frozen=True, slots=True)
class TopologyConfig:
    n_cells: int = 5
    holders: str = "both"

    def validate(self) -> None:
        _positive_int(self.n_cells, "topology.n_cells")
        if self.holders not in {"both", "none"}:
            raise ValueError("topology.holders must be 'both' or 'none'.")


@dataclass(frozen=True, slots=True)
class CellConfig:
    mass_per_node_kg: float | None = None
    specific_heat_j_per_kg_k: float | None = None

    def effective(self) -> tuple[float, float]:
        mass = 0.36 if self.mass_per_node_kg is None else self.mass_per_node_kg
        cp = 1100.0 if self.specific_heat_j_per_kg_k is None else self.specific_heat_j_per_kg_k
        return _finite(mass, "cell.mass_per_node_kg", positive=True), _finite(
            cp, "cell.specific_heat_j_per_kg_k", positive=True
        )


@dataclass(frozen=True, slots=True)
class HolderConfig:
    mass_kg: float = 0.474
    specific_heat_j_per_kg_k: float = 460.0

    def validate(self) -> None:
        _finite(self.mass_kg, "holder.mass_kg", positive=True)
        _finite(self.specific_heat_j_per_kg_k, "holder.specific_heat_j_per_kg_k", positive=True)


@dataclass(frozen=True, slots=True)
class HeaterPowerProfileConfig:
    time_s: tuple[float, ...]
    total_power_w: tuple[float, ...]
    front_fraction: float = 0.5
    efficiency: float = 1.0

    def validate(self) -> None:
        if len(self.time_s) < 2 or len(self.time_s) != len(self.total_power_w):
            raise ValueError(
                "trigger.heater_profile time_s and total_power_w must have equal length >= 2."
            )
        times = tuple(
            _finite(value, f"trigger.heater_profile.time_s[{index}]", nonnegative=True)
            for index, value in enumerate(self.time_s)
        )
        if times[0] != 0.0:
            raise ValueError("trigger.heater_profile.time_s must start at 0 s.")
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("trigger.heater_profile.time_s must be strictly increasing.")
        for index, value in enumerate(self.total_power_w):
            _finite(value, f"trigger.heater_profile.total_power_w[{index}]", nonnegative=True)
        fraction = _finite(self.front_fraction, "trigger.heater_profile.front_fraction")
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("trigger.heater_profile.front_fraction must be in [0, 1].")
        efficiency = _finite(self.efficiency, "trigger.heater_profile.efficiency")
        if not 0.0 < efficiency <= 1.0:
            raise ValueError("trigger.heater_profile.efficiency must be in (0, 1].")


@dataclass(frozen=True, slots=True)
class TriggerConfig:
    kind: str = "needle"
    cell: int = 1
    heating_stop_mode: str = "self_heating_rate"
    trigger_energy_j: float = 4.0e5
    spontaneous_energy_j: float = 3.7e5
    heater_power_w_per_node: float = 1000.0
    release_duration_s: float = 10.0
    tr_threshold_k: float = 260.0 + 273.15
    self_heating_stop_rate_k_per_s: float = 1.0
    kinetic_tr_rate_k_per_s: float = 1.0
    heater_stop_surface_temperature_k: float | None = None
    heater_stop_surface_rate_k_per_s: float | None = None
    spontaneous_short_enabled: bool = True
    heater_profile: HeaterPowerProfileConfig | None = None

    def validate(self, n_cells: int) -> None:
        if self.kind not in {"needle", "heater"}:
            raise ValueError("trigger.kind must be 'needle' or 'heater'.")
        _positive_int(self.cell, "trigger.cell")
        if self.cell > n_cells:
            raise ValueError(f"trigger.cell must be in [1, {n_cells}].")
        if self.heating_stop_mode not in {"temperature", "self_heating_rate", "profile"}:
            raise ValueError(
                "trigger.heating_stop_mode must be 'temperature', 'self_heating_rate', or 'profile'."
            )
        _finite(self.trigger_energy_j, "trigger.trigger_energy_j", positive=True)
        _finite(self.spontaneous_energy_j, "trigger.spontaneous_energy_j", positive=True)
        _finite(self.release_duration_s, "trigger.release_duration_s", positive=True)
        _finite(self.tr_threshold_k, "trigger.tr_threshold_k", positive=True)
        _finite(self.self_heating_stop_rate_k_per_s, "trigger.self_heating_stop_rate_k_per_s", positive=True)
        _finite(self.kinetic_tr_rate_k_per_s, "trigger.kinetic_tr_rate_k_per_s", positive=True)
        if self.heater_stop_surface_temperature_k is not None:
            _finite(
                self.heater_stop_surface_temperature_k,
                "trigger.heater_stop_surface_temperature_k",
                positive=True,
            )
        if self.heater_stop_surface_rate_k_per_s is not None:
            _finite(
                self.heater_stop_surface_rate_k_per_s,
                "trigger.heater_stop_surface_rate_k_per_s",
                positive=True,
            )
        if (
            self.kind != "heater"
            and (
                self.heater_stop_surface_temperature_k is not None
                or self.heater_stop_surface_rate_k_per_s is not None
            )
        ):
            raise ValueError("Heater surface stop thresholds require trigger.kind='heater'.")
        _finite(self.heater_power_w_per_node, "trigger.heater_power_w_per_node")
        if not isinstance(self.spontaneous_short_enabled, bool):
            raise ValueError("trigger.spontaneous_short_enabled must be a bool.")
        if self.kind == "heater" and self.heater_power_w_per_node <= 0:
            raise ValueError("trigger.heater_power_w_per_node must be > 0 for heater triggering.")
        if self.heater_profile is not None:
            if not isinstance(self.heater_profile, HeaterPowerProfileConfig):
                raise ValueError("trigger.heater_profile must be a HeaterPowerProfileConfig.")
            self.heater_profile.validate()
            if self.kind != "heater" or self.heating_stop_mode != "profile":
                raise ValueError(
                    "trigger.heater_profile requires trigger.kind='heater' and heating_stop_mode='profile'."
                )
        elif self.heating_stop_mode == "profile":
            raise ValueError("trigger.heating_stop_mode='profile' requires trigger.heater_profile.")


@dataclass(frozen=True, slots=True)
class ThermalNetworkConfig:
    r_holder: float = 0.01 / 40.0
    r_ren: float = 0.0015 / 0.08
    r_shell: float = 0.001 / 238.0
    r_jr_12: float = 0.006 / 1.5
    r_jr_34: float = 0.070 / 30.0
    r_jr_56: float = 0.041 / 30.0
    r_air: float = 0.005 / 0.0321
    r_cc: float = 0.0025 / 0.25
    r_contact: float = 1.0 / 420.0
    r_ap_out: float = 1.0 / 195.0
    r_ap_in: float = 1.0 / 1000.0
    r_convection: float = 1.0 / 25.0
    r_add: float = 0.001 / 1.0e8
    area_12_m2: float = 0.01354
    area_34_m2: float = 0.001202
    area_56_m2: float = 0.001952
    area_holder_m2: float = 0.05
    adiabatic_boundary: bool = False
    adiabatic_release_temperature_k: float | None = None

    def validate(self) -> None:
        values = asdict(self)
        adiabatic = values.pop("adiabatic_boundary")
        release_temperature = values.pop("adiabatic_release_temperature_k")
        if not isinstance(adiabatic, bool):
            raise ValueError("thermal.adiabatic_boundary must be a bool.")
        if release_temperature is not None:
            _finite(
                release_temperature,
                "thermal.adiabatic_release_temperature_k",
                positive=True,
            )
            if not adiabatic:
                raise ValueError(
                    "thermal.adiabatic_release_temperature_k requires "
                    "thermal.adiabatic_boundary=True."
                )
        for name, value in values.items():
            _finite(value, f"thermal.{name}", positive=True)


@dataclass(frozen=True, slots=True)
class ReactionConfig:
    model: str = "standard"
    active: tuple[str, ...] = STANDARD_REACTIONS
    overrides: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def normalized(self) -> "ReactionConfig":
        model = self.model.lower()
        active = tuple(str(item) for item in self.active)
        return replace(self, model=model, active=active, overrides={k: dict(v) for k, v in self.overrides.items()})

    def validate(self) -> None:
        cfg = self.normalized()
        if cfg.model != "standard":
            raise ValueError("reactions.model must be 'standard'.")
        if not cfg.active:
            raise ValueError("reactions.active cannot be empty.")
        unknown = set(cfg.active) - set(STANDARD_REACTIONS)
        if unknown:
            raise ValueError(f"Unknown standard reaction(s): {', '.join(sorted(unknown))}.")
        if "anode" in cfg.active and "SEI" not in cfg.active:
            raise ValueError("The anode reaction depends on SEI; enable both.")


SOLVER_PRESETS: dict[str, dict[str, float]] = {
    "fast": {"rel_tol": 3e-3, "max_step_s": 5.0, "abs_tol_temperature": 3e-3, "abs_tol_concentration": 3e-6, "abs_tol_energy": 1e-2},
    "default": {"rel_tol": 1e-3, "max_step_s": 1.0, "abs_tol_temperature": 1e-3, "abs_tol_concentration": 1e-6, "abs_tol_energy": 1e-3},
    "calibration": {"rel_tol": 3e-4, "max_step_s": 0.2, "abs_tol_temperature": 3e-4, "abs_tol_concentration": 3e-7, "abs_tol_energy": 1e-4},
}


@dataclass(frozen=True, slots=True)
class SolverConfig:
    method: str = "auto"
    preset: str = "default"
    rel_tol: float | None = None
    max_step_s: float | None = None
    abs_tol_temperature: float | None = None
    abs_tol_concentration: float | None = None
    abs_tol_energy: float | None = None
    max_segments: int = 10_000

    def resolved(self) -> "SolverConfig":
        if self.preset not in SOLVER_PRESETS:
            raise ValueError(f"solver.preset must be one of: {', '.join(SOLVER_PRESETS)}.")
        values = SOLVER_PRESETS[self.preset]
        return replace(
            self,
            rel_tol=values["rel_tol"] if self.rel_tol is None else self.rel_tol,
            max_step_s=values["max_step_s"] if self.max_step_s is None else self.max_step_s,
            abs_tol_temperature=values["abs_tol_temperature"] if self.abs_tol_temperature is None else self.abs_tol_temperature,
            abs_tol_concentration=values["abs_tol_concentration"] if self.abs_tol_concentration is None else self.abs_tol_concentration,
            abs_tol_energy=values["abs_tol_energy"] if self.abs_tol_energy is None else self.abs_tol_energy,
        )

    def validate(self) -> None:
        cfg = self.resolved()
        if cfg.method not in {"auto", "BDF", "Radau", "LSODA"}:
            raise ValueError("solver.method must be 'auto', 'BDF', 'Radau', or 'LSODA'.")
        for name in ("rel_tol", "max_step_s", "abs_tol_temperature", "abs_tol_concentration", "abs_tol_energy"):
            _finite(getattr(cfg, name), f"solver.{name}", positive=True)
        _positive_int(cfg.max_segments, "solver.max_segments")


@dataclass(frozen=True, slots=True)
class ModuleConfig:
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    cell: CellConfig = field(default_factory=CellConfig)
    holder: HolderConfig = field(default_factory=HolderConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    thermal: ThermalNetworkConfig = field(default_factory=ThermalNetworkConfig)
    reactions: ReactionConfig = field(default_factory=ReactionConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    t_end_s: float = 2000.0
    ambient_temperature_k: float = 26.0 + 273.15
    initial_temperature_k: float | None = None

    def normalized(self) -> "ModuleConfig":
        return replace(self, reactions=self.reactions.normalized(), solver=self.solver.resolved())

    def validate(self) -> None:
        cfg = self.normalized()
        cfg.topology.validate()
        cfg.holder.validate()
        cfg.trigger.validate(cfg.topology.n_cells)
        cfg.thermal.validate()
        cfg.reactions.validate()
        cfg.solver.validate()
        cfg.cell.effective()
        _finite(cfg.t_end_s, "t_end_s", positive=True)
        _finite(cfg.ambient_temperature_k, "ambient_temperature_k", positive=True)
        if cfg.initial_temperature_k is not None:
            _finite(cfg.initial_temperature_k, "initial_temperature_k", positive=True)
        from .reactions import validate_reaction_overrides

        validate_reaction_overrides(cfg.reactions.model, cfg.reactions.overrides)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self.normalized()))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModuleConfig":
        if not isinstance(data, Mapping):
            raise ValueError("ModuleConfig input must be a mapping.")
        allowed = {
            "topology",
            "cell",
            "holder",
            "trigger",
            "thermal",
            "reactions",
            "solver",
            "t_end_s",
            "ambient_temperature_k",
            "initial_temperature_k",
        }
        _strict_fields(data, allowed, "config")

        def nested(name: str, typ: type, *, transform: Any = None) -> Any:
            raw = data.get(name, {})
            if not isinstance(raw, Mapping):
                raise ValueError(f"{name} must be a mapping.")
            fields = set(typ.__dataclass_fields__)
            _strict_fields(raw, fields, name)
            values = dict(raw)
            if transform:
                values = transform(values)
            return typ(**values)

        reactions = nested(
            "reactions",
            ReactionConfig,
            transform=lambda value: {**value, **({"active": tuple(value["active"])} if "active" in value else {})},
        )
        config = cls(
            topology=nested("topology", TopologyConfig),
            cell=nested("cell", CellConfig),
            holder=nested("holder", HolderConfig),
            trigger=nested(
                "trigger",
                TriggerConfig,
                transform=lambda value: {
                    **value,
                    **(
                        {
                            "heater_profile": HeaterPowerProfileConfig(
                                **{
                                    **value["heater_profile"],
                                    "time_s": tuple(value["heater_profile"]["time_s"]),
                                    "total_power_w": tuple(value["heater_profile"]["total_power_w"]),
                                }
                            )
                        }
                        if isinstance(value.get("heater_profile"), Mapping)
                        else {}
                    ),
                },
            ),
            thermal=nested("thermal", ThermalNetworkConfig),
            reactions=reactions,
            solver=nested("solver", SolverConfig),
            t_end_s=data.get("t_end_s", 2000.0),
            ambient_temperature_k=data.get("ambient_temperature_k", 299.15),
            initial_temperature_k=data.get("initial_temperature_k"),
        ).normalized()
        config.validate()
        return config


def validate_config(config: ModuleConfig) -> ModuleConfig:
    if not isinstance(config, ModuleConfig):
        raise TypeError("config must be a ModuleConfig.")
    config = config.normalized()
    config.validate()
    return config


def with_heat_dissipation(config: ModuleConfig, h_dis_w_per_m2_k: float) -> ModuleConfig:
    """Return a config whose average ambient heat-transfer coefficient is ``h_dis``."""
    h_dis = _finite(h_dis_w_per_m2_k, "h_dis_w_per_m2_k", positive=True)
    return replace(config, thermal=replace(config.thermal, r_convection=1.0 / h_dis))


def with_thermal_barrier(
    config: ModuleConfig,
    thickness_m: float,
    conductivity_w_per_m_k: float,
) -> ModuleConfig:
    """Return a config with an added inter-battery layer, ``R_add = d / k``."""
    thickness = _finite(thickness_m, "thickness_m", positive=True)
    conductivity = _finite(conductivity_w_per_m_k, "conductivity_w_per_m_k", positive=True)
    return replace(config, thermal=replace(config.thermal, r_add=thickness / conductivity))


def without_thermal_barrier(config: ModuleConfig) -> ModuleConfig:
    """Return a config with a numerically negligible added inter-battery resistance."""
    return replace(config, thermal=replace(config.thermal, r_add=1.0e-15))


def with_short_energy_scale(config: ModuleConfig, ratio: float) -> ModuleConfig:
    """Scale triggered and spontaneous short-circuit energies together."""
    factor = _finite(ratio, "ratio", positive=True)
    trigger = replace(
        config.trigger,
        trigger_energy_j=config.trigger.trigger_energy_j * factor,
        spontaneous_energy_j=config.trigger.spontaneous_energy_j * factor,
    )
    return replace(config, trigger=trigger)


def with_heater_profile(
    config: ModuleConfig,
    time_s: Any,
    total_power_w: Any,
    *,
    cell: int | None = None,
    front_fraction: float = 0.5,
    efficiency: float = 1.0,
) -> ModuleConfig:
    """Return a heater-driven config using a total electrical-power time series."""
    profile = HeaterPowerProfileConfig(
        time_s=tuple(time_s),
        total_power_w=tuple(total_power_w),
        front_fraction=front_fraction,
        efficiency=efficiency,
    )
    profile.validate()
    trigger = replace(
        config.trigger,
        kind="heater",
        cell=config.trigger.cell if cell is None else cell,
        heating_stop_mode="profile",
        heater_profile=profile,
    )
    output = replace(config, trigger=trigger)
    output.validate()
    return output
