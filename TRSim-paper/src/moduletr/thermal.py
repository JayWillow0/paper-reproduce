from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix

if TYPE_CHECKING:
    from .config import ModuleConfig, ThermalNetworkConfig


@dataclass(frozen=True, slots=True)
class NodeTopology:
    labels: tuple[str, ...]
    node_map: dict[str, int]
    types: tuple[str, ...]
    cell_index: np.ndarray
    is_holder: np.ndarray
    is_battery: np.ndarray
    is_front: np.ndarray
    neighbor_front: np.ndarray
    neighbor_back: np.ndarray
    has_holder_front: bool
    has_holder_back: bool

    @property
    def n_nodes(self) -> int:
        return len(self.labels)


@dataclass(frozen=True, slots=True)
class ThermalObservationGeometry:
    battery_front_nodes: np.ndarray
    battery_back_nodes: np.ndarray
    core_front_neighbors: np.ndarray
    core_front_weights: np.ndarray
    core_back_weights: np.ndarray
    shell_nodes: np.ndarray
    shell_neighbors: np.ndarray
    shell_weights: np.ndarray
    interface_left_nodes: np.ndarray
    interface_right_nodes: np.ndarray
    interface_right_weights: np.ndarray


def _read_only(values: np.ndarray) -> np.ndarray:
    values.setflags(write=False)
    return values


def create_node_mapping(n_cells: int, holders: str) -> NodeTopology:
    has_front = holders == "both"
    has_back = holders == "both"
    labels: list[str] = []
    types: list[str] = []
    cell_index: list[int] = []
    if has_front:
        labels.append("holder_f")
        types.append("holder_f")
        cell_index.append(-1)
    for cell in range(n_cells):
        labels.extend((f"bat{cell + 1}_f", f"bat{cell + 1}_b"))
        types.extend(("bat_front", "bat_back"))
        cell_index.extend((cell, cell))
    if has_back:
        labels.append("holder_b")
        types.append("holder_b")
        cell_index.append(-1)
    node_map = {label: index for index, label in enumerate(labels)}
    type_array = np.asarray(types, dtype=object)
    cell_array = np.asarray(cell_index, dtype=int)
    is_holder = np.isin(type_array, ("holder_f", "holder_b"))
    is_battery = ~is_holder
    is_front = type_array == "bat_front"
    neighbor_front = np.full(len(labels), -1, dtype=int)
    neighbor_back = np.full(len(labels), -1, dtype=int)
    for index, kind in enumerate(types):
        if kind == "holder_f":
            neighbor_back[index] = node_map["bat1_f"]
        elif kind == "holder_b":
            neighbor_front[index] = node_map[f"bat{n_cells}_b"]
        elif kind == "bat_front":
            cell = cell_array[index]
            neighbor_front[index] = node_map["holder_f"] if cell == 0 and has_front else (
                -1 if cell == 0 else node_map[f"bat{cell}_b"]
            )
            neighbor_back[index] = node_map[f"bat{cell + 1}_b"]
        else:
            cell = cell_array[index]
            neighbor_front[index] = node_map[f"bat{cell + 1}_f"]
            neighbor_back[index] = node_map["holder_b"] if cell == n_cells - 1 and has_back else (
                -1 if cell == n_cells - 1 else node_map[f"bat{cell + 2}_f"]
            )
    return NodeTopology(
        labels=tuple(labels),
        node_map=node_map,
        types=tuple(types),
        cell_index=cell_array,
        is_holder=is_holder,
        is_battery=is_battery,
        is_front=is_front,
        neighbor_front=neighbor_front,
        neighbor_back=neighbor_back,
        has_holder_front=has_front,
        has_holder_back=has_back,
    )


def calculate_thermal_resistance(
    node_index: int,
    direction: int,
    topology: NodeTopology,
    thermal: ThermalNetworkConfig,
    n_cells: int,
) -> float:
    raw, area = calculate_area_normalized_thermal_resistance(
        node_index, direction, topology, thermal, n_cells
    )
    return raw / area


def calculate_area_normalized_thermal_resistance(
    node_index: int,
    direction: int,
    topology: NodeTopology,
    thermal: ThermalNetworkConfig,
    n_cells: int,
) -> tuple[float, float]:
    """Return source-model area-normalized layer resistance and contact area."""
    kind = topology.types[node_index]
    if kind == "holder_f":
        raw = (
            thermal.r_holder + thermal.r_ren + thermal.r_contact + thermal.r_shell + thermal.r_ap_out + thermal.r_jr_12
            if direction == 2
            else thermal.r_convection
        )
        return raw, thermal.area_12_m2 if direction == 2 else thermal.area_holder_m2
    if kind == "holder_b":
        raw = (
            thermal.r_holder + thermal.r_ren + thermal.r_contact + thermal.r_shell + thermal.r_ap_out + thermal.r_jr_12
            if direction == 1
            else thermal.r_convection
        )
        return raw, thermal.area_12_m2 if direction == 1 else thermal.area_holder_m2

    cell = int(topology.cell_index[node_index])
    is_front = bool(topology.is_front[node_index])
    if direction == 1:
        if is_front:
            if cell == 0:
                raw = (
                    thermal.r_holder + thermal.r_ren + thermal.r_contact + thermal.r_shell + thermal.r_ap_out + thermal.r_jr_12
                    if topology.has_holder_front
                    else thermal.r_contact + thermal.r_shell + thermal.r_ap_out + thermal.r_convection
                )
            else:
                raw = 2.0 * (thermal.r_contact + thermal.r_shell + thermal.r_ap_out + thermal.r_jr_12) + thermal.r_add
        else:
            raw = 2.0 * (thermal.r_jr_12 + thermal.r_ap_in)
        return raw, thermal.area_12_m2
    if direction == 2:
        if not is_front:
            if cell == n_cells - 1:
                raw = (
                    thermal.r_holder + thermal.r_ren + thermal.r_contact + thermal.r_shell + thermal.r_ap_out + thermal.r_jr_12
                    if topology.has_holder_back
                    else thermal.r_contact + thermal.r_shell + thermal.r_ap_out + thermal.r_convection
                )
            else:
                raw = 2.0 * (thermal.r_contact + thermal.r_shell + thermal.r_ap_out + thermal.r_jr_12) + thermal.r_add
        else:
            raw = 2.0 * (thermal.r_jr_12 + thermal.r_ap_in)
        return raw, thermal.area_12_m2
    if direction in (3, 4):
        return thermal.r_jr_34 + thermal.r_ap_out + thermal.r_cc + thermal.r_shell + thermal.r_convection, thermal.area_34_m2
    if direction == 5:
        return thermal.r_jr_56 + thermal.r_ap_out + thermal.r_air, thermal.area_56_m2
    if direction == 6:
        return thermal.r_jr_56 + thermal.r_ap_out + thermal.r_shell + thermal.r_convection, thermal.area_56_m2
    raise ValueError("direction must be in [1, 6].")


def assemble_thermal_network(config: ModuleConfig, topology: NodeTopology) -> tuple[csr_matrix, np.ndarray]:
    n_nodes = topology.n_nodes
    conductance = np.zeros((n_nodes, n_nodes), dtype=float)
    ambient_coeff = np.zeros(n_nodes, dtype=float)
    for node in range(n_nodes):
        for direction in range(1, 7):
            resistance = calculate_thermal_resistance(node, direction, topology, config.thermal, config.topology.n_cells)
            coefficient = 1.0 / resistance
            neighbor = topology.neighbor_front[node] if direction == 1 else (
                topology.neighbor_back[node] if direction == 2 else -1
            )
            if neighbor >= 0:
                conductance[node, node] -= coefficient
                conductance[node, neighbor] += coefficient
            elif not config.thermal.adiabatic_boundary:
                ambient_coeff[node] += coefficient
    return csr_matrix(conductance), ambient_coeff


def prepare_observation_geometry(
    config: ModuleConfig,
    topology: NodeTopology,
) -> ThermalObservationGeometry:
    """Precompute fixed interpolation geometry used by RHS and observers."""
    n_cells = config.topology.n_cells
    thermal = config.thermal
    front_nodes = np.asarray([topology.node_map[f"bat{cell + 1}_f"] for cell in range(n_cells)], dtype=int)
    back_nodes = np.asarray([topology.node_map[f"bat{cell + 1}_b"] for cell in range(n_cells)], dtype=int)
    front_neighbors = topology.neighbor_front[front_nodes].astype(int, copy=True)
    front_weights = np.empty(n_cells, dtype=float)
    back_weights = np.empty(n_cells, dtype=float)
    for cell, (front, back) in enumerate(zip(front_nodes, back_nodes, strict=True)):
        front_path, _ = calculate_area_normalized_thermal_resistance(
            int(front), 1, topology, thermal, n_cells
        )
        back_path, _ = calculate_area_normalized_thermal_resistance(
            int(back), 1, topology, thermal, n_cells
        )
        front_weights[cell] = thermal.r_jr_12 / front_path
        back_weights[cell] = thermal.r_jr_12 / back_path

    shell_nodes = np.column_stack((front_nodes, back_nodes))
    shell_neighbors = np.column_stack(
        (topology.neighbor_front[front_nodes], topology.neighbor_back[back_nodes])
    ).astype(int, copy=False)
    inner_resistance = (
        thermal.r_jr_12 + thermal.r_ap_out + thermal.r_shell
    ) / thermal.area_12_m2
    shell_weights = np.empty((n_cells, 2), dtype=float)
    for cell in range(n_cells):
        shell_weights[cell, 0] = inner_resistance / calculate_thermal_resistance(
            int(front_nodes[cell]), 1, topology, thermal, n_cells
        )
        shell_weights[cell, 1] = inner_resistance / calculate_thermal_resistance(
            int(back_nodes[cell]), 2, topology, thermal, n_cells
        )

    interface_left = back_nodes[:-1].copy()
    interface_right = front_nodes[1:].copy()
    interface_right_weights = np.empty(max(n_cells - 1, 0), dtype=float)
    for index, (left, right) in enumerate(zip(interface_left, interface_right, strict=True)):
        r_left = calculate_thermal_resistance(int(left), 2, topology, thermal, n_cells)
        r_right = calculate_thermal_resistance(int(right), 1, topology, thermal, n_cells)
        interface_right_weights[index] = r_left / (r_left + r_right)

    return ThermalObservationGeometry(
        battery_front_nodes=_read_only(front_nodes),
        battery_back_nodes=_read_only(back_nodes),
        core_front_neighbors=_read_only(front_neighbors),
        core_front_weights=_read_only(front_weights),
        core_back_weights=_read_only(back_weights),
        shell_nodes=_read_only(shell_nodes),
        shell_neighbors=_read_only(shell_neighbors),
        shell_weights=_read_only(shell_weights),
        interface_left_nodes=_read_only(interface_left),
        interface_right_nodes=_read_only(interface_right),
        interface_right_weights=_read_only(interface_right_weights),
    )


def calculate_core_temperatures(temperatures_k: np.ndarray, parameters: object) -> tuple[np.ndarray, np.ndarray]:
    geometry: ThermalObservationGeometry = parameters.observation_geometry
    ambient = parameters.config.ambient_temperature_k
    front_node = temperatures_k[geometry.battery_front_nodes]
    back_node = temperatures_k[geometry.battery_back_nodes]
    previous = np.full(parameters.n_batteries, ambient, dtype=float)
    available = geometry.core_front_neighbors >= 0
    previous[available] = temperatures_k[geometry.core_front_neighbors[available]]
    front = front_node + geometry.core_front_weights * (previous - front_node)
    back = back_node + geometry.core_back_weights * (front_node - back_node)
    return front, back


def calculate_heat_transfer(temperatures_k: np.ndarray, parameters: object) -> np.ndarray:
    return parameters.conductance @ temperatures_k + parameters.ambient_coeff * (
        parameters.config.ambient_temperature_k - temperatures_k
    )


def calculate_shell_surface_values(
    node_values: np.ndarray,
    parameters: object,
    *,
    boundary_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map node values linearly to front/back metal-shell surface values.

    ``boundary_value`` is ambient temperature for temperatures and zero for
    time derivatives. Contact and barrier resistances remain outside the
    shell observation point.
    """
    values = np.asarray(node_values, dtype=float)
    if values.shape != (parameters.n_nodes,):
        raise ValueError(f"node_values must have shape ({parameters.n_nodes},).")
    geometry: ThermalObservationGeometry = parameters.observation_geometry
    node_values_2d = values[geometry.shell_nodes]
    outward = np.full_like(node_values_2d, boundary_value)
    available = geometry.shell_neighbors >= 0
    outward[available] = values[geometry.shell_neighbors[available]]
    surface = node_values_2d - geometry.shell_weights * (node_values_2d - outward)
    return surface[:, 0], surface[:, 1]
