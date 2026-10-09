"""沿MEA厚度方向的一维分层有限体积网格。"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .parameters import Material, ModelParameters


@dataclass(frozen=True)
class Mesh:
    x_m: np.ndarray
    dx_m: np.ndarray
    face_x_m: np.ndarray
    layer_index: np.ndarray
    layer_name: np.ndarray
    porosity: np.ndarray
    ionomer_fraction: np.ndarray
    permeability_m2: np.ndarray
    contact_angle_deg: np.ndarray
    electronic_s_per_m: np.ndarray
    thermal_w_per_m_k: np.ndarray
    dry_heat_capacity_j_per_m3_k: np.ndarray
    pore_cells: np.ndarray
    ionomer_cells: np.ndarray
    membrane_cells: np.ndarray
    anode_pore_cells: np.ndarray
    cathode_pore_cells: np.ndarray
    anode_cl_cells: np.ndarray
    cathode_cl_cells: np.ndarray
    electronic_cells: np.ndarray

    @property
    def n_cells(self) -> int:
        return self.x_m.size


def _count(material: Material, resolution: str) -> int:
    factor = {"coarse": 0.5, "base": 1.0, "fine": 2.0}[resolution]
    return max(1, int(round(material.cells_base * factor)))


def build_mesh(params: ModelParameters, resolution: str = "base") -> Mesh:
    faces = [0.0]
    xs: list[float] = []
    dxs: list[float] = []
    layer_ids: list[int] = []
    names: list[str] = []
    mats: list[Material] = []
    for lid, material in enumerate(params.materials):
        n = _count(material, resolution)
        dx = material.thickness_m / n
        for _ in range(n):
            xs.append(faces[-1] + 0.5 * dx)
            dxs.append(dx)
            faces.append(faces[-1] + dx)
            layer_ids.append(lid)
            names.append(material.name)
            mats.append(material)
    arr_names = np.asarray(names, dtype="U20")
    indices = np.arange(len(xs), dtype=int)
    pore = indices[np.asarray([m.porosity > 0 for m in mats])]
    ionomer = indices[np.asarray([m.ionomer_fraction > 0 for m in mats])]
    membrane = indices[arr_names == "membrane"]
    anode_pore = indices[np.char.startswith(arr_names, "anode_") & np.isin(indices, pore)]
    cathode_pore = indices[np.char.startswith(arr_names, "cathode_") & np.isin(indices, pore)]
    anode_cl = indices[arr_names == "anode_cl"]
    cathode_cl = indices[arr_names == "cathode_cl"]
    electronic = indices[arr_names != "membrane"]
    return Mesh(
        x_m=np.asarray(xs), dx_m=np.asarray(dxs), face_x_m=np.asarray(faces),
        layer_index=np.asarray(layer_ids), layer_name=arr_names,
        porosity=np.asarray([m.porosity for m in mats]),
        ionomer_fraction=np.asarray([m.ionomer_fraction for m in mats]),
        permeability_m2=np.asarray([m.permeability_m2 for m in mats]),
        contact_angle_deg=np.asarray([m.contact_angle_deg for m in mats]),
        electronic_s_per_m=np.asarray([m.electronic_s_per_m for m in mats]),
        thermal_w_per_m_k=np.asarray([m.thermal_w_per_m_k for m in mats]),
        dry_heat_capacity_j_per_m3_k=np.asarray([m.dry_heat_capacity_j_per_m3_k for m in mats]),
        pore_cells=pore, ionomer_cells=ionomer, membrane_cells=membrane,
        anode_pore_cells=anode_pore, cathode_pore_cells=cathode_pore,
        anode_cl_cells=anode_cl, cathode_cl_cells=cathode_cl,
        electronic_cells=electronic,
    )
