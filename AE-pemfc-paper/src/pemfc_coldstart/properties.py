"""论文物性关联及明确记录的修正实现。"""

from __future__ import annotations

import numpy as np

from .parameters import ModelParameters


def smooth_pore_availability(total_saturation: np.ndarray | float,
                             onset: float = 0.95) -> np.ndarray:
    """高饱和区的有界C1 Hermite sigmoid，可用率在总饱和度1处严格归零。"""
    if not 0.0 < onset < 1.0:
        raise ValueError("孔隙平滑起点必须位于0和1之间")
    saturation = np.asarray(total_saturation, dtype=float)
    width = 1.0 - onset
    x = (saturation - onset) / width
    # h(0)=1、h'(0)=-1、h(1)=0、h'(1)=0，与线性余量C1连接。
    transition = width * (x**3 - x**2 - x + 1.0)
    return np.where(saturation <= onset, 1.0 - saturation,
                    np.where(saturation < 1.0, transition, 0.0))


def saturation_pressure_pa(temperature_k: np.ndarray | float) -> np.ndarray:
    """表3饱和压，二次项按作者博士论文修正为10^-5。"""
    temperature_k = np.asarray(temperature_k, dtype=float)
    tau = temperature_k - 273.15
    exponent = -2.1794 + 0.02953 * tau - 9.1837e-5 * tau**2 + 1.4454e-7 * tau**3
    return 101325.0 * 10.0**exponent


def lambda_saturation(temperature_k: np.ndarray | float) -> np.ndarray:
    temperature_k = np.asarray(temperature_k, dtype=float)
    low = np.full_like(temperature_k, 4.837)
    mid = 1.0 / (-1.304 + 0.01479 * temperature_k - 3.594e-5 * temperature_k**2)
    return np.where(temperature_k < 223.15, low, np.where(temperature_k <= 273.15, mid, np.inf))


def lambda_equilibrium(activity: np.ndarray | float) -> np.ndarray:
    activity = np.asarray(activity, dtype=float)
    polynomial = 0.043 + 17.81 * activity - 39.85 * activity**2 + 36.0 * activity**3
    high = 14.003 + 1.4 * (activity - 1.0)
    return np.where(activity <= 1.0, polynomial, high)


def membrane_diffusivity_m2_per_s(
    lambda_n: np.ndarray,
    temperature_k: np.ndarray,
    mode: str = "paper_literal",
    transition_half_width: float = 0.10,
) -> np.ndarray:
    """论文式(11)及其获批的λ=2局部C2连续化。

    ``paper_literal``逐字保留四支和原文跳变。``paper_c2_regularized``只在
    λ=2±transition_half_width内用五次smootherstep连接两侧延拓，区间外
    严格返回原式。半宽是无量纲含水量，不是拟合参数。
    """
    lam = np.asarray(lambda_n, dtype=float)
    t = np.asarray(temperature_k, dtype=float)
    mid = 1.0e-10 * (0.87 * (3.0 - lam) + 2.95 * (lam - 2.0))
    upper = 1.0e-10 * (2.9514 * (4.0 - lam) + 1.642454 * (lam - 3.0))
    cubic = 1.0e-10 * (2.563 - 0.33 * lam + 0.0264 * lam**2 - 0.000671 * lam**3)
    f_t = np.exp(2416.0 * (1.0 / 303.15 - 1.0 / t))
    with_temperature = np.where(lam <= 3.0, mid, np.where(lam <= 4.0, upper, cubic)) * f_t
    literal = np.where(lam <= 2.0, 2.69266e-10, with_temperature)
    if mode == "paper_literal":
        return literal
    if mode != "paper_c2_regularized":
        raise ValueError(f"未知膜水扩散模式 {mode}")
    if not 0.0 < transition_half_width < 1.0:
        raise ValueError("膜水扩散连续化半宽必须位于0和1之间")
    left = 2.0 - transition_half_width
    right = 2.0 + transition_half_width
    xi = np.clip((lam - left) / (right - left), 0.0, 1.0)
    weight = xi**3 * (10.0 + xi * (-15.0 + 6.0 * xi))
    right_extension = mid * f_t
    regularized = (1.0 - weight) * 2.69266e-10 + weight * right_extension
    return np.where((lam > left) & (lam < right), regularized, literal)


def proton_conductivity_s_per_m(lambda_n: np.ndarray, temperature_k: np.ndarray) -> np.ndarray:
    lam = np.asarray(lambda_n, dtype=float)
    t = np.asarray(temperature_k, dtype=float)
    return (0.5139 * lam - 0.326) * np.exp(1268.0 * (1.0 / 303.15 - 1.0 / t))


def liquid_viscosity_pa_s(temperature_k: np.ndarray | float) -> np.ndarray:
    t = np.asarray(temperature_k, dtype=float)
    return 2.414e-5 * 10.0 ** (247.8 / (t - 140.0))


def surface_tension_n_per_m(temperature_k: np.ndarray | float) -> np.ndarray:
    return 0.1218 - 1.676e-4 * np.asarray(temperature_k, dtype=float)


def gas_diffusivity_m2_per_s(species: str, temperature_k: np.ndarray, pressure_pa: float) -> np.ndarray:
    # 前四项为论文表3。n2论文未给，取同族PEMFC参数集的2.88e-5（补充来源S，与表3阴极侧
    # O2、蒸气值配套），替代早期复用氧气值的假设A。
    refs = {"h2": 1.005e-4, "vapor_anode": 1.055e-4,
            "o2": 2.652e-5, "n2": 2.88e-5, "vapor_cathode": 2.982e-5}
    return refs[species] * (np.asarray(temperature_k) / 333.15) ** 1.5 * (101325.0 / pressure_pa)


def standard_voltage_v(temperature_k: np.ndarray | float, params: ModelParameters) -> np.ndarray:
    c = params.constants
    t = np.asarray(temperature_k, dtype=float)
    delta_h_ref = c.reference_temperature_k * params.entropy_j_per_mol_k - 2.0 * c.faraday_c_per_mol * params.standard_voltage_v
    delta_cp = (c.water_kg_per_mol * params.liquid_cp_j_per_kg_k
                - c.hydrogen_kg_per_mol * params.hydrogen_cp_j_per_kg_k
                - 0.5 * c.oxygen_kg_per_mol * params.oxygen_cp_j_per_kg_k)
    dh = delta_h_ref + delta_cp * (t - c.reference_temperature_k)
    ds = params.entropy_j_per_mol_k + delta_cp * np.log(t / c.reference_temperature_k)
    return -(dh - t * ds) / (2.0 * c.faraday_c_per_mol)


def relative_enthalpies_j_per_mol(temperature_k: np.ndarray, params: ModelParameters) -> dict[str, np.ndarray]:
    c = params.constants
    theta = np.asarray(temperature_k) - c.reference_temperature_k
    fusion_ref = params.fusion_j_per_kg_at_273k + (params.liquid_cp_j_per_kg_k - params.ice_cp_j_per_kg_k) * (c.reference_temperature_k - 273.15)
    vapor_ref = 3170700.0 - 2438.5 * c.reference_temperature_k
    return {
        "n": c.water_kg_per_mol * params.liquid_cp_j_per_kg_k * theta,
        "l": c.water_kg_per_mol * params.liquid_cp_j_per_kg_k * theta,
        "v": c.water_kg_per_mol * (params.vapor_cp_j_per_kg_k * theta + vapor_ref),
        "i": c.water_kg_per_mol * (params.ice_cp_j_per_kg_k * theta - fusion_ref),
        "f": c.water_kg_per_mol * (params.ice_cp_j_per_kg_k * theta - fusion_ref),
        "h2": c.hydrogen_kg_per_mol * params.hydrogen_cp_j_per_kg_k * theta,
        "o2": c.oxygen_kg_per_mol * params.oxygen_cp_j_per_kg_k * theta,
        "n2": c.nitrogen_kg_per_mol * params.nitrogen_cp_j_per_kg_k * theta,
    }
