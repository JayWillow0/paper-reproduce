"""物理常数、材料参数、工况参数和数值参数。内部全部采用SI单位。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


MODEL_PROTOCOL = "full_charge_geometric_v3_lambda2_c2"


@dataclass(frozen=True)
class PhysicalConstants:
    faraday_c_per_mol: float = 96485.33212  # 法拉第常数，C/mol，CODATA
    gas_j_per_mol_k: float = 8.314462618  # 通用气体常数，J/(mol K)，CODATA
    water_kg_per_mol: float = 0.01801528  # 水摩尔质量，kg/mol，通用物性
    hydrogen_kg_per_mol: float = 0.00201588  # 氢气摩尔质量，kg/mol
    oxygen_kg_per_mol: float = 0.0319988  # 氧气摩尔质量，kg/mol
    nitrogen_kg_per_mol: float = 0.0280134  # 氮气摩尔质量，kg/mol
    reference_temperature_k: float = 298.0  # 相对焓参考温度，K，本项目约定
    reference_pressure_pa: float = 101325.0  # 标准参考压力，Pa，论文表2


@dataclass(frozen=True)
class Material:
    name: str  # 材料英文标识
    thickness_m: float  # 单层厚度，m，论文表1
    cells_base: int  # 基准网格数，本项目离散方案
    porosity: float = 0.0  # 孔隙率，1，论文表1
    ionomer_fraction: float = 0.0  # 离聚物体积分数，1，论文表1
    permeability_m2: float = 0.0  # 绝对渗透率，m²，论文表1
    contact_angle_deg: float = 0.0  # 接触角，deg，论文表1
    electronic_s_per_m: float = 0.0  # 有效电子电导率，S/m，论文表1
    thermal_w_per_m_k: float = 1.0  # 有效导热率，W/(m K)，论文表5
    dry_density_kg_per_m3: float = 1000.0  # 干态总体积基准密度，kg/m³，补充来源或假设
    dry_cp_j_per_kg_k: float = 1000.0  # 干态比热，J/(kg K)，论文表5

    @property
    def dry_heat_capacity_j_per_m3_k(self) -> float:
        return self.dry_density_kg_per_m3 * self.dry_cp_j_per_kg_k


def default_materials() -> tuple[Material, ...]:
    """返回从阳极外侧到阴极外侧的11层材料。"""
    bp = dict(thickness_m=2.0e-3, cells_base=4, electronic_s_per_m=2.0e4,
              thermal_w_per_m_k=20.0, dry_cp_j_per_kg_k=1580.0)
    # 流道等效层按肋占比f_r=0.5平行等效（Part2 §1.2/§6.1）：
    # k=0.5·k_BP+0.5·k_gas，阳极气相0.1672（氢气）、阴极0.0246 W/(m·K)（氧气近似空气，标A）；
    # 干态热容按肋/气体加权0.5×1580=790 J/(kg·K)，配肋密度1000 kg/m³。
    channel_anode = dict(thickness_m=1.0e-3, cells_base=2, electronic_s_per_m=1.0e4,
                         thermal_w_per_m_k=10.0836, dry_density_kg_per_m3=1000.0,
                         dry_cp_j_per_kg_k=790.0)
    channel_cathode = dict(thickness_m=1.0e-3, cells_base=2, electronic_s_per_m=1.0e4,
                           thermal_w_per_m_k=10.0123, dry_density_kg_per_m3=1000.0,
                           dry_cp_j_per_kg_k=790.0)
    gdl = dict(thickness_m=270e-6, cells_base=12, porosity=0.6,
               permeability_m2=6.2e-12, contact_angle_deg=120.0,
               electronic_s_per_m=300.0, thermal_w_per_m_k=1.0,
               dry_cp_j_per_kg_k=568.0)
    mpl = dict(thickness_m=30e-6, cells_base=6, porosity=0.4,
               permeability_m2=6.2e-12, contact_angle_deg=150.0,
               electronic_s_per_m=300.0, thermal_w_per_m_k=1.0,
               dry_cp_j_per_kg_k=2000.0)
    cl = dict(thickness_m=10e-6, cells_base=20, porosity=0.3,
              ionomer_fraction=0.4, permeability_m2=6.2e-13,
              contact_angle_deg=100.0, electronic_s_per_m=300.0,
              thermal_w_per_m_k=1.0, dry_cp_j_per_kg_k=3300.0)
    membrane = dict(thickness_m=30e-6, cells_base=20, ionomer_fraction=1.0,
                    electronic_s_per_m=0.0, thermal_w_per_m_k=0.95,
                    dry_density_kg_per_m3=1980.0, dry_cp_j_per_kg_k=833.0)
    return (
        Material("anode_bp", **bp), Material("anode_channel", **channel_anode),
        Material("anode_gdl", **gdl), Material("anode_mpl", **mpl),
        Material("anode_cl", **cl), Material("membrane", **membrane),
        Material("cathode_cl", **cl), Material("cathode_mpl", **mpl),
        Material("cathode_gdl", **gdl), Material("cathode_channel", **channel_cathode),
        Material("cathode_bp", **bp),
    )


@dataclass(frozen=True)
class ModelParameters:
    constants: PhysicalConstants = field(default_factory=PhysicalConstants)
    equivalent_weight_kg_per_mol: float = 1.1  # 膜当量质量，kg/mol，论文表1换算
    membrane_density_kg_per_m3: float = 1980.0  # 干膜密度，kg/m³，论文表1
    liquid_density_kg_per_m3: float = 990.0  # 液水密度，kg/m³，论文表3
    ice_density_kg_per_m3: float = 920.0  # 冰密度，kg/m³，论文表3
    liquid_cp_j_per_kg_k: float = 4182.0  # 液水比热，J/(kg K)，补充表5.4
    vapor_cp_j_per_kg_k: float = 2014.0  # 水蒸气比热，J/(kg K)，补充表5.4
    ice_cp_j_per_kg_k: float = 2050.0  # 冰比热，J/(kg K)，补充表5.4
    hydrogen_cp_j_per_kg_k: float = 14283.0  # 氢气比热，J/(kg K)，补充表5.4
    oxygen_cp_j_per_kg_k: float = 919.31  # 氧气比热，J/(kg K)，补充表5.4
    nitrogen_cp_j_per_kg_k: float = 1040.0  # 氮气比热，J/(kg K)，项目假设
    fusion_j_per_kg_at_273k: float = 333600.0  # 熔化潜热，J/kg，补充表5.4
    entropy_j_per_mol_k: float = -163.110  # 液态产水反应熵变，J/(mol K)，论文式33换算
    standard_voltage_v: float = 1.23  # 298 K标准可逆电势，V，论文式2
    heat_transfer_w_per_m2_k: float = 100.0  # 外表面对流系数，W/(m² K)，论文表5
    kappa_nv_per_s: float = 1.0  # 离聚物水与蒸气交换速率，1/s，论文表2
    kappa_nl_per_s: float = 1.0  # 离聚物析液速率，1/s，图8变量
    kappa_nf_per_s: float = 1.0  # 膜水冻融速率，1/s，项目补充假设
    kappa_vl_per_s: float = 1.0  # 蒸发凝结速率，1/s，论文表2
    kappa_vi_per_s: float = 1.0  # 凝华速率，1/s，论文表2
    kappa_li_per_s: float = 1.0  # 液冰转换速率，1/s，论文表2
    j0_anode_a_per_m3: float = 1.0e9  # 353.15 K阳极参考交换电流，A/m³，补充附录
    j0_cathode_a_per_m3: float = 1.0e4  # 353.15 K阴极参考交换电流，A/m³，补充附录
    activation_anode_k: float = 1400.0  # 阳极交换电流温度系数，K，补充附录
    activation_cathode_k: float = 7900.0  # 阴极交换电流温度系数，K，补充附录
    concentration_ref_mol_per_m3: float = 40.0  # 反应物参考浓度，mol/m³，补充附录解释
    transfer_coefficient: float = 0.5  # 电荷转移系数，1，论文表3
    cathode_bv_exponent_multiplier: float = 1.0  # 阴极BV指数4αF/RT乘数，1.0=论文原式；0.25对应Tajiri 2007图3实验Tafel斜率约100 mV/dec
    anode_vapor_boundary: str = "reservoir"  # 阳极蒸气边界：reservoir=干储库汇；dead_end=H2供应不变、蒸气零通量（Huo 2019表2死端阳极）
    membrane_diffusivity_mode: str = "paper_c2_regularized"  # 膜水扩散模式，逐字式或λ=2局部C2连续化，用户确认方案
    membrane_diffusivity_transition_half_width: float = 0.10  # λ=2连续化半宽，无量纲，敏感性0.05至0.10
    pore_freeze_k: float = 273.15  # 孔隙冻结阈值，K，用户确认基准
    freeze_smoothing_k: float = 0.05  # 冻融门控平滑宽度，K，数值参数
    pore_blockage_smoothing_onset: float = 0.95  # 总饱和度达到该值后平滑关闭气相孔隙，1，用户建议范围0.9至0.99
    relative_permeability_exponent: float = 3.0  # 液水相对渗透率指数，1，项目假设
    active_area_m2: float = 0.0025  # 活性面积，m²，论文表1
    materials: tuple[Material, ...] = field(default_factory=default_materials)


@dataclass(frozen=True)
class CaseConfig:
    name: str = "baseline"  # 工况名称
    lambda_initial: float = 3.4  # 初始离聚物含水量，mol水/mol磺酸基，论文工况
    temperature_initial_k: float = 253.15  # 初始温度，K，论文表2
    temperature_environment_k: float = 253.15  # 环境和干气储库温度，K，论文表2
    current_a_per_m2: float = 1000.0  # 恒流密度，A/m²，即100 mA/cm²
    pressure_pa: float = 101325.0  # 两侧储库压力，Pa，论文表2
    cathode_oxygen_fraction: float = 0.21  # 干空气氧摩尔分数，1
    phase_mode: str = "non_equilibrium"  # 非平衡或有序平衡相分配
    product_mode: str = "dissolved"  # 产水进入dissolved、liquid或vapor
    thermal_mode: str = "coupled"  # coupled或isothermal
    resolution: str = "base"  # coarse、base或fine
    t_end_s: float = 40.0  # 物理积分终点，s
    output_interval_s: float = 0.1  # 保存时间间隔，s
    voltage_stop_v: float = 0.05  # 工程电压停止阈值，V


@dataclass(frozen=True)
class SolverOptions:
    method: str = "BDF"  # 刚性积分方法，BDF或Radau
    rtol: float = 1.0e-6  # 相对误差容限
    atol: float = 1.0e-8  # 默认状态绝对容差，库存用mol/m³、能量用J/m³
    ice_inventory_atol: float = 1.0e-12  # 孔隙冰qi独立绝对容差，mol/m³，零源换支数值验证确定
    liquid_inventory_atol: float = 1.0e-12  # 孔隙液水ql独立绝对容差，mol/m³，近零库存误差控制
    max_step_s: float = 0.05  # 最大时间步，s
    first_step_s: float = 1.0e-6  # 首步，s
    charge_tolerance: float = 1.0e-12  # 电荷残差相对阈值
    pore_gas_margin: float = 1.0e-6  # 几何气孔余量停止阈值，1，数值安全余量并非完全堵塞
    pressure_domain_fraction: float = 0.05  # 气相总压相对储库的适用域偏移，1；冰堵压缩气孔时可按需放宽，仅数值保护
    max_step_retries: int = 12  # 接受步之前代数失败的减步重试上限，数值参数
    phase_preserving_jacobian: bool = True  # 有序平衡库存差分留在当前相分配分支，数值方法选项
    max_accepted_steps: int = 100000  # 接受步计算上限，数值保护，不代表物理停机
    charge_max_iterations: int = 60  # 电势牛顿最大次数


def to_serializable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {k: to_serializable(v) for k, v in asdict(value).items()}
    if isinstance(value, (tuple, list)):
        return [to_serializable(v) for v in value]
    return value


def validate_case(case: CaseConfig) -> None:
    if case.phase_mode not in {"non_equilibrium", "ordered_equilibrium"}:
        raise ValueError(f"未知相变模式 {case.phase_mode}")
    if case.product_mode not in {"dissolved", "liquid", "vapor"}:
        raise ValueError(f"未知产水机制 {case.product_mode}")
    if case.thermal_mode not in {"coupled", "isothermal"}:
        raise ValueError(f"未知热模式 {case.thermal_mode}")
    if case.phase_mode == "ordered_equilibrium" and case.thermal_mode == "coupled":
        raise ValueError(
            "ordered_equilibrium + coupled 暂不支持：有序容量分配的能量反算非单调，"
            "存在多根。请使用有序平衡等温对照，或非平衡热耦合模型。"
        )
    if case.resolution not in {"coarse", "base", "fine"}:
        raise ValueError(f"未知网格等级 {case.resolution}")
    if case.lambda_initial <= 0 or case.current_a_per_m2 <= 0:
        raise ValueError("初始含水量和恒流密度必须为正")


def validate_model_parameters(params: ModelParameters) -> None:
    """检查会改变模型闭合形式的参数，避免配置静默退化。"""
    allowed = {"paper_literal", "paper_c2_regularized"}
    if params.membrane_diffusivity_mode not in allowed:
        raise ValueError(f"未知膜水扩散模式 {params.membrane_diffusivity_mode}")
    width = params.membrane_diffusivity_transition_half_width
    if not 0.0 < width < 1.0:
        raise ValueError("膜水扩散连续化半宽必须位于0和1之间，不能越过相邻支点")
    if not 0.0 < params.cathode_bv_exponent_multiplier <= 1.0:
        raise ValueError("阴极BV指数乘数必须位于(0,1]，仅允许降低Tafel斜率方向的校准")
    if params.anode_vapor_boundary not in {"reservoir", "dead_end"}:
        raise ValueError(f"未知阳极蒸气边界模式 {params.anode_vapor_boundary}")
