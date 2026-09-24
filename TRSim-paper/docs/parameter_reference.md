# 参数说明

公共配置使用冻结的 `dataclass`，可以从嵌套 JSON 或 dict 构造。遇到未知字段会立即
报错。`ModuleConfig.to_dict()` 会把 ndarray、NumPy scalar、tuple 和嵌套容器转换成
JSON 可接受的类型；输出可直接交给 `json.dumps`，也能由 `ModuleConfig.from_dict()`
重新加载。结果对象的各个 detail level 也遵守这条规则。

`workers` 属于标定、敏感性和 sweep 的运行选项，不写进物理配置。它默认为 1，旧的
配置 JSON 可以照常加载。

| 配置 | 默认值 | 单位/约束 |
|---|---:|---|
| `topology.n_cells` | 5 | 正整数 |
| `topology.holders` | `both` | `both` / `none` |
| `cell.mass_per_node_kg` | 0.36 | kg/热节点 |
| `cell.specific_heat_j_per_kg_k` | 1100 | J/(kg·K) |
| `holder.mass_kg` | 0.474 | kg/holder 节点 |
| `holder.specific_heat_j_per_kg_k` | 460 | J/(kg·K) |
| `trigger.kind` | `needle` | `needle` / `heater` |
| `trigger.cell` | 1 | 1-based 电芯编号 |
| `trigger.trigger_energy_j` | 4.0e5 | J/整电芯 |
| `trigger.spontaneous_energy_j` | 3.7e5 | J/整电芯 |
| `trigger.heater_power_w_per_node` | 1000 | W/节点 |

## 时变外部加热

`TriggerConfig.heater_profile` 用于实测电功率驱动的外部加热工况。其字段为：

| 字段 | 含义 |
|---|---|
| `time_s` | 从 0 s 开始、严格递增的功率采样时刻 |
| `total_power_w` | 单路加热器总电功率 |
| `front_fraction` | 分配到目标电芯 front 节点的比例，默认 0.5 |
| `efficiency` | 电功率进入电芯的有效比例，默认 1.0 |

使用 profile 时必须配置 `trigger.kind="heater"` 和
`trigger.heating_stop_mode="profile"`。功率在采样点之间线性插值，在最后一个
采样时刻通过 `heater_stop` 事件精确停止。原有
`heater_power_w_per_node` 恒功率语义不变。

profile 加热还可以配置两个独立安全联锁：

| 字段 | 单位 | 含义 |
|---|---|---|
| `heater_stop_surface_temperature_k` | K | 加热电芯任一大面壳温达到该值后强制停止加热；`null` 禁用 |
| `heater_stop_surface_rate_k_per_s` | K/s | 加热电芯任一大面壳温升速率达到该值后强制停止加热；`null` 禁用 |

两个阈值是“或”关系，也与功率曲线结束条件并列。最先满足的条件触发 `heater_stop`。
此后即使 `total_power_w` 还有非零值，RHS 也不再施加外部加热。这里检查的是壳体
外表面，不是核心边缘的自发短路阈值 `tr_threshold_k`。

```python
config = with_heater_profile(
    config,
    time_s=[0.0, 10.0, 20.0],
    total_power_w=[0.0, 800.0, 0.0],
    cell=3,
    front_fraction=0.5,
    efficiency=0.85,
)
```

profile 和功率数组都会写入 `ModuleConfig.to_dict()`。单独保存这份配置 JSON 即可复算。

## 触发与热边界

| 配置 | 默认值 | 单位/约束 |
|---|---:|---|
| `trigger.release_duration_s` | 10 | s |
| `trigger.tr_threshold_k` | 533.15 | K |
| `trigger.self_heating_stop_rate_k_per_s` | 1 | K/s |
| `trigger.kinetic_tr_rate_k_per_s` | 1 | K/s；关闭自发内短时，以整只电芯反应热功率除以整只电芯热容所得自生温升速率判定 TR |
| `trigger.heater_stop_surface_temperature_k` | `null` | K；加热电芯任一大面壳温联锁 |
| `trigger.heater_stop_surface_rate_k_per_s` | `null` | K/s；加热电芯任一大面温升速率联锁 |
| `trigger.spontaneous_short_enabled` | `true` | 是否允许非触发电芯发生自发短路 |

`spontaneous_short_enabled=true` 时，`cell_tr_time_s` 取 Feng 模型中的短路事件时刻。
关闭该开关后，后处理按 `kinetic_tr_rate_k_per_s` 判断反应自生温升，不会额外加入
短路热。`propagation_interval_s[i]` 是 Cell i 与 Cell i+1 的非负时间差；任一电芯没有
达到 TR 判据时，该项为 `null`。这种定义也适用于从模组中间向两侧传播的试验。

| 配置 | 默认值 | 单位/约束 |
|---|---:|---|
| `thermal.adiabatic_boundary` | `false` | 为真时取消所有环境换热，保留内部导热 |
| `thermal.adiabatic_release_temperature_k` | `null` | K；仅在绝热边界开启时有效，温升穿越该值后恢复环境换热 |
| `ambient_temperature_k` | 299.15 | K |
| `initial_temperature_k` | `null` | K；电芯节点初始温度，`null` 时回退到环境温度 |
| `t_end_s` | 2000 | s |

`ambient_temperature_k` 是固定的环境边界温度。`initial_temperature_k` 只用于初始化
电芯热节点，端板节点仍从环境温度开始。例如，环境可以设为 25°C，而电芯在试验开始
前已预热到 57°C。`initial_temperature_k=null` 时，所有节点都从环境温度开始。

`ThermalNetworkConfig` 暴露模型的 13 项层/接触热阻和 4 项面积。实现先累加这些层参数再除以面积，当前版本保持这一计算语义，不额外改变单位比例。

论文因素研究使用纯配置转换函数，避免重复真值来源：

```python
config = with_heat_dissipation(config, 70.0)       # r_convection = 1 / hdis
config = with_thermal_barrier(config, 0.001, 0.2)  # r_add = dD / kD
config = without_thermal_barrier(config)            # kD -> infinity
config = with_short_energy_scale(config, 0.75)      # 同时缩放两类短路能量
```

`adiabatic_release_temperature_k` 是预先给定的事件阈值，求解器不会提前知道峰温。
Fig.10 先做一次绝热 scout run，再把峰温以下 0.5 K 设为正式 run 的解除阈值。其他
工况如果已知拆除保温层的控制温度，可以直接填写；留空则全程绝热。

反应热 `dH` 使用 J/g，质量 `m` 使用 g，活化能使用 J/mol，指前因子使用 1/s。标准机理的浓度门控 `c_min=0.01`。

`reaction_overrides` 示例：

```python
config = build_config(
    reaction_overrides={"SEI": {"Ea": 1.4e5, "A": 2e15}}
)
```

`cell.mass_per_node_kg=null` 时沿用 Feng 2015 论文电芯的默认值 `0.36 kg`，比热默认
`1100 J/(kg·K)`。针对其他容量或结构的电芯，应显式填写整芯实测质量的一半以及实测/文献比热。
这里的整芯热容质量与反应参数中的反应物质量 `m` 不同：前者用于
`M·Cp·dT/dt`，后者是参与反应放热的反应物质量。
