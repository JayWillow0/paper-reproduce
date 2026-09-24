# Feng 2015 Fig.10–19 复现说明

## 复现范围

参考文献是 Feng Xuning et al. (2015), *Thermal runaway propagation model for designing a safer battery pack with 25 Ah LiNixCoyMnzO2 large format lithium ion battery*。
脚本重做 Fig.10–19 中的仿真研究和版式。没有原始实验数据的图只画计算曲线；数值也
不要求与论文逐点相同。

运行入口：

```bash
python examples/fengxuning_paper_reproduce.py --figures 10-19 --preset default
```

默认输出到 `results/fengxuning/`，包含 PNG、SVG、NPZ 和扫描指标 CSV。

## 模型约定

- Bat1 使用 `trigger_energy_j`，Bat2–6 使用 `spontaneous_energy_j`。Fig.12/13 不设
  per-cell 能量数组；Fig.16 用 `with_short_energy_scale` 一起缩放两类能量。
- 模型内部使用 `r_convection=1/hdis` 和 `r_add=dD/kD`。转换函数负责把论文参数换算成
  热阻，配置中不再存第二份等价参数。
- 核心边缘温度已按论文 Eq.(28)/(29) 修正，并同时用于自发短路事件和 Fig.13–18 后处理。
- RHS 不记录热功率曲线。`reconstruct_heat_power(result)` 根据求解器接受的
  `result.t/result.Y` 重算反应热、短路热、外部加热、热传递和净热功率。
- 旧 `run_TR_sensitivity` 保持 JSON-safe 摘要行为；需要绘图时使用 `run_simulation_sweep` 保留每个工况的完整 `SimulationResult`。
- `run_simulation_sweep(..., workers=N)` 可显式并行因素工况，但默认仍为串行；有序映射
  保证工况、图例和输出文件的逻辑顺序不变。Windows 调用方需使用 `__main__` 保护。
- TR 时刻采用短路事件语义：针刺电池为 `t=0`，其他电池为首次 `spontaneous_short_start`；未传播的相邻间隔保留 `None`。
- Fig.10 分两次运行。绝热 scout run 先找峰值；正式 run 在低于峰值 0.5 K 的上升沿
  触发 `adiabatic_boundary_release`。事件发生后恢复环境换热，状态、反应和内部导热
  连续。后处理也在同一时刻切换边界热流。
- 绘图采用低饱和度固定色系、无顶部总标题、可编辑 SVG 文本和紧凑双栏宽度。Fig.15–18 的 panel a 使用四个代表工况的小面板，不再把工况编码为线型叠在同一坐标轴。

## 逐图能力

| 图 | 仿真内容 | 实现 |
|---|---|---|
| 10 | 绝热 ARC：T–t、dT/dt–T、局部放大、分热源 Q–T | scout 峰值识别 + 事件解除绝热 + 求解后热功率重算 |
| 11 | 单电池针刺：T–t、分热源 Q–t | 5 s 短路释放 + 求解后热功率重算 |
| 12 | 6 电池中心温度及相邻界面温度 | 400/370 kJ 两类现有能量标量 |
| 13 | 6 电池 T̃i_f/T̃i_b | 修正后的核心边缘温度 |
| 14 | 正常传播与无自发短路对照 | `spontaneous_short_enabled` |
| 15 | TTR,ARC 因素研究 | panel a 四个代表工况 + 传播时间/核心边缘温度/参数关系 |
| 16 | ΔHe 因素研究 | 两类能量联动缩放，布局同 Fig.15 |
| 17 | hdis 因素研究 | `hdis -> r_convection`，布局同 Fig.15 |
| 18 | kD/dD 隔热因素研究 | `dD/kD -> r_add`；增加 0.08/0.04 W/(m·K) 高热阻点 |
| 19 | 2 mm、0.08 W/(m·K) 隔热层两工况 | 400 kJ/26°C 与 320 kJ/17°C |

## 数据与显示范围

- 热网络、触发、因素扫描和绘图框架为单一 6 反应机理服务；热源图例按激活反应集动态生成。
- 公式、单位、事件条件和研究变量按论文定义处理。动力学参数不同，峰温、延时和传播
  结果也可能不同。
- Fig.10–12 和 Fig.19 只有拿到原始数据或可靠的数字化数据后才叠加实验线。
- 热路径优化只改变计算组织方式，不改变 Feng 方程、事件 ID、状态布局、默认配置或
  图形输出协议；标量/批量反应和现有数值基线均有回归锁定。
- 统一显示窗口为：Fig.12/13 0–500 s，Fig.15 panel a/c 0–1000 s，Fig.16–18 panel a/c 0–500 s，Fig.19 0–2000 s。完整求解时序仍保存在 NPZ，不因显示裁剪而丢失。
