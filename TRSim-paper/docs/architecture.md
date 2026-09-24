# 架构和数值语义

## 数据流

```text
ModuleConfig / build_config
  -> strict validation and reaction overrides
  -> initialize_parameters
       -> node topology and state layout
       -> sparse thermal conductance matrix
       -> reaction metadata and initial state
  -> segmented solve_ivp
       -> pure battery_rhs
       -> terminal event surfaces
       -> accepted mode update and restart
  -> SimulationResult
       -> summary / JSON payload / plotting
```

`ModuleConfig` 只保存用户配置，并且不可变、可序列化。节点索引、插值权重、反应参数
数组、heater profile 和环境换热系数由求解器预先整理，放在本次求解使用的
`ModelParameters` 中。运行时数据不会写回配置。

每个节点的状态连续排列：

```text
[temperature_K, concentration_1, ..., concentration_n, short_energy_released_J]
```

电池编号对外保持 1-based，数组和 `node_index` 对内使用 0-based。节点顺序为：

- `holders="both"`：`holder_f, bat1_f, bat1_b, ..., batN_f, batN_b, holder_b`
- `holders="none"`：`bat1_f, bat1_b, ..., batN_f, batN_b`

标准机理的 Arrhenius 指数取节点温度；onset sigmoid 和负极 `A -> A_high` 取热阻插值
得到的核心温度。

初始化阶段会把反应字段、核心温度/壳体表面插值权重、相邻节点索引和 heater profile
数组整理为只读运行时数据。RHS 对全部电池节点批量计算反应速率；公开标量反应函数
保留为方程参考实现。标准 SEI 方程仍采用未门控的 regeneration 和
`abs(net dc/dt)` 产热定义。要改这两处，需先有独立的物理验证。

核心边缘温度按 Feng 2015 Eq.(28)/(29) 计算。front 边缘取前方相邻节点和完整
`R1,y`，back 边缘取同一电芯的 front/back 节点和完整内部路径。自发短路事件和论文
因素研究都用这个温度，不用电芯中心温度代替。

壳体表面温度是独立的观测算子，不是新的 ODE 状态。它使用节点与外向相邻节点
之间的热流，沿 `r_jr_12 + r_ap_out + r_shell` 插值至金属外壳表面。
`r_contact` 和 `r_add` 位于相邻电芯两个壳表面之间。标定和绘图只在求解结束后
计算该观测量，不改变 RHS 或事件状态。

时变加热 profile 是不可变配置的一部分。RHS 只读取当前时刻的线性插值功率，
功率曲线结束由事件定位；后处理使用同一纯函数重建功率，不记录 RHS 试探点评估。

绝热解除前后的环境换热系数分别保存在两个只读数组中，`RuntimeModes` 决定使用哪一组。
事件只切换 mode，不修改热参数数组。求解结束后仍可查看初始边界、解除后的边界和
最终模式。

模式变化不在 RHS 内发生。针刺结束、自发短路开始/结束和 heater 停止均由终止事件定位；事件接受后批量更新运行标志，再从同一时刻重启积分。浓度不做非负投影，耗尽由 smoothstep 门控；短路累计能量只在事件段边界清理舍入级负值。

每个积分段只建立当前 mode 下可能发生的事件。SciPy 的标量事件共享一个段内缓存；
mode 更新后，新积分段会新建缓存。事件面先按各自的物理尺度归一化，再判断是否同时
发生。全局事件 ID 和事件名称没有变化。

`solver.method="auto"` 选择 LSODA。测试中，5 节标准机理用 BDF 运行 300 s 仍未完成，
LSODA 可以完成；显式指定 BDF/Radau 仅用于诊断。

显式使用 BDF 或 Radau 时，如果某个事件段遇到奇异 Jacobian、非有限牛顿试探或求解
失败，该段从已接受的初态改用 LSODA，后续段也继续使用 LSODA。实际路径写入
`SegmentRecord.solver_method` 和 JSON 的 `segments`。

候选解只要有限且 `success=True`，即使出现 RuntimeWarning 也会保留。去重后的 warning
写入 `SegmentRecord.runtime_warnings`。失败状态、非有限接受值和求解异常会触发 fallback。

也可以显式指定 `Radau` 或 `LSODA` 做诊断。极端刚性条件下，不同隐式求解器给出的
步长序列不同，回归测试比较公开的物理指标，而不是逐步对齐时间网格。

热功率曲线不在 RHS 内记录。`reconstruct_heat_power(result)` 只遍历求解器接受的 `result.t/result.Y`，重算各反应热、短路热、外部加热、热传递和净热功率；短路开闭区间从语义化事件记录恢复。

后处理按节点批量处理全部接受时间点。核心温度、壳体表面温度、电芯界面中面温度、
反应热和动力学 TR 时刻共享预计算观测几何；两个既有界面温度 API 保留原签名与
holder/壳体表面语义。

批量参数更新先按给定顺序收集一组路径变更，再执行一次 `to_dict/from_dict/validate`。
重复路径保持“后值覆盖前值”；原单参数更新接口委托给同一实现。
