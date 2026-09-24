# 运行时 API

## 仿真

```python
result = run_simulation(config, verbose=False)
```

`SimulationResult` 包含：

- `t`：时间数组，s
- `Y`：`[n_time, n_dof]` 状态矩阵
- `params`：解析后的模型参数和最终运行模式
- `segments`：各事件积分段
- `events`：语义化事件记录
- `summary`：峰温、TR 时间、传播间隔和短路能量

`result.to_dict("summary")` 返回摘要和事件。`series` 再加入 front/back/center 温度序列，
`full` 还会写入状态矩阵和状态布局。这三种输出都可以直接传给 `json.dumps`。

每个 `SegmentRecord` 还包含 `solver_method` 和只读的 `runtime_warnings: tuple[str, ...]`。
相同的警告消息只保存一次。有限且 `success=True` 的解即使带有 `RuntimeWarning` 也会被接受。

当 `thermal.adiabatic_boundary=True` 且配置了 `adiabatic_release_temperature_k` 时，事件列表会包含一次 `adiabatic_boundary_release`。该事件发生后仅恢复环境换热系数；内部导热矩阵、浓度和短路能量状态不重置。

默认 `method="auto"` 选择 LSODA。批量计算或 Web 请求建议使用 `solver.preset="fast"`。本机参考：5 节标准机理、2000 s、`fast + LSODA` 约 14 s；`default` 用于更严格的单案例分析。实际时间取决于硬件、触发传播段数和参数刚性。

## 标定

```python
calibration = run_TR_calibration(
    base_config,
    target_data,
    [FitParameter("reaction_overrides.SEI.Ea", 1.35e5, 1e5, 1.7e5)],
    {"algorithm": "auto", "max_iter": 20, "rng_seed": 1},
)
```

目标列名可以是 `T_cellN_center_K`、`T_cellN_front_K` 或 `T_cellN_back_K`。
`fminsearch` 对应多起点 Nelder–Mead，`ga` 对应 SciPy differential evolution，`auto`
先运行 differential evolution，再用 Nelder–Mead 精修。传入 `particleswarm` 会抛出
`ValueError`。

`T_cellN_front_K/back_K` 始终表示 Feng Eq.(28)/(29) 的核心边缘温度。外壳热电偶
标定必须使用新增目标：

- `T_cellN_surface_front_K`
- `T_cellN_surface_back_K`

壳体温度根据热网络瞬时热流，沿 `r_jr_12 + r_ap_out + r_shell` 回算到金属壳
外表面；气凝胶和接触热阻位于两个壳表面之间。目标温度允许按传感器包含
`NaN` 缺测值，但每列至少保留两个有效点；时间必须完整且严格递增。

`FitParameter.scale` 用于归一化优化变量，历史记录、最优参数和配置仍使用物理量。

标定 history 用 `status="ok"` 标记成功评价。失败评价写入 `status="failed"`，并附带
`error_type` 和 `error_message`。没有成功评价时，函数抛出 `CalibrationError`。

差分进化可显式启用进程并行：

```python
calibration = run_TR_calibration(..., options={"workers": 4})
```

`workers=1` 时使用 `updating="immediate"`。`workers>1` 时改用
`updating="deferred"`，Nelder-Mead 精修仍为串行。父进程按输入顺序合并 history，
不采用子进程的完成顺序。

## 加热与边界配置示例

实测功率 profile 可叠加大面壳温安全联锁：

```python
config = replace(
    config,
    trigger=replace(
        config.trigger,
        heater_stop_surface_temperature_k=473.15,
        heater_stop_surface_rate_k_per_s=1.0,
    ),
)
```

任一阈值先满足都会触发 `heater_stop`；功率曲线中剩余的非零输入不会继续施加。

环境边界与电芯初温可以独立配置：

```python
config = replace(
    config,
    ambient_temperature_k=298.15,
    initial_temperature_k=330.55,
)
```

电芯节点从 `initial_temperature_k` 开始，holder 节点从环境温度开始；环境边界在整个
仿真期间保持 `ambient_temperature_k` 不变。

## 敏感性

```python
records = run_TR_sensitivity(
    base_config,
    [SensitivityParameter("E_short_total", (3e5, 4e5, 5e5), "J")],
    {"mode": "oat", "workers": 1},
)
```

模式支持 `oat` 和 `grid`，返回 JSON-safe 记录列表。`export_sensitivity_csv` 可写 CSV。

需要绘制每个扫描工况的完整时序时，使用独立接口，不改变旧敏感性 API：

```python
cases = [SimulationCase("hdis=70", config, "hdis", 70.0, "W m-2 K-1")]
runs = run_simulation_sweep(cases, workers=1)
result = runs[0].result
```

`workers>1` 使用有序进程映射，结果顺序与输入一致。Windows 下要把多进程调用放在
`if __name__ == "__main__":` 中。子进程报错后，父进程会补上候选、case 或参数组合
信息并重新抛出异常。

`reconstruct_heat_power(result)` 返回 `HeatPowerSeries`，包含 6 个命名反应热分量，并提供节点、电池和模组聚合。

Feng 2015 复现入口：

```bash
python examples/fengxuning_paper_reproduce.py --figures 10-19 --preset default
```

默认输出位于 `results/fengxuning/`，图片同时保存 PNG/SVG，单工况保存 NPZ，扫描指标保存 CSV。SVG 保留可编辑文本；所有图片使用低饱和度出版配色且不添加顶部总标题。
