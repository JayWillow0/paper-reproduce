# AE PEMFC 冷启动论文复现

本项目以学习与研究为目的，用 Python 独立重建 Huo 等人 2019 年提出的 PEMFC 冷启动一维多层模型，并复现论文图4、图5、图8和图9所对应的工况。代码依据论文公式和守恒定律独立实现，运行索引和诊断保留在本目录。

## 论文

Sen Huo, Kui Jiao, Jae Wan Park. On the water transport behavior and phase transition mechanisms in cold start operation of PEM fuel cell. Applied Energy 233–234 (2019), 776–788.

DOI https://doi.org/10.1016/j.apenergy.2018.10.068

- 论文原文受版权保护，不随本仓库分发、不进入 Git；请通过出版商 DOI 自行获取。`input_data.sha256` 记录其哈希，仅供本地核对。
- 未发现可直接使用的论文官方代码仓库，本项目不含任何上游代码。
- 独立虚拟环境 `.venv/`。

## 目录结构

| 路径 | 内容 |
|---|---|
| `src/pemfc_coldstart/` | 共享物理与数值模块 |
| `experiments/` | 统一重跑、核验与绘图脚本，运行索引与汇总 JSON |
| `experiments/exp_a/`–`exp_d/` | 图4/5/8/9 实验：配置、入口脚本、参考数据与逐次运行数据（`runs/` 体积较大，不进入 Git） |
| `results/` | 文章图 PNG（当前为 2026-10-09 校准版图4/5/8/9） |
| `tests/` | 物性、守恒、空间离散、求解器与完整性测试 |
| `REPRODUCIBILITY.md` | 环境、命令、运行标识、结果哈希与结论边界 |
| `input_data.sha256` | 固定输入哈希清单 |
| `THIRD_PARTY_NOTICE.md` | 第三方材料与许可登记 |

逐次运行的数据目录 `experiments/exp_<letter>/runs/<run_id>/`（状态数组、日志、指标与诊断图）体积较大，已被根目录 `.gitignore` 排除，不进入 Git；各实验的配置、入口脚本、`reference/` 参考数据以及 `experiments/` 顶层的脚本、运行索引（`*_runs.json`）和汇总文件保留在仓库中用于追溯。

## 脚本功能

### 物理模块 `src/pemfc_coldstart/`

| 模块 | 功能 |
|---|---|
| `parameters.py` | 物理常数、材料、工况与数值参数，内部全部 SI 单位 |
| `mesh.py` | 沿 MEA 厚度方向的一维分层有限体积网格 |
| `state.py` | 守恒库存布局、初态与派生状态 |
| `properties.py` | 论文物性关联及明确记录的修正实现 |
| `electrochemistry.py` | 恒流条件下的电子/质子电势代数约束 |
| `transport.py` | 气体、膜水、液水和热的公共面通量 |
| `phase_change.py` | 五态水转换速率，每条边只计算一次并按计量关系分配 |
| `energy.py` | 能量源项与热量分解 |
| `rhs.py` | 守恒右端组装 |
| `solver.py` | 刚性时间积分（BDF/Radau）与保存点诊断 |
| `postprocess.py` | 运行结果绘图（只输出 PNG） |
| `diagnostics.py` | 运行预算、范围与保存完整性检查 |
| `io.py` | 配置解析、运行目录与结构化输出 |
| `cli.py` | 命令行入口：`simulate` 执行单工况模拟，`validate` 核验已保存运行，`plot` 生成诊断图 |

### 实验脚本 `experiments/`

| 脚本 | 功能 |
|---|---|
| `run_full_reproduction_v3.py` | 按式(11) C2 连续化协议统一重跑图4/5/8/9 共11个工况，登记到 `full_reproduction_v3_runs.json` |
| `finalize_full_reproduction.py` | 核验统一重算的必需文件、输出哈希、有限数、时间单调性、物理域、动态电荷约束与源码哈希，建立当前文章索引与汇总 |
| `make_article_figures.py` | 从统一重跑索引读取真实计算结果，生成四张直接预测文章 PNG（`--assets` 指定输出目录） |
| `make_calibration_figures.py` | 从校准运行索引 `calibration_runs.json` 生成四张校准版文章 PNG |
| `summarize_selected.py` | 汇总历史选中运行，计算网格、容差与求解器差异，输出 `summary_metrics.json` |

### 各实验入口 `experiments/exp_<letter>/`

| 脚本 | 功能 |
|---|---|
| `exp_a/run.py` | 图4 基准工况入口（调用 CLI `simulate` 执行该目录 `config.json`） |
| `exp_a/run_repair_validation.py` | 三项修复协议代表工况重跑；按源码与配置哈希复用已登记有效运行，写入 `repair_runs.json` |
| `exp_a/run_repair_checks.py` | 修复协议的测试、编译与短时积分对照，输出独立诊断目录，不改文章选中运行 |
| `exp_b/run.py` | 图5 工况入口 |
| `exp_b/run_lambda2_validation.py` | 式(11)连续化的半宽（0.05/0.10）、求解器（BDF/Radau）与时间容差敏感性验证，写入 `lambda2_regularization_validation.json` |
| `exp_c/run.py` | 图8 四组相变机制入口（κ_nl=1/2/3 与有序平衡） |
| `exp_c/run_equilibrium_validation.py` | 平衡分支 Jacobian 修复后的长时求解器与容差对照，写入 `equilibrium_repair_runs.json` |
| `exp_c/run_equilibrium_grid.py` | 平衡分支空间网格对照，写入 `equilibrium_grid_runs.json` |
| `exp_d/run.py` | 图9 三种产水机制入口（dissolved/liquid/vapor） |

各实验目录下的 `config*.json` 为对应工况与数值变体的完整配置；`reference/` 存放论文印刷图的人工数字化 CSV。

## 实验协议

模型采用守恒型有限体积离散，覆盖阳极 GDL、MPL、CL、膜和阴极对称区域。主模型保留热耦合，并提供等温对照。孔隙液水冻结阈值以 273.15 K 为基准，从反应气体已就绪的恒流加载时刻开始计时。

当前代码采用完整电势约束的动态耦合，每次ODE右端都求电子/质子电势及恒流约束，反应、拖曳和热使用同一局部电流。历史预测器只用于诊断对照。计算仍为直接预测，没有参数拟合。当前文章运行登记在 `experiments/full_reproduction_runs.json`，历史文章运行继续保留在 `experiments/selected_runs.json` 和版本化索引中；不同模型协议的结果不可混用。

生产协议为 `full_charge_geometric_v3_lambda2_c2`。膜水扩散率默认使用 `paper_c2_regularized`，只在 `λ=2±0.10` 内以五次smootherstep连接论文式(11)两侧的延拓，区间外逐点恢复原式。`paper_literal` 保留用于历史复核。连续化模式与半宽均写入解析配置和运行清单。

液水与冰总饱和度达到0.95后，输运和反应可达率使用有界C1 Hermite sigmoid。库存浓度与平衡分配使用几何气孔体积 ε(1−s_l−s_i)。最早适用域事件优先终止，近堵塞余量为1e-6，标记 `pore_near_blocked`；压力偏移5%可先触发终止。历史平滑起点敏感性尚未按新协议重跑，旧堵塞时刻不得作为当前物理结论。

| 实验 | 目标 |
|---|---|
| exp_a | 图4，初始含水量3.4 |
| exp_b | 图5，初始含水量2.9 |
| exp_c | 图8，平衡机制和脱附速率 |
| exp_d | 图9，产水机制和空间分布 |

共享物理模块放在 `src/pemfc_coldstart/`。2026-10-09统一重跑的文章运行由 `experiments/full_reproduction_runs.json` 固定，汇总结果写入 `experiments/full_reproduction_summary.json`。历史索引继续保留，不改写其出处。

## 运行

以下命令在本项目根目录（`AE-pemfc-paper/`）执行。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .

# 单一工况
.venv/bin/python -m pemfc_coldstart.cli simulate --config experiments/exp_a/config.json

# 用当前v3协议统一重跑图4、5、8、9
.venv/bin/python experiments/run_full_reproduction_v3.py

# 图5的半宽、求解器和时间容差验证
.venv/bin/python experiments/exp_b/run_lambda2_validation.py

# 新协议修复验证（保存新运行与独立诊断，不覆盖文章图）
.venv/bin/python experiments/exp_a/run_repair_validation.py
.venv/bin/python experiments/exp_a/run_repair_checks.py

# 测试与历史已选运行汇总
.venv/bin/python -m pytest
.venv/bin/python experiments/summarize_selected.py

# 从统一重跑索引生成四张直接预测文章 PNG（输出到 results/）
.venv/bin/python experiments/make_article_figures.py --assets results

# 从校准运行索引生成四张校准版文章 PNG（输出到 results/）
.venv/bin/python experiments/make_calibration_figures.py --assets results
```

完整环境、运行标识、限制和验证结果见 `REPRODUCIBILITY.md`。Part2 数值方案记录于外部工作笔记 `part2_numerical_plan.md`（不随本仓库分发）。

## 结果

`results/` 保存当前文章图。现有四张为 2026-10-09 校准试验（用户授权）生成的 `part3_fig4/5/8/9_calibrated.png`，运行登记见 `experiments/calibration_runs.json`。直接预测版本可用 `make_article_figures.py --assets results` 重新生成（`part3_fig*_reproduction.png`）。校准结论与直接预测结论分开陈述，见下方校准试验记录与 `REPRODUCIBILITY.md`。

## 技术记录与结论边界

### 2026-10-09 统一重跑与误差控制

统一重跑图4、5、8、9时，图5暴露近零液水库存的误差尺度问题。`liquid_inventory_atol=1e-12 mol/m³`只作用于非平衡模型的孔隙液水库存，与`ice_inventory_atol`分开记录。该设置不裁剪状态、不更改相变源项；四组文章结果在此修复后统一重算，最终图只保存PNG。

逐字式图5等温BDF曾在18.3971334 s到达 `λ=2` 的吸引型切换面后步长下溢；对应Radau诊断运行持续54 min仍不能有效推进。获批的C2连续化实施后，图5等温BDF用30.15 s墙钟时间完成90 s，Radau用43.72 s完成90 s。两者电压轨迹最大差为6.92e-8 V。半宽0.05与0.10的关键非零统计量差小于0.067%，收紧时间容差十倍后的差小于0.0013%。11个文章工况中9个完成请求终点，另外2个以压力适用域事件结束，没有数值失败。根因、方案和结论边界见外部工作笔记 `part3_fig5_lambda2_diagnosis.md` 与 `part3_lambda2_regularization_proposal.md`（不随本仓库分发）。

### 三项修复协议（用户已授权）

实施细则见外部工作笔记 `part3_repair_plan.md`（不随本仓库分发）。欧姆热为体积源，储库焓交换不反加热流道等效层；完整电势解进入动态右端；几何孔隙与反应平滑分开，接受步和事件有效性先于电压输出。代表工况新运行独立保存，以 `experiments/repair_runs.json` 登记，不覆盖既有选中运行和文章图。修复验收以功率、守恒、耦合反馈及最早事件的独立测试为准。

接受步数量上限仅作计算保护。触及上限时记录数值失败并保存最后有效轨迹，不解释为物理停机。默认100000步，三项修复的代表运行限制5000步。

修复说明及结论边界见外部工作笔记 `part3_repair_results.md`（不随本仓库分发）。平衡分支的长时切换收敛尚未解决，输出中的 `valid` 只表示已保存文件与状态有效，不表示70 s计算完成；同时检查 `status` 和 `end_time_s`。

### 平衡分支继续修复（2026-09-30）

`phase_preserving_jacobian` 修复跨相分配阈值的差分割线。等温粗网格BDF、Radau、收紧容差均推进到66.4024 s附近的压力边界，取代前述“19 s停滞尚未定位”的状态。新运行由 `experiments/equilibrium_repair_runs.json` 固定，空间网格探索由 `experiments/equilibrium_grid_runs.json` 固定，验证索引为 `experiments/equilibrium_repair_checks.json`。

31项测试通过。基准网格在22.80135 s触发微小负冰库存事件，加密网格探索中止；空间收敛未通过。有序热平衡能量反算发现多根，因此不能把等温验证推广到该热平衡分支。完整说明见外部工作笔记 `part3_equilibrium_convergence.md`（不随本仓库分发）。旧结果、旧哈希清单和旧文章图保留，不修改历史清单使其假装来自当前代码。

```bash
.venv/bin/python experiments/exp_c/run_equilibrium_validation.py
.venv/bin/python experiments/exp_c/run_equilibrium_grid.py
.venv/bin/python -m pytest -q
```

2026-10-07负库存诊断表明BDF接受步的历史多项式可在零冰源期间产生微小负值。孔隙冰库存qi采用独立绝对容差1e-12 mol/m³，其余默认尺度不变；不裁剪状态、不放宽1e-9 mol/m³有效性门限。保留全部失败记录，用基准/加密网格和Radau验证；接受步预算独立记录，触顶仍判失败。

2026-10-07用户已确认有序平衡仅作等温对照。生产simulate入口拒绝ordered_equilibrium与coupled组合，说明该有序容量分配存在非单调能量反算；非平衡热耦合继续支持。底层状态解码保留用于历史数据和显式诊断，不作为获准运行该物理组合的入口。新热平衡闭合须另行审阅。

2026-10-07继续空间审查。对CL/MPL制造状态发现液水共同压力无根时静默返回零通量。该路径改为明确的LiquidInterfaceClosureError，由现有积分器记录并减步重试，最终无法闭合则保存最后有效状态并报告数值失败。已保存基准/细网格轨迹未触发此分支，不以该缺陷解释已经观测的网格误差。

2026-10-07局部网格诊断：固定材料物性与模型参数，分别只将膜或阴极CL的cells_base翻倍，其余单元数保持基准。由独立配置完整记录materials（含原物性），入口必须能够无损重载嵌套材料配置。两项验证分别运行并保留日志，用于定位空间误差，不以局部加密直接宣称全网格收敛。

### 空间诊断最终状态（2026-10-07）

最新单层网格索引为 `experiments/local_grid_runs.json`。膜20至40格的BDF在67.150250953 s触发压力域事件；阴极CL20至40格的BDF于28.343859150 s耗尽12000步预算，其有效前缀保留；同一CL网格的Radau在67.925937865 s触发压力域事件。CL局部加密相对全域加密的最大冰量轨迹差为0.003280，事件时差0.0378245 s，但两者的CL均为40格，仍不能证明空间收敛。四组时间/全域对照加上三组局部运行中，六组到达压力域事件、一组数值失败；文件有效性与工况完成状态分别记录。

跨材料液水共同压力无根时已由静默零通量改为明确异常；制造状态测试验证错误可见，已有两条完整保存轨迹未触发该路径。通用配置入口支持完整嵌套材料及常数重载，避免局部网格配置无法复现。43项测试通过，记录在诊断归档的 `pytest_local_grid.log`。未调整物理参数，未进行拟合。PNG空间剖面已人工复核，图像哈希记录在外部工作笔记 `part3_current_assets.json`（不随本仓库分发）。

诊断目录保留各阶段清单，最终 `manifest.json` 记录当前归档，`integrity.json` 记录编译、固定输入及七组输出验证。历史运行源码哈希保持原值。

### 校准试验（2026-10-09，用户授权）

依据 Jiao 2009（Electrochim Acta 54:6876，Table 5 与本复现动力学数值完全一致）与 Tajiri 2007（JPS 165:279，图4/5实验数据来源）做有限校准，目标为图4/5/8/9的电压水平与趋势。校准自由度三个：`cathode_bv_exponent_multiplier=0.25`（阴极Tafel斜率25.1→约100 mV/dec，依据Tajiri图3实验斜率从25 °C约66升至−25 °C约103 mV/dec）、`j0_cathode_a_per_m3`由1e4调至1e7 A/m³@353.15 K（在实验斜率下使图4初始电压命中实验0.620±0.015 V）、`anode_vapor_boundary="dead_end"`（Huo表2死端阳极，蒸气零通量）。数值选项：`voltage_stop_v=0.3`与实验cut-off一致，`pressure_domain_fraction=0.2`。所有新参数默认值保持原直接预测行为不变。

图4为校准工况：初始电压0.628 V（实验0.620），0–50 s逐点差≤0.02 V，电压RMSE由0.504降至0.211 V，70 s后出现冰堵型骤降；0.3 V停机外推约77.5 s，比实验约58.5 s晚约19 s，残余差距可能与停机前仍有约三分之一产水离开阴极CL/膜体系有关，也可能包含参数取值或求解器数值差异的贡献。图5独立验证：初始0.612 V（实验0.590），RMSE 0.173 V，停机差约8 s。图8恢复κ_nl=1/2/3停机提前的排序（77.1/73.3/71.9 s）。图9液态产水37.7 s接近论文40 s；dissolved与vapor仍几乎重合（77.1/77.6 s），与论文48 s与62 s的间隔不一致，可能是产水机制相关参数或求解细节差异造成的。骤降段被压力适用域事件截断于0.47 V附近，可能是数值适用域门限与堵塞物理冲突所致，不宜解读为物理停机。

校准运行由 `experiments/calibration_runs.json` 登记，配置为 `experiments/exp_*/config_cal_*.json`，独立保存，不覆盖任何直接预测运行、旧索引与文章图。直接预测结论与校准结论分开陈述。

## 独立审查产物

只读公式与物理诊断可保存在 `experiments/exp_a/runs/<UTC>_adversarial_audit/`，同目录包含诊断脚本、日志、指标和哈希。此类目录标记 `diagnostic_only`，不作为论文复现实验，不写入 `selected_runs.json`，不自动修改物理模块或删除既有运行。

## 许可

原创增量代码遵守上级仓库许可；论文和第三方材料不受该许可覆盖，见 THIRD_PARTY_NOTICE.md。
