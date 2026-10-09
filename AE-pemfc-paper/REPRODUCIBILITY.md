# 可复现性记录

## 2026-09-30 当前修复协议

当前生产模块为 `full_charge_geometric_v2`。每次动态右端求完整恒流电势；欧姆热以W/m³入账；外部物种焓仅通过物质域边界收支；同步积分水量与能量预算。几何孔隙与平滑反应可达率分开，最早压力、库存、物性或近堵塞事件决定终止。参数仍为直接预测，未调参。

新代表运行见 `experiments/repair_runs.json`，独立测试与5 s BDF/Radau/收紧容差对照见 `experiments/repair_checks.json`。新运行保存模块源码哈希、完整解析参数、状态数组、累计收支、输出哈希及PNG；诊断运行保存编译清单、测试日志、对照数组和脚本哈希。原始输入继续由 `input_data.sha256` 固定。

复现命令为 `.venv/bin/python experiments/exp_a/run_repair_validation.py` 和 `.venv/bin/python experiments/exp_a/run_repair_checks.py`。第一个入口以源码哈希和输出完整性判断是否复用已登记运行，包括如实保留的数值失败；需改变求解协议才能得到不同结论。计算步数上限触发 `failed`，不是论文的停机时间。

图4等温/热耦合代表运行均完成70 s，液态产水工况先触发压力适用域事件。平衡分支仍有相分配切换附近的数值停滞，不能沿用旧版64.61 s堵塞结论。详见外部工作笔记 `part3_repair_results.md`（不随本仓库分发）。

`selected_runs.json`、`summary_metrics.json`及现有文章图仍对应历史版本。以下记录保留用于追溯，旧网格和长时求解器比较不证明新耦合实现已收敛。


## 2026-09-29 历史状态

2026-09-29 完成两轮：第一轮为 Part3 模块化实现、四组论文工况、网格与求解器检查、守恒预算和文章图生成；第二轮按公式审计（外部工作笔记 `part3_code_formula_audit.md`，不随本仓库分发）修复七处与原文或冻结规格的不一致后全部重跑。修复项：式(11)膜水扩散系数四支照抄原文（含 λ=2 跳变，新增支点测试）、GDL 外侧液水单向外排、能量方程随流焓通量 Σ N_k h_k(T_up)、欧姆热面耗散半单元分配、阴极流道等效层热导 10.0123 与两侧热容 0.79e6、D_N2=2.88e-5 m²/s、膜内融冰硬限幅、水预算含液相边界项。结果仍为直接预测，没有逐图调参。图4和图5的电压及停机时间与论文结果存在偏差，可能是某些参数取值或求解器差异造成的；图8脱附速率趋势与平衡分支 64.61 s 堵塞事件得到复现；图9液态产水工况在 39.5538 s 触发孔隙堵塞事件。

## 固定输入

`input_data.sha256` 使用 SHA-256，包含论文原文与三份人工数字化 CSV 的哈希。论文 PDF 受版权保护，不随仓库分发、不进入 Git；如需本地核对，将自行获取的原文以 `huo2019-pemfc.pdf` 命名放在项目根目录后运行验证命令。CSV 来自论文印刷图，位于各实验 `reference/` 目录，含读取不确定度，不是作者原始数据。

可用 `shasum -a 256 -c input_data.sha256` 验证。

## 环境

- macOS，Python 3.9.6
- numpy 2.0.2
- scipy 1.13.1
- matplotlib 3.9.4
- pytest 8.4.2
- pypdf 6.19.0
- pypdfium2 5.13.0

虚拟环境固定在项目根目录 `.venv/`。`requirements.txt` 固定直接依赖，完整传递依赖随每次运行写入 `manifest.json`。

## 运行与检查

```bash
# 自仓库根目录进入本项目
cd AE-pemfc-paper
.venv/bin/python -m compileall -q src experiments tests
.venv/bin/python -m pytest
.venv/bin/python experiments/summarize_selected.py
.venv/bin/python experiments/make_article_figures.py --assets results
```

`experiments/selected_runs.json` 固定文章引用的运行目录。每个运行目录含解析后的参数、压缩状态数组、信号、事件、预算、指标、日志和递归输出哈希。`experiments/summary_metrics.json` 汇总最终状态及网格、容差和 Radau 对照。

## 2026-09-29 历史实现边界

动态积分内直接嵌套完整电势非线性解会使有限差分雅可比代价过高。本版本采用预测器和保存点校正器分裂。总法拉第电流严格等于给定恒流，但预测器与校正器的局部电流分布仍有差异，指标 `charge_predictor_l1_relative_max` 给出最大相对 L1 差。这一差异在当前结果中不可忽略。

水量预算使用保存时刻的边界通量梯形积分（含液相外排），并非求解器每个接受步的通量账本。热耦合代表工况的相对残差约为 10⁻⁴，未达到 Part2 设定的 10⁻⁵ 目标。网格加密后电压和最大含水量变化小于 1%，最大冰体积分数相对变化约 1.16%，因此冰分布尚未满足预定网格标准。

高饱和度孔隙可用率在总液冰饱和度 0.95 后采用有界 C1 Hermite sigmoid，在 1 处严格归零并触发终止事件。液态产水工况在 39.5538 s 记录 `pore_blocked`。0.90 与 0.99 两个敏感性端点分别在 39.7833 s 与 39.4275 s 终止。有序平衡粗网格分支在 64.6072 s 同样以 `pore_blocked` 终止。

该液态产水时刻与论文图9的 40.0 s 接近，但末端电压为 0.842 V，停机判据与论文不同。其余工况未出现论文中的电压骤降，偏差可能是某些参数取值或求解器差异造成的，本复现尚不宜宣称与论文定量一致。

## 2026-09-29 修复版独立复核

只读诊断保存于 `experiments/exp_a/runs/20260929T131930Z_adversarial_audit/`。复核发现热源量纲及外部焓流仍未通过，平衡分支堵塞前已有适用域越界，不能沿用“已全部修复”和“64.61 s堵塞已复现”的强结论。详见外部工作笔记 `part3_adversarial_review.md`（不随本仓库分发）。本次未回滚或修改物理代码、已选数据与图像。

## 平衡分支继续修复（2026-09-30）

`phase_preserving_jacobian` 修复跨相分配阈值的差分割线。等温粗网格BDF、Radau、收紧容差均推进到66.4024 s附近的压力边界，取代前述“19 s停滞尚未定位”的状态。新运行由 `experiments/equilibrium_repair_runs.json` 固定，空间网格探索由 `experiments/equilibrium_grid_runs.json` 固定，验证索引为 `experiments/equilibrium_repair_checks.json`。

31项测试通过。基准网格在22.80135 s触发微小负冰库存事件，加密网格探索中止；空间收敛未通过。有序热平衡能量反算发现多根，因此不能把等温验证推广到该热平衡分支。完整说明见外部工作笔记 `part3_equilibrium_convergence.md`（不随本仓库分发）。旧结果、旧哈希清单和旧文章图保留，不修改历史清单使其假装来自当前代码。

```bash
.venv/bin/python experiments/exp_c/run_equilibrium_validation.py
.venv/bin/python experiments/exp_c/run_equilibrium_grid.py
.venv/bin/python -m pytest -q
```

## 2026-10-07 initial voltage and ice-inventory error control

Diagnostic scripts and hashes are archived in `experiments/exp_a/runs/20261007_initial_voltage_inventory_audit/`. New production cases are indexed by `experiments/ice_tolerance_runs.json`; older indices are unchanged. The new parameter `ice_inventory_atol=1e-12` applies to pore ice qi only, in mol/m³. Tight validation also reduces rtol, default atol and ice atol tenfold. It does not relax state validity or change phase sources.

Base BDF/Radau and tight BDF reach the pressure boundary near67.3019 s; fine BDF reaches67.9638 s. Time integration agrees, but spatial differences in maximum and mean ice saturation remain1.69% and2.01% of the base trajectory peaks, and maximum liquid saturation differs0.00754 (33.38% of its base peak). Spatial convergence is not established. Initial voltage remains0.966999 V without tuning.

After those runs finished, the user-approved guard rejects ordered_equilibrium + coupled in simulate/validate_case. Existing state decoders remain available for reading historical files and explicit counterexamples. The stored runs retain their pre-guard source hashes. Forty tests pass. Only PNG figures are exported.

## 2026-10-07 空间诊断最终状态

最新单层网格索引为 `experiments/local_grid_runs.json`。膜20至40格的BDF在67.150250953 s触发压力域事件；阴极CL20至40格的BDF于28.343859150 s耗尽12000步预算，其有效前缀保留；同一CL网格的Radau在67.925937865 s触发压力域事件。CL局部加密相对全域加密的最大冰量轨迹差为0.003280，事件时差0.0378245 s，但两者的CL均为40格，仍不能证明空间收敛。四组时间/全域对照加上三组局部运行中，六组到达压力域事件、一组数值失败；文件有效性与工况完成状态分别记录。

跨材料液水共同压力无根时已由静默零通量改为明确异常；制造状态测试验证错误可见，已有两条完整保存轨迹未触发该路径。通用配置入口支持完整嵌套材料及常数重载，避免局部网格配置无法复现。43项测试通过，记录在诊断归档的 `pytest_local_grid.log`。未调整物理参数，未进行拟合。PNG空间剖面已人工复核，图像哈希记录在外部工作笔记 `part3_current_assets.json`（不随本仓库分发）。

诊断目录保留各阶段清单，最终 `manifest.json` 记录当前归档，`integrity.json` 记录编译、固定输入及七组输出验证。历史运行源码哈希保持原值。

## 2026-10-09 四组图逐字式历史运行

当前文章运行由 `experiments/full_reproduction_runs.json` 固定，完整数值和验证状态由 `experiments/full_reproduction_summary.json` 汇总。`experiments/finalize_full_reproduction.py` 检查11组运行的必需文件、输出哈希、有限数、时间单调性、物理域、动态电荷约束和运行源码哈希。全部保存结果通过这些文件与状态检查，且与当前 `src/` 的SHA-256一致。

11组中8组达到请求终点。图8有序平衡粗网格在66.4023566 s、图9液态产水在34.1925346 s先触发压力适用域事件。图5等温BDF在18.3971334 s步长下溢，其有效前缀照实保存。图5热耦合完成90 s。当前结果均为直接预测，没有参数拟合。

图5等温失败点的阳极CL局部膜含水量为2.0000000000000004。论文式(11)逐字实现后，253.15 K下 `λ=2` 两侧的膜水扩散率分别为2.69266e-10与约1.80259e-11 m²/s，相差14.94倍。完整右端探针在切换点左、右两侧分别给出正、负 `dλ/dt`，形成吸引型切换面。BDF缩步到浮点间隔以下；Radau连续54 min仍在有限差分Jacobian、完整动态电势解和稀疏LU之间反复工作，随后主动中断。中断目录为 `experiments/exp_b/runs/20261009T001518Z_7e7f839a`，没有结果文件，也没有登记为完成运行。

四张文章图由 `experiments/make_article_figures.py` 从统一索引生成，600 dpi，只输出PNG。图5等温线只绘制到最后有效时刻并明确标注停止。旧索引、旧运行和旧图保留用于追溯。

复现与验证命令为

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python experiments/finalize_full_reproduction.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python experiments/make_article_figures.py --assets results
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPYCACHEPREFIX=/tmp/pemfc-pycache .venv/bin/python -m compileall -q src experiments tests
```

## 2026-10-09 式(11)C2连续化正式运行

用户审阅并确认文章项目 `notes/part3_lambda2_regularization_proposal.md` 后，生产协议升级为 `full_charge_geometric_v3_lambda2_c2`。默认模式 `paper_c2_regularized` 在 `λ=2±0.10` 内用五次smootherstep连接原文两侧延拓，区间外逐点恢复论文式(11)。`paper_literal` 继续保留。模式和半宽进入模型参数、解析配置、运行索引与源码哈希。

当前文章索引为 `experiments/full_reproduction_runs.json`，由 `experiments/full_reproduction_v3_runs.json` 固定本次11个运行。上一版逐字式索引另存为 `experiments/full_reproduction_v2_runs.json`，没有改写旧运行。`experiments/finalize_full_reproduction.py` 已核查必需文件、输出哈希、有限数、时间单调、状态物理域、动态电荷约束、输入哈希、模型协议和当前源码哈希。

11个工况中9个达到请求终点。图8有序平衡等温对照在66.4023566 s、图9液态产水等温工况在34.1848022 s触发压力适用域事件。没有数值失败。图5等温BDF完成90 s，用时30.15 s；逐字式BDF在18.3971334 s失败，逐字式Radau在54 min后主动中断，作为根因对照保留。

验证索引 `experiments/lambda2_regularization_validation.json` 固定三组图5等温对照。默认半宽0.10的Radau完成90 s，用时43.72 s；相对BDF的电压轨迹最大差为6.92e-8 V，平均含水量和最大冰饱和度的峰值归一化差分别为4.13e-6和4.92e-7。收紧时间容差十倍的BDF完成90 s，各检查量的峰值归一化差不超过1.23e-5。半宽0.05的BDF完成90 s；相对半宽0.10，电压最大差为6.39e-5 V，含水量与冰量的峰值归一化差不超过6.69e-4。接近零量的液水峰值相对差为0.9925%，绝对差为1.16e-4。

四张文章图从v3索引重新生成，600 dpi，只保留PNG。图5等温曲线覆盖完整90 s。连续化解决了 `λ=2` 吸引型切换面造成的运行停滞，但计算电压相对论文仍整体偏高，该偏差可能是某些参数取值或求解器差异造成的；当前结果仍为未拟合的直接预测。

复现与验证命令为

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python experiments/run_full_reproduction_v3.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python experiments/exp_b/run_lambda2_validation.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python experiments/finalize_full_reproduction.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python experiments/make_article_figures.py --assets results
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPYCACHEPREFIX=/tmp/pemfc-pycache .venv/bin/python -m compileall -q src experiments tests
```
