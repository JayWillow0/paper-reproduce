# v1 运行清单（Feng 2015 Fig.10–19 复现）

- 运行日期：2026-09-24
- 论文：Feng Xuning et al. (2015), Applied Energy 154, 74–91, DOI 10.1016/j.apenergy.2015.04.096
- 实现来源：论文未发布官方代码；`moduletr` 为本项目出于学习用途、基于论文
  公开公式独立编写的 Python 实现（仅实现论文六反应动力学，模型名 `standard`）
- 复现脚本：`examples/fengxuning_paper_reproduce.py`（确定性，无随机源，无外部输入数据）

## 环境

- macOS 26.6.2（arm64，Apple Silicon，CPU 求解）
- Python 3.11.15（`/opt/homebrew/opt/python@3.11`，venv 位于 `TRSim-paper/.venv`）
- numpy 2.4.6，scipy 1.17.1，matplotlib 3.11.2，pytest 9.1.1，pillow 12.3.0
- 安装：`python3.11 -m venv .venv && .venv/bin/python -m pip install -e . &&
  .venv/bin/python -m pip install pytest pillow`

## 运行命令

```bash
.venv/bin/python examples/fengxuning_paper_reproduce.py --figures 10-19 --preset default
```

- 求解器：`solve_ivp` LSODA（分段积分 + 终止事件），preset `default`
  （rtol 1e-3，max_step 1 s）
- 共 36 次仿真（fig10 为 scout+正式两次 190,000 s 绝热运行；fig15–18 为
  6/6/7/8 工况扫描），串行执行，总耗时约 4 分钟
- 日志：`experiments/exp_a/run.log`

## 运行代码哈希（SHA-256）

见 `experiments/exp_a/code_hashes.sha256`。复现入口与动力学核心：

| 文件 | SHA-256（前 16 位） |
|---|---|
| `examples/fengxuning_paper_reproduce.py` | `00555a5103289e7c` |
| `src/moduletr/reactions.py` | `d7fb09cfce8c5f92` |
| `src/moduletr/config.py` | `d9a9b4aec5571b31` |
| `src/moduletr/model.py` | `15647bc1c3a56001` |
| `src/moduletr/solver.py` | `09fb0418b45f0316` |

## 验证

- `python -m py_compile src/moduletr/*.py examples/*.py tests/*.py`：通过
- `python -m pytest`：64 passed（含六反应方程标量/批量一致性、
  峰温回归带 `1509.88/1509.98/1510.37 K ± 2 K` 等数值锁定用例）
- 已移除动力学字样零命中：`grep -rniE 'coms[o]l|lf[p]' . --exclude-dir=.venv --exclude-dir=__pycache__`

## 结果

输出目录 `results/fengxuning/`，共 34 个文件：fig10–19 的 PNG+SVG、
6 个单工况 `figN_data.npz`、4 组扫描的 `figN_curves.npz` + `figN_metrics.csv`。
完整哈希清单见 `experiments/exp_a/result_hashes.sha256`。

关键数值抽查（`ModuleTR complete` 行）：

| 工况 | 峰温 | 接受点数 |
|---|---|---|
| fig15 TTR=260°C | 1498.421 K | 6892 |
| fig16 ΔHe=100% | 1498.421 K | 6892 |
| fig17 hdis=25 | 1498.421 K | 6892 |
| fig18 no layer | 1498.385 K | 6937 |
| fig19 400 kJ, 26°C | 1435.969 K | 4643 |

## 解释边界

- 复现对象是论文 Fig.10–19 的**仿真内容与版式**；论文原始实验数据未数字化，
  图中不含实验对照线，数值不要求与论文逐点相同。
- fig10 的绝热解除阈值（峰温 −0.5 K）由 scout run 自动确定；两次运行的
  轨迹由同一确定性流程生成，重跑结果应逐位一致（同机、同版本依赖下）。
- NPZ 数据文件被根 `.gitignore` 排除，可用上述命令再生；PNG/SVG/CSV 纳入 Git。
- 本项目出于学习用途，尝试用 Python 复现文献；建模思路、方法与模型参数归原文献作者所有（见 README 与 CITATION.md）。
