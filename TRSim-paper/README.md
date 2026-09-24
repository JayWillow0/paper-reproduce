# Feng 2015 热失控传播模型 Python 复现

本项目用 Python 复现冯旭宁热失控经典文献：

> Feng Xuning et al. (2015), *Thermal runaway propagation model for designing a safer
> battery pack with 25 Ah LiNixCoyMnzO2 large format lithium ion battery*,
> Applied Energy 154, 74–91. DOI: [10.1016/j.apenergy.2015.04.096](https://doi.org/10.1016/j.apenergy.2015.04.096)

**本项目出于学习用途，尝试用 Python 复现上述文献。**

仿真内核 `moduletr` 是基于论文公开公式的独立 Python 实现，覆盖论文对应的
六反应动力学（SEI / 负极 / 隔膜 / 电解液 / 正极两段反应，代码内模型名 `standard`），
求解器为 SciPy `solve_ivp`（LSODA，分段积分 + 终止事件）。复现脚本重做论文
Fig.10–19 的全部仿真研究：绝热 ARC 温升（Fig.10）、单电池针刺（Fig.11）、
6 电池模组传播（Fig.12–14）以及 TTR/ΔHe/hdis/kD 四组因素研究（Fig.15–19）。

## 目录

```text
TRSim-paper/
├── AGENTS.md               # 项目规则与固定来源登记
├── CITATION.md             # 论文引用方式
├── REPRODUCIBILITY.md      # 运行环境、命令、结果哈希与结论边界
├── input_data.sha256       # 输入数据说明（本项目无外部输入数据）
├── requirements.txt        # Python 依赖（numpy/scipy/matplotlib/pytest）
├── pyproject.toml          # src-layout 包定义（moduletr）
├── src/moduletr/           # 仿真内核：config/reactions/model/solver/events/thermal/...
├── examples/               # fengxuning_paper_reproduce.py 及最小示例
├── tests/                  # pytest 回归（含六反应方程数值锁定用例）
├── docs/                   # 架构、参数、运行时 API、逐图复现说明
├── experiments/exp_a/      # 复现运行日志与指标
└── results/fengxuning/     # 复现输出：fig10–19 PNG/SVG + NPZ/CSV 数据
```

## 环境

需要 Python ≥ 3.11。在 `TRSim-paper/` 目录内：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## 运行

```bash
# 全量复现（Fig.10–19，default 精度，串行约 20–40 分钟）
python examples/fengxuning_paper_reproduce.py --figures 10-19 --preset default

# 快速冒烟（单图）
python examples/fengxuning_paper_reproduce.py --figures 11 --preset fast
```

输出写入 `results/fengxuning/`：每图 `figN.png` + `figN.svg`，单工况图附
`figN_data.npz`，因素研究附 `figN_curves.npz` 与 `figN_metrics.csv`。

## 验证

```bash
python -m py_compile src/moduletr/*.py examples/*.py tests/*.py
python -m pytest
```

## 许可与引用

- 本项目代码遵循 MIT License（见仓库根目录 LICENSE）。
- **建模思路、方法与模型参数归原文献作者所有**：
  Feng Xuning et al. (2015), *Thermal runaway propagation model for designing a
  safer battery pack with 25 Ah LiNixCoyMnzO2 large format lithium ion battery*,
  Applied Energy 154, 74–91. 引用方式见 [CITATION.md](CITATION.md)。
- 仿真内核 `moduletr` 是为学习复现文献而独立编写的 Python 实现，不包含任何
  未公开的内部项目材料。
