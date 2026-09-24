# TRSim-paper 项目规则

## 项目目标

出于学习用途，尝试用 Python 复现 Feng Xuning et al. (2015) 的电池模组热失控
传播模型（论文 Fig.10–19 的全部仿真内容）。仿真内核 `moduletr` 是基于论文
公开公式独立编写的 Python 实现，不引用任何未公开的内部资料。

## 固定来源

| 项 | 内容 |
|---|---|
| 论文 | Feng Xuning et al. (2015), Applied Energy 154, 74–91 |
| DOI | 10.1016/j.apenergy.2015.04.096 |
| 官方代码仓库 | 无（论文未发布官方代码） |
| 上游代码 | 无；`moduletr` 为本项目基于论文公开公式独立编写 |
| 上游许可状态 | 不涉及上游代码；本项目实现归 LiuYang，MIT 覆盖 |
| 引用方式 | 见 `CITATION.md` |

## 复现约定

- 动力学只实现论文对应的六反应模型（代码内模型名 `standard`，默认且唯一）。
- 代码、注释、文档、文件名中不得出现其他电池体系动力学或商业仿真工具的字眼。
- 不引入任何未公开数据（内部实测/标定数据）或与论文复现无关的工具代码。

## 目录规则

- `src/moduletr/`：仿真内核（src-layout，editable 安装）。
- `examples/`：入口脚本；`fengxuning_paper_reproduce.py` 为复现主入口。
- `tests/`：pytest 回归；六反应方程与数值基线用例不得放松容差。
- `docs/`：架构/参数/API/逐图复现说明，与代码同步维护。
- `experiments/exp_a/`：复现运行日志与指标，同一运行的产物放在一起。
- `results/fengxuning/`：复现输出图与数据（NPZ 不进 Git，可由脚本再生）。
- `.venv/`、`__pycache__/`、`*.egg-info/` 不进 Git。

## 协议约束

- 复现脚本是确定性的（无随机源）；修改求解协议前先改本文件和 README。
- 数值回归基线（`tests/test_solver.py` 的回归带用例）是数值正确性的门槛。
- 修改动力学方程、事件语义或默认参数前，必须有独立物理验证依据。

## 验证

- `python -m py_compile src/moduletr/*.py examples/*.py tests/*.py`
- `python -m pytest`
- 已移除动力学字样零命中检查：`grep -rniE 'coms[o]l|lf[p]' . --exclude-dir=.venv --exclude-dir=__pycache__`

## 许可与引用

见根仓库 `AGENTS.md` 许可一节与本项目 `CITATION.md`：代码 MIT；论文建模
思路、方法与参数归 Feng Xuning et al. (2015) 所有。
