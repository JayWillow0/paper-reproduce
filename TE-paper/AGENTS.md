# AGENTS.md

## 项目目标

本目录复现并审计 TE-method-2024 的温度扩展训练方法。当前目标是区分切片随机验证与整电池泛化，不把 TE 切片数量当成独立电池数量。

## 固定来源

- 论文 DOI 为 `10.1016/j.joule.2024.07.002`。
- 上游仓库为 `Lucywang94/TE-method-2024`。
- 固定 commit 为 `21b2209508f23d7739e499fd4a90825ce4b3d33f`。
- 上游根目录没有 LICENSE。公开分发其代码、数据或衍生产物前必须单独确认许可。

## 目录规则

- `experiments/exp_a` 保存官方 50 epoch demo 基线及最新运行结果。
- `experiments/exp_b` 保存同一 80/20 权重比较随机验证与三只 Test 电池的实验。
- `experiments/exp_c` 保存邻片剔除、单电池留一和三种子实验。
- `REPRODUCIBILITY.md` 保存运行环境、命令、结果哈希和结论边界；`input_data.sha256` 登记固定输入哈希。
- `repo/TE-method-2024` 是本地上游副本，不进入 Git。

## 许可与引用

- LiuYang 原创的复现协议、验证工具和说明文档采用根目录 MIT License。
- `exp_a` 是官方 demo 的本地副本，不纳入本项目 MIT 授权范围。
- `exp_b`、`exp_c` 的原创实验增量采用 MIT；沿用官方数据构造、网络结构或其他实现的部分仍受上游权利约束。
- 引用实验数字或图片时必须区分论文原始结果与本地复现结果，并同时登记论文 DOI、官方仓库和固定 commit。
- 详细引用格式见 `CITATION.md`，第三方边界见根目录 `THIRD_PARTY_NOTICE.md`。

## 协议约束

- v2 训练和验证 loss 按样本数加权。
- 实验 B 对每个 seed 只训练一个固定 80/20 模型，同一权重评估随机验证和 Test 电池。
- 三种子实验统一使用 seed 1、7、42，不允许硬编码历史结果。
- NCM622 留一为 n=1 探索，不得写成总体跨电池误差。
- 50 epoch 和 `delta_SP=0.1` 属于部分复现，不等于论文 500 epoch 的完整设置。

## 清理与验证

- 只保留当前协议的日志、指标和图片。
- 缓存可重建，不进入 Git。
- 新结果替换旧结果时同步更新 `REPRODUCIBILITY.md`、`input_data.sha256` 和 README。
- 验证命令见 README。训练运行前确认上游 commit 和数据清单哈希。
