# 引用指南

使用本项目时需要区分原论文、官方仓库和本地复现结果，不能只引用其中一项来替代其余来源。

## 原论文

Yu Wang, Xuning Feng, Dongxu Guo, Hungjen Hsu, Junxian Hou, Fangshu Zhang, Chengshan Xu, Xiang Chen, Li Wang, Qiang Zhang, and Minggao Ouyang. “Temperature excavation to boost machine learning battery thermochemical predictions.” *Joule* 8, 2639–2651, 2024. https://doi.org/10.1016/j.joule.2024.07.002

```bibtex
@article{wang2024temperature,
  title   = {Temperature excavation to boost machine learning battery thermochemical predictions},
  author  = {Wang, Yu and Feng, Xuning and Guo, Dongxu and Hsu, Hungjen and Hou, Junxian and Zhang, Fangshu and Xu, Chengshan and Chen, Xiang and Wang, Li and Zhang, Qiang and Ouyang, Minggao},
  journal = {Joule},
  volume  = {8},
  pages   = {2639--2651},
  year    = {2024},
  doi     = {10.1016/j.joule.2024.07.002}
}
```

## 官方仓库

引用官方实现或数据时，同时给出仓库地址和本项目固定的版本。

- Repository `https://github.com/Lucywang94/TE-method-2024`
- Tag `v1.0.0`
- Commit `21b2209508f23d7739e499fd4a90825ce4b3d33f`

## 本地复现结果

引用 `experiments/exp_b`、`experiments/exp_c` 的数字或诊断图时，应明确写成 LiuYang 的本地复现结果，并记录所用仓库版本或 commit、运行日期、seed、epoch 和对应指标文件。推荐同时引用原论文与官方仓库。

示例写法如下。

> LiuYang 对 TE-method-2024 的本地复现与评估协议审计，v2，2026-09-01，固定上游 commit 21b2209508f23d7739e499fd4a90825ce4b3d33f。

本地复现结果不应被表述为论文作者报告的原始结果。
