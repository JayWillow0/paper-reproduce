# Paper Reproduce

这个仓库用于集中管理论文复现代码、受控实验结果和证据清单。每个一级子目录对应一篇论文，目标是让代码能运行、结果可追溯、结论边界写清楚。

## 当前项目

| 目录 | 论文或方法 | 状态 |
|---|---|---|
| [`TE-paper`](TE-paper/) | Thermal runaway prediction based on temperature expansion | 已完成统一协议的本地审计，仍缺多电池和批次级验证 |

## 统一结构

```text
paper-reproduce/
├── AGENTS.md
├── README.md
├── .gitignore
├── LICENSE
├── THIRD_PARTY_NOTICE.md
└── <paper-project>/
    ├── AGENTS.md
    ├── README.md
    ├── requirements.txt
    ├── experiments/
    ├── REPRODUCIBILITY.md
    ├── input_data.sha256
    ├── scripts/
    └── repo/               # 上游仓库，不进入 Git
```

新项目必须先补齐目录规范、环境版本、上游来源、固定 commit、运行命令和验收方式，再开始训练。

## Git 边界

默认提交实验源码、README、依赖锁定、指标 JSON、运行日志、诊断图和哈希清单。虚拟环境、缓存、上游仓库副本、模型权重和系统文件不进入 Git。

## 许可与第三方材料

LiuYang 原创的复现框架、验证工具、实验增量代码和说明文档采用 [MIT License](LICENSE)。论文、官方代码、原始数据、论文图片、第三方权重及衍生材料不被重新许可，具体边界见 [THIRD_PARTY_NOTICE.md](THIRD_PARTY_NOTICE.md)。

公开推送前必须逐个核对上游代码、数据和权重的再分发权限。学术引用要求独立于开源许可，使用论文方法、脚本或数据时仍需引用原论文和对应官方来源。
