# TE-method-2024 本地复现与审计

本项目保存对 Joule 论文 Temperature excavation to boost machine learning battery thermochemical predictions 的本地复现代码和统一协议结果。重点不是复述论文数字，而是检查随机切片验证、整电池留出和邻片剔除分别回答什么问题。

## 当前结论

- seed 42 训练 200 epoch 后，同一权重的随机切片验证 MSE 为 0.0419，三只 Test 电池分别为 0.6876、0.7778 和 1.9694。
- 邻片剔除 ±0.25 ℃ 和 ±0.5 ℃ 后，相对等量随机子集对照只增加 5.7% 和 9.5%。最近邻切片复制不足以单独解释性能。
- 三种子 50 epoch 结果中，NCM811_80 的 MSE 范围为 0.4783–1.1079，NCM811_VC 为 0.3845–0.4592，NCM523 为 1.3350–1.7450。
- NCM622 单电池留一 MSE 为 3.9458，但 n=1 把电池身份、化学域和曲线复杂度混在一起。
- 这些结果支持采用电池级和批次级分组验证，不能判定具体泄漏机制，也不能否定论文全部结果。

完整数字和哈希见 [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)。

## 目录

```text
TE-paper/
├── AGENTS.md
├── CITATION.md
├── README.md
├── requirements.txt
├── experiments/
│   ├── exp_a/            # 官方 demo 基线和最新结果
│   ├── exp_b/            # 同一权重的随机验证与 Test 电池比较
│   └── exp_c/            # 邻片剔除、留一和三种子实验
├── REPRODUCIBILITY.md    # 运行环境、命令、结果哈希和结论边界
├── input_data.sha256     # 固定输入的 SHA-256 清单
├── scripts/              # 上游准备脚本
└── repo/
    └── TE-method-2024/   # 本地上游副本，不进入 Git
```

## 上游准备

上游仓库固定在 commit `21b2209508f23d7739e499fd4a90825ce4b3d33f`。

```bash
git clone https://github.com/Lucywang94/TE-method-2024.git repo/TE-method-2024
git -C repo/TE-method-2024 checkout 21b2209508f23d7739e499fd4a90825ce4b3d33f
python scripts/prepare_upstream.py
```

最后一条命令会核对 commit，并为官方 demo 创建本地数据软链接。上游仓库在本次审查时没有 LICENSE。`repo/`、原始数据和本地模型权重默认不进入 Git。公开推送前需要确认相关再分发权限。

## 环境

本次结果使用 Python 3.9.6、macOS arm64 和 CPU 运行。

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 运行

实验 A 保留官方 50 epoch demo 行为。它没有固定 seed，每次结果会变化。

```bash
cd experiments/exp_a
MPLBACKEND=Agg ../../.venv/bin/python Main.py | tee exp_a.log
cd ../..
```

实验 B 用同一 seed 42 权重比较随机切片验证和三只 Test 电池。

```bash
MPLCONFIGDIR=/tmp/te-paper-mpl EXP_B_EPOCHS=200 .venv/bin/python experiments/exp_b/experiment_b.py | tee experiments/exp_b/experiment_b_v2.log
```

实验 C 和三种子实验耗时更长，按顺序运行。

```bash
MPLCONFIGDIR=/tmp/te-paper-mpl .venv/bin/python experiments/exp_c/experiment_c.py | tee experiments/exp_c/experiment_c_v2.log
MPLCONFIGDIR=/tmp/te-paper-mpl .venv/bin/python experiments/exp_c/exp_b_multiseed.py | tee experiments/exp_c/exp_b_multiseed_v2.log
```

## 验证

代码至少通过语法编译检查，不依赖第三方包。

```bash
python -m py_compile experiments/exp_a/Main.py experiments/exp_a/supply_function.py experiments/exp_b/experiment_b.py experiments/exp_c/te_common.py experiments/exp_c/experiment_c.py experiments/exp_c/exp_b_multiseed.py
```

## 结果边界

当前三种子实验使用 50 epoch 和演示步长 0.1，论文采用 500 epoch 与更细的数据扩展。当前结果属于部分复现。进一步判断需要更多独立电池、批次和化学体系的预注册分组验证。

## 许可与引用

LiuYang 原创的复现协议、验证工具和说明文档采用根目录 [MIT License](../LICENSE)。MIT 不覆盖论文、官方代码、原始数据、论文图片、第三方权重及其衍生材料。

`experiments/exp_a` 是官方 demo 的本地运行副本。`experiments/exp_b` 和 `exp_c` 包含原创实验增量，也沿用了官方数据构造逻辑和网络结构。文件来源、许可边界和公开分发注意事项见根目录 [THIRD_PARTY_NOTICE.md](../THIRD_PARTY_NOTICE.md)。

使用论文方法、官方脚本或数据时需要引用原论文和官方仓库。引用本地数字或诊断图时还要明确标注为 LiuYang 的复现结果。完整格式见 [CITATION.md](CITATION.md)。
