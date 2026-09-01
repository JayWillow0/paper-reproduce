# v2 运行清单

日期为 2026-09-01。三组运行均使用官方仓库 commit `21b2209508f23d7739e499fd4a90825ce4b3d33f` 的数据，固定输入哈希清单见 `input_data.sha256`，清单内容哈希为 `e596538b502aa13dcfe109caf1c9b2375e0a5d60b87de9d3284f7187d9c81e08`。

## 环境

- macOS 26.6.2 arm64，Apple M5，CPU-only
- Python 3.9.6
- torch 2.8.0
- numpy 2.0.2
- scipy 1.13.1
- matplotlib 3.9.4
- `delta_SP=0.1`，`Dataset_length=100`，5 个 DSC 通道
- batch size 128，Adam，学习率 1e-4

## 运行代码哈希

- 实验 B 脚本 `d12f5ad7edbd1768d76c15a43a2c7c8e24d4bb9c52b9a49ca293baa5ef68c03d`
- 实验 C 脚本 `1a14e7f5c15dbca1c1f8dc59240cf2427a7b2aa07dba1237085a705c4b41f9e5`
- 三种子脚本 `5ebf6c2bce7321f2bf4f15c9dd877688a275e3a43647eaa1afbcbee07c15e431`
- 公共模块 `1bf00e0d01e6a49d7e8330d2750923f8acd5f72efeb08b8d861cd24821ee4455`

## 实验 B

命令为 `env MPLCONFIGDIR=/tmp/wc7-mpl EXP_B_EPOCHS=200 work/.venv/bin/python work/exp_b/experiment_b.py`。

- seed 42，固定 80/20 划分，200 epoch
- 随机验证 MSE 0.041891570829124335
- NCM811_80 为 0.6876197457313538
- NCM811_VC 为 0.7778177857398987
- NCM523 为 1.9694194793701172
- 指标哈希 `ff0c9d1d33100d14de86e626ef32e8cf8e14099c124d4b0ede752054a3bef04e`
- 日志哈希 `b6ff6a2f6c9280f5fdccc8c2dc56bf613af9e4ec313635d7ac1e3f3e343e7038`
- 权重哈希 `34f27f05d230f54aae4d3d747193766ccf5b8b0edfef2c52836299251f4c0120`

## 实验 C

命令为 `env MPLCONFIGDIR=/tmp/wc7-mpl work/.venv/bin/python work/exp_c/experiment_c.py`。

- seed 42，固定 80/20 划分，50 epoch
- 未剔除基线 MSE 0.14874081313610077
- ±0.25 ℃ 为 0.3748767673969269，等量对照 0.3545790910720825，相对增加 5.7%
- ±0.5 ℃ 为 0.6282162070274353，等量对照 0.5739164352416992，相对增加 9.5%
- ±1 ℃ 为 7.973092079162598，等量对照 8.956867218017578，样本量塌缩，不能识别邻近温度效应
- NCM622 留一 MSE 3.945789337158203，内部随机监控 0.1043866906986862，n=1
- 指标哈希 `6dbdb413e3f9bbc58adb7787c8c4ed9e423f35bf4391cb0aaff14b315659e460`
- 日志哈希 `3a8cb9cb0cf2cd47bd10402658c4016ae9fd9a6a6ce3174ccf7bd5faac67f383`

## 三种子统一协议

命令为 `env MPLCONFIGDIR=/tmp/wc7-mpl work/.venv/bin/python work/exp_c/exp_b_multiseed.py`。

| Test 电池 | 三种子均值 | 范围 | 论文 Table S8 |
|---|---:|---:|---:|
| NCM811_80 | 0.7004850506782532 | 0.4782508611679077–1.1078907251358032 | 0.47 |
| NCM811_VC | 0.4195888539155324 | 0.38450154662132263–0.4592350125312805 | 0.46 |
| NCM523 | 1.5118019580841064 | 1.3350201845169067–1.7450027465820312 | 0.18 |

- 指标哈希 `853ac97cb80e2468f5b1391593870797d093d99d8a2f3504dd5202b256903e0c`
- 日志哈希 `3c6241e9be765838da3cc3ace96ce838172048719bc2252cdf6a84088465b391`
- seed 1 权重哈希 `b3f36c898e57aa2e8442f8b291f92a603e154159017e8dab6dcfdc2238296b1f`
- seed 7 权重哈希 `dfb94f9128bcf388c733b22fbec595b557212b2e9e397e921ffe62b717a4d540`
- seed 42 权重哈希 `2fd1543960df41e1c088a5248b4becf406fb022066de57c9bcd9ef32aff6a62a`

## 解释边界

三组运行可以审计统一协议下的局部现象，不能替代更多独立电池、批次和化学体系的分组验证。50 epoch 多种子结果也不等于论文 500 epoch 与细步长设置的完整复现。
