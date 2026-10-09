# results 文章图

本目录保存当前文章图，只含 PNG。

现有四张为 2026-10-09 校准试验（用户授权，依据 Jiao 2009 与 Tajiri 2007 的有限校准）生成的结果：

| 文件 | 内容 |
|---|---|
| `part3_fig4_calibrated.png` | 图4 校准工况电压曲线 |
| `part3_fig5_calibrated.png` | 图5 独立验证电压曲线 |
| `part3_fig8_calibrated.png` | 图8 脱附速率机制对照 |
| `part3_fig9_calibrated.png` | 图9 产水机制对照 |

生成命令（在本项目根目录执行）：

```bash
.venv/bin/python experiments/make_calibration_figures.py --assets results
```

运行登记见 `experiments/calibration_runs.json`，校准结论边界见项目 `README.md` 与 `REPRODUCIBILITY.md`。直接预测版本可由 `experiments/make_article_figures.py --assets results` 生成（`part3_fig*_reproduction.png`），校准结论与直接预测结论分开陈述。
