# harness-test

Harness Engineering 考核 (2026 夏) 个人解答与探索报告。

## 任务

在 `max_prompt_tokens=2048`、仅允许 stdlib + numpy、不可读写文件的硬约束下，
为 BANKING77 客服意图分类任务设计文本分类 Harness。模型为 Qwen3-8B Instruct
（非思考模式），通过 OpenAI 兼容 API 调用。

## 最终结果（V12）

| 测试集 | 准确率 |
|---|---|
| 干净 DEV | 83.0% (±0.3) |
| 注入合成集 | 62.0% (基线 24%, +38pt) |
| 代号 label 合成集（OOD 替身）| 75.9% |
| Mini MCQA 合成集 | 94.6% |

累计提升：基线 67.5% → 83.0%（+15.5 pt）。

## 仓库结构

```
student_package/    考试官方代码包 + 我们的 solution.py 与本地实验脚本
report/             LaTeX 探索报告（xelatex 编译）
设计方案.md         所有方案的技术文档
实施计划.md         时间预算 + 实验方法学
探索日志.md         逐 phase 的开发记录
```

## 编译报告

```bash
cd report && xelatex exploration_report.tex && xelatex exploration_report.tex
```

## 快速跑测

```bash
cd student_package
python run.py --runs 1 --workers 8           # 干净 DEV
python run.py --dev data/test_dev_inject.jsonl --runs 1 --workers 6  # 注入版
```

## 14 个失败实验

`solution.py` 顶部有所有被禁用的实验性开关与失败原因说明。详见探索报告 §7。
