# Blackwell PG-19：full KV 与 Quest 困惑度对照

## 实现与协议

入口：`python -m evaluation.baseline.pg19`。模型为本地
LongChat-7B-v1.5-32K，revision `389eea141540fe7d2c314eacdfe980ae1415c6c0`；
FP16、batch=1、RTX PRO 6000 Blackwell，沿用已验证的原生分页注意力后端。

数据采用原仓库使用的 `emozilla/pg19-test`，固定 revision
`c5e39bf32e33f9111323aa68d7d9000d22722035`。本次取第 0 本书，
保存相同的 8193 个 tokenizer IDs（含 tokenizer 添加的特殊 token），
逐次输入 `x[t]`，计算 `x[t+1]` 的负对数似然，共计分 8192 个目标。
没有不计分的 dense prefill，也没有窗口重置；多书模式只在书间重置缓存。

`PPL = exp(sum(token NLL) / scored_tokens)`。预算 0 表示完整注意力；
4096 表示 256 页，页大小 16，包含强制保留的尾页，前两层始终完整注意力。
Quest 仍保存全部 KV，仅限制每步读取参与注意力的页，不是缓存淘汰实验。

## 8192-target 实测

| 模式 | 计分目标数 | PPL | 相对 full 的 PPL 差值 | 相对变化 |
| --- | ---: | ---: | ---: | ---: |
| Full KV | 8192 | 7.2921040065 | 0 | 0 |
| Quest，4096-token 预算 | 8192 | 7.2969447354 | +0.0048407290 | +0.066383% |

前 4096 个目标对应的上下文尚未超过预算，两组 NLL 逐项完全相同。
后 4096 步在可稀疏层触发选页；仅对这段相同目标计算的 PPL 分别为
7.3256289136（full）和 7.3353581095（Quest）。

该结果仅是一本文本、8192 个目标的验证，不是完整测试集评估，也不是论文
32K 长度结果的复现。不同区间的 PPL 不应直接比较，因为目标文本不同。

## 结果与验证材料

- `results/baseline/pg19-8k-full-vs-4096/manifest.json`：环境、输入参数、源码哈希、模型/数据版本，状态为 complete。
- 同目录 `input-0.json`：精确 token IDs、文本和输入哈希。
- `nll-doc0-budget*.jsonl`：逐 token 损失与累计 PPL。
- `summary.json`、`summary.csv`：聚合指标及差值。
- `curve.json`、`perplexity.svg`：累计 PPL 曲线。
- `verification.json`：从逐 token 日志独立重算，核验 8192 个目标、输入哈希和阈值前一致性。

新增 3 个测试覆盖标签对齐、token 加权聚合、整段 HF causal 参考、
大预算退化为 dense 以及稀疏路径对照 gather+SDPA；原有 6 个测试也通过。

## 运行

本次命令（使用物理 GPU 1，进程内逻辑 GPU 0）：

```bash
CUDA_VISIBLE_DEVICES=1 /home/bzq/.local/envs/quest/bin/python \
  -m evaluation.baseline.pg19 \
  --output results/baseline/pg19-8k-full-vs-4096 \
  --num-eval-tokens 8192 --token-budgets 0 4096 --log-every 512
```

沿用原仓库脚本的 30,000-target 配置：

```bash
CUDA_VISIBLE_DEVICES=1 /home/bzq/.local/envs/quest/bin/python \
  -m evaluation.baseline.pg19 \
  --output results/baseline/pg19-30k-full-vs-4096 \
  --num-eval-tokens 30000 --token-budgets 0 4096
```

输出目录必须是新目录。可增加 `--documents 100` 请求全部测试书籍，
或使用 `--num-eval-tokens 32000` 扩展长度；这些配置本次未运行。
短书按实际可用目标数计分，多书汇总按 token 数加权。

参考：[Quest 论文 §4.2.1](https://arxiv.org/html/2406.10774v1#S4.SS2.SSS1)、
仓库 `evaluation/pg19/ppl_eval.py` 和 `scripts/ppl_eval.sh`。
