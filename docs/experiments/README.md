# Blackwell 实验归档索引

本地 `results/`、`data/` 被 Git 忽略。下表是保存和复跑的入口，
不是实验数据的远程备份；迁移时需单独复制结果目录与模型。

| 目录（相对仓库根目录） | 协议与用途 |
| --- | --- |
| `results/baseline/passkey-10k-100` | 100 个精确 10K-token 输入，Full / 64 / 128 页检索 |
| `results/baseline/sweep` | 4K / 8K / 16K，Full / 64 / 128 页，未启用 profiling 的主性能对照及独立插桩运行 |
| `results/baseline/profile-isolated-{0,64,128}` | 历史独立进程 profiling；不能用作普通计时 |
| `results/baseline/pg19-8k-full-vs-4096` | 第 0 本书，8192 目标，Full / Quest 4096 |
| `results/baseline/pg19-30k-full-vs-4096` | 第 0 本书，30000 目标，Full / Quest 4096 |
| `results/baseline/stability-profile-smoke` | 128-token 上下文、两步 decode、Full / 4 页；仅验证新的进程隔离流程 |
| `results/baseline/frozen` | 修复前已有源码、构建和结果索引 |
| `results/baseline/frozen-stable` | 本轮修复后的源码、构建、环境、模型与结果哈希归档 |

PG-19 数据版本固定为 `c5e39bf32e33f9111323aa68d7d9000d22722035`。
各实验目录中的 manifest、保存的精确输入 IDs、逐 token NLL/延迟记录
和 summary 是对应结果的核验依据。不要用当前源码哈希代替旧实验的源码哈希。

新的 `--profile` 入口在 `measure/` 保存普通测量，在独立子目录保存每个
profile 配置；`processes.json` 保存完整命令、输出路径和完成状态。
旧实验的目录布局保持原样，复跑命令见 `evaluation/baseline/README.md`。

归档命令：

```bash
/home/bzq/.local/envs/quest/bin/python -m evaluation.baseline.freeze \
  --output results/baseline/frozen-stable
```

`source-build.zip` 保存源码和 Release 内核；`requirements.txt`、`nvcc.txt`、
`gpu.csv`、`submodules.txt` 记录环境；`model-index.json` 对本地权重和
分词器逐文件计算 SHA256；`artifact-index.json` 为实验结果逐文件建立
路径、大小和 SHA256 索引。`checksums.json` 校验这些归档文件。
模型和结果载荷需要与索引一起保留；单独复制索引无法恢复数据。
