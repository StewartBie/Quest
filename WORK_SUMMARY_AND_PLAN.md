# Quest Blackwell 复现：工作总结与后续安排

更新日期：2026-10-26

研究负责人：Ziqing Bie（ZiqingBie）

范围：本项目迄今的工作总结，不限定为最近一周的 Git 提交。

## 1. 工作定位

在 NVIDIA Blackwell GPU 上适配 Quest 的查询感知稀疏注意力方法，建立可追溯的正确性、精度和性能基线，分析选页、访存与调度开销，并探索向其他模型和硬件迁移。

当前已完成核心算法的跨平台适配和部分实验验证，尚未完整复现论文的全部任务、全部模型及性能结论。

Quest 在本项目中的含义是：保存完整 KV-cache，通过每页 Key 的 min/max 元数据和当前 Query 估计重要性，只读取选中页计算注意力。预算限制的是参与注意力的 KV 读取量，不等于实际缓存容量，也不是永久淘汰历史 KV。

## 2. 已完成工作

### 2.1 Blackwell 环境与构建适配

- 目标 GPU：NVIDIA RTX PRO 6000 Blackwell Workstation Edition，计算能力 `sm_120`。
- 验证环境：Python 3.10、PyTorch 2.7.1+cu128、CUDA compiler 12.8.93、GCC 13、Transformers 4.45.2。
- 调整编译器及依赖配置，修复缺失头文件、CPU reference 模板编译兼容等问题。
- 建立最小 CUDA 构建入口，通过 C ABI 调用原始内核，绕过旧 RAFT/PyTorch 扩展栈的依赖问题。
- 针对旧 Flash-Attention 二进制 ABI 不兼容，适配路径使用 SDPA prefill。

已提交记录：`c585534`，作者 Ziqing Bie，时间 2026-10-12，说明 `fit for Blackwell`；修改 7 个文件，新增 40 行、删除 14 行。该提交不包含全部后续 baseline 成果。截至本次核对，最近一周没有新提交，baseline、实验报告等仍主要位于未提交工作区，不能仅凭 Git 判断各项成果的完成日期。

### 2.2 推理与评测基础设施

建立了如下 Blackwell decode 路径：

```text
Q/K/V 投影与 RoPE
        ↓
分页 KV 追加与 min/max 元数据更新
        ↓
当前 Query 的页重要性评分
        ↓
Top-K 历史页选择 + 强制保留尾页
        ↓
分页注意力 → 输出投影
```

- 复用原始 CUDA 页评分和分区分页注意力内核。
- 使用 `torch.topk` 替换 RAFT 选择；使用 HND 缓存布局。
- PyTorch 更新 KV 和元数据；缓存 decode 执行计划。
- 前两层保留完整注意力，页预算包含尾页。
- 增加独立的 gather+SDPA 路径用于诊断和数值对照。
- 建立 manifest、源码/输入哈希、精确 token 保存、轨迹审计、快照重放和报告工具。

当前实现边界为 batch=1、FP16、Llama MHA、head_dim=64/128；尚不支持 GQA、padding、自定义 mask、chunked prefill 和训练。

FlashInfer 提供分页读取、注意力计算、调度及底层 CUDA 工具的实现基础；Quest 在其派生内核上加入动态选页。当前不是简单调用最新版 FlashInfer Python API。

### 2.3 正确性验证与代码审查

已经验证：

- 页评分与独立 min/max 参考公式一致。
- KV 追加、页边界、元数据块边界和尾页有效长度处理。
- 完整预算输出与 Hugging Face 对照。
- 稀疏注意力与选中 KV 上的 SDPA 对照。
- 同 stream 预建计划后的 CUDA Graph 重放一致性。
- 额外 958 组 decode 数值对照通过，覆盖 64/128 维、连续追加及不同预算。
- 新增 PG-19 的 3 个计分测试通过，原有 6 个 baseline/分析测试也通过。

此前已修复：直接非分区调用不满足特殊尾页寻址的问题，以及 CUDA Graph 捕获临时 host 数据上传造成重放非法访存的问题。

原审查发现的问题及阶段一处理状态见第 4、8 节。测试通过仅覆盖已验证的输入和执行范围，不代表通用推理后端已完成验证。

### 2.4 Passkey 精度验证

模型为 LongChat-7B-v1.5-32K。使用相同的 100 条、每条精确 10,000 tokens 的合成检索输入，覆盖五个插入深度；最后 64 个输入 token 逐个处理，随后最多生成 10 tokens。

| 配置 | 对应 token 预算 | 正确数 |
| --- | ---: | ---: |
| Full | 完整 | 100/100 |
| Quest 64 页 | 1024 | 100/100 |
| Quest 128 页 | 2048 | 100/100 |

这是记录了协议差异的合成检索验证，不等同于论文全部精度评测，也不代替 LongBench 或自然文本分析。

### 2.5 端到端性能与访存分析

完成 18 个配置、90 条运行记录；以相同 teacher-forced 输入比较上下文长度、预算和后端。未插桩运行的主结果如下：

| 上下文 tokens | Full，ms/步 | Quest 64 页，ms/步 | 相对 Full 加速 |
| --- | ---: | ---: | ---: |
| 4096 | 14.480 | 15.456 | 0.937× |
| 8192 | 14.566 | 15.362 | 0.948× |
| 16384 | 16.256 | 15.442 | 1.053× |

当前条件下，4K/8K 尚未取得端到端收益，16K 获得小幅收益。不能用选中 token 比例直接推算加速倍数。

其他分析成果：

- 采集 294,912 条逐层、逐步、逐头选页记录。
- 相邻步历史页保留比例约 72%–87%，说明存在复用机会，但不证明实际 L2 命中。
- 采用六个真实 Q/KV 快照进行物理页布局和访问顺序配对实验。
- 区分融合注意力与独立 gather+SDPA：独立 gather 的物化成本不是融合内核的纯读取时间。
- 区分 host 发射间隙、GPU kernel 时间和同步开销；观察到 profiling 会影响同进程后续普通计时。
- 硬件计数器尚未采集成功，驱动返回 `ERR_NVGPUCTRPERM`；目前无法据此确定 DRAM 饱和、L2 miss 或 warp 内存等待瓶颈。

### 2.6 PG-19 Full/Quest 困惑度对照

新增入口：`python -m evaluation.baseline.pg19`。

评测协议：固定 `emozilla/pg19-test` revision `c5e39bf32e33f9111323aa68d7d9000d22722035`；本次取第 0 本书，逐次输入真实 token `x[t]`，预测并计分 `x[t+1]`。各预算复用相同输入 IDs；没有不计分的长前缀 prefill，没有滑动窗口重置，多书模式只在书间重置缓存。

```text
PPL = exp(全部目标 token 的 NLL 总和 / 目标 token 总数)
```

多书聚合按 token 数加权，不平均各书 PPL。4096-token 预算对应 256 页；前两层完整注意力，尾页计入预算。

| 评测范围 | Full PPL | Quest 4096-token PPL | PPL 相对变化 |
| --- | ---: | ---: | ---: |
| 首本书，8192 个目标 | 7.292104 | 7.296945 | +0.0664% |
| 首本书，30000 个目标 | 6.837542 | 6.832020 | −0.0808% |

30K 运行由 Ziqing Bie 执行，随后对产物进行分析与核验：

- manifest 状态为 complete；两组输入哈希和 30,000 个计分位置一致。
- 从逐 token 日志独立重算 PPL，与汇总一致。
- 前 4096 个目标的 NLL 完全相同，此时尚未超过预算。
- 30K 运行的前 8192 个 NLL 与此前 8K 运行逐项一致。
- 后续 25,904 步在可稀疏层触发选页；该区间 Full/Quest PPL 分别为 6.773218/6.766883。

30K 分段结果：

| 目标位置 | Full PPL | Quest PPL | Quest 相对变化 |
| --- | ---: | ---: | ---: |
| 1–4096 | 7.258733 | 7.258733 | 0 |
| 4097–8192 | 7.325629 | 7.335358 | +0.133% |
| 8193–16384 | 6.780820 | 6.753072 | −0.409% |
| 16385–24576 | 6.799719 | 6.815428 | +0.231% |
| 24577–30000 | 6.335556 | 6.318075 | −0.276% |

同一区间内可以比较方法差异；不同区间的目标文本不同，不能据其 PPL 高低直接推断上下文长度的效果。部分 token 损失改善、部分退化，整体平均接近；Quest 的微小优势不代表其普遍优于 Full。

30K 的总评测耗时分别约 492.36/458.10 秒，相当于本次运行约 1.075× 的时间比。但其中包含计分、日志、同步与计划创建，且没有重复性能测量，不能代替正式速度基准。

**当前结论：在这一本书上，Quest 在 4096-token 预算下保持了与 Full 非常接近的语言建模困惑度。尚不能外推为完整 PG-19 测试集结论。**

### 2.7 H2O、Qwen 和 FPGA 的前期分析

- **H2O**：仓库没有可直接运行的真实 H2O PG-19 对照入口。README 中的 H2O/TOVA 性能估算及 `local_heavy_hitter_mask` 命名函数不构成完整 H2O 实现。尚未接入或运行 H2O。
- **Qwen2.5-7B**：算法可迁移，但其 28 个 Query 头共享 4 个 KV 头，属于 GQA，需重构当前 MHA 接口、缓存和内核。尚未实现或评测。
- **Qwen3.5**：当前核对的是官方 Qwen3.5-9B，尚未确认“Qwen3.5-7B”对应权重。9B 为混合架构，32 层中有 24 层线性注意力和 8 层完整注意力；Quest 应首先限于后者。尚未实现或评测。
- **FPGA**：已做架构可行性分析，候选范围是页评分、选页和注意力子系统。KV 容量、驻留位置、传输与同步是关键限制；尚无 HLS/RTL 实现、综合结果或硬件加速数据。

## 3. 关键代码与实验材料

| 位置 | 用途 |
| --- | --- |
| [evaluation/baseline/attention.py](evaluation/baseline/attention.py) | 模型接入、分页缓存、选页与注意力流程 |
| [evaluation/baseline/native.py](evaluation/baseline/native.py) | Python/CUDA 接口与执行计划 |
| [kernels/minimal/probe.cu](kernels/minimal/probe.cu) | 原始 CUDA 内核的 C ABI 桥接 |
| [evaluation/baseline/run.py](evaluation/baseline/run.py) | Passkey 和性能评测入口 |
| [evaluation/baseline/pg19.py](evaluation/baseline/pg19.py) | PG-19 困惑度对照入口 |
| [evaluation/baseline/test_pg19.py](evaluation/baseline/test_pg19.py) | 计分对齐、聚合及稀疏对照测试 |
| [evaluation/baseline/README.md](evaluation/baseline/README.md) | 构建与实验使用说明 |
| [docs/reports/quest-blackwell-baseline.md](docs/reports/quest-blackwell-baseline.md) | 基线性能、精度与访存分析 |
| [docs/reports/quest-blackwell-pg19.md](docs/reports/quest-blackwell-pg19.md) | 初始 8K PG-19 验证报告；30K 结果由本文补充 |
| [results/baseline/pg19-30k-full-vs-4096/summary.json](results/baseline/pg19-30k-full-vs-4096/summary.json) | 30K 聚合结果 |
| [results/baseline/pg19-30k-full-vs-4096/perplexity.svg](results/baseline/pg19-30k-full-vs-4096/perplexity.svg) | 30K 累计困惑度曲线 |

`data/` 和 `results/` 中的本地材料受忽略规则影响，不保证随 Git 克隆获得。报告引用的本地结果需单独归档。

## 4. 已知问题与待修正项

| 优先级 | 问题 | 影响与处理方向 |
| --- | --- | --- |
| 已修复 | 原生调用设备上下文保护 | score、计划创建、decode、release 绑定输入或计划设备，退出时恢复调用者设备；双 GPU 回归通过 |
| 已修复 | 原生计划生命周期管理 | 增加自动 finalizer 和上下文管理器；close 幂等且允许复用；自动销毁释放已验证 |
| 已修复 | profiling 与普通计时可能混跑 | --profile 由父进程启动普通测量 worker 和逐配置独立 profiling worker，三组真实模型运行及审计通过；旧主 sweep 不变 |
| 已修复 | CPU reference 的历史断言变量错误 | 修正 MHA、页数整除和尾页长度断言；无 NDEBUG 构建和 HND/NHD 参考测试通过 |
| 待补 | 稀疏精度任务覆盖有限 | 扩展多本 PG-19、不同预算及自然长文本任务 |
| 外部条件 | 性能计数器权限 | 取得允许采集的环境后补充硬件证据，不以逻辑页复用率替代 L2 命中率 |

## 5. 后续安排

以下是建议推进顺序，不代表已完成或已承诺具体日期。

### 阶段一：稳定当前基线并归档

1. 修复设备上下文、原生计划释放和 Debug 断言问题。
2. 将性能采集与 profiling 分进程隔离。
3. 整理并提交 baseline 实现、测试和文档；为实验结果建立独立归档索引。
4. 固定模型、数据版本、输入、依赖和内核二进制哈希。

验收：现有回归测试通过；跨设备调用、资源释放和启用断言构建获得针对性验证；旧结果可以按清晰的协议重跑。

### 阶段二：扩展 PG-19 精度实验

1. 在相同模型上对比 Full 与 1024、2048、4096-token Quest 预算。
2. 先扩大到预先确定的多本书子集，再根据资源扩展至测试集 100 本书。
3. 统一每书最大计分长度（30K 或明确记录的 32K），记录短书实际 token 数。
4. 同时报告 token 加权整体 PPL、逐书差异和进入稀疏状态后的配对区间结果。
5. 分开进行重复性能测试，不将 PPL 脚本的总耗时直接解释为正式加速比。

验收：所有预算输入和目标完全配对；输出整体与逐书结果，说明不同文本间的变化，避免只依据一本书下结论。

### 阶段三：接入 H2O 对照

1. 核对官方 H2O 实现的历史注意力累计、重要/近期 token 预算划分及淘汰时机。
2. 明确是否采用前两层保持 Full 的 H2O* 设置，与 Quest 论文的对照口径保持可追溯。
3. 接入同一 PG-19 计分流程，实现 Full / Quest / H2O 三路比较。
4. 分别报告算法精度和执行性能，不能把未优化参考实现的耗时作为方法本身的性能结论。

验收：大预算退化为 Full；小样本淘汰状态与独立参考一致；同文本、同目标、同名义预算下输出 PPL。明确 Quest 的读取预算和 H2O 的保留预算含义不同。

### 阶段四：迁移到 Qwen2.5-7B

1. 选定具体权重与精度，建立原生 Qwen Full 的 PG-19 参考。
2. 实现 PyTorch 版 GQA Quest：KV 按 4 个 KV 头保存，28 个 Query 头分别评分和选页。
3. 保留原模型投影、RoPE 和位置语义，验证完整预算下逐 token logits/NLL 等价性。
4. 完成多预算 PPL 后，再实现原生 GQA 内核和性能优化。

同一 KV 组内的 Query 头可以有不同选页结果；组内共享选页应作为独立算法变体评估。把 KV 复制为 28 个头可用于临时数值验证，但不能作为最终性能方案，否则会失去 GQA 的存储优势。

验收：缓存不因展开头而膨胀；完整预算与原生 Qwen 对齐；稀疏预算获得可复核精度结果。不同 tokenizer 的模型不直接比较绝对 PPL，以各自 Full 为参照。

### 阶段五：按研究收益选择扩展方向

**Qwen3.5 混合架构：**先确认模型 ID，再只替换完整注意力层，保留 Gated DeltaNet 状态、门控和位置编码；适配混合缓存、head_dim=256 及软件版本。重新研究需要保留 dense 的层，不能照搬“前两层”规则。该方向尚无精度或速度承诺。

**FPGA 原型：**先用真实 Q/KV 快照验证单层数据通路，完成 KV 容量、有效带宽、PCIe 通信和每层同步预算，再决定是否实现评分—选页—注意力流水线。必须把传输成本计入评估；仅有算子加速不能证明模型加速。

## 6. 可用于工作汇报的表述

> 在 NVIDIA Blackwell 平台上完成 Quest 查询感知稀疏注意力的核心适配与验证，建立正确性、检索精度和端到端性能基线，并通过选页轨迹和配对实验分析复用机会与调度成本。进一步实现 PG-19 逐 token 困惑度对照：在首本书的 30,000 个预测目标上，Full 与 4096-token 预算 Quest 的 PPL 分别为 6.837542 和 6.832020，保持接近。后续重点是稳定工程实现、扩展多文本和多预算评测、接入 H2O，并优先推进 Qwen2.5 GQA 适配。

## 7. 参考

- [Quest 论文](https://arxiv.org/abs/2406.10774)
- [原始 PG-19 评测脚本](evaluation/pg19/ppl_eval.py)
- [原始 PPL 启动脚本](scripts/ppl_eval.sh)
- [Qwen2.5-7B 官方配置](https://huggingface.co/Qwen/Qwen2.5-7B/blob/main/config.json)
- [Qwen3.5-9B 官方配置](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json)
- [Qwen3.5 Transformers 文档](https://huggingface.co/docs/transformers/main/model_doc/qwen3_5)

## 8. 阶段一执行记录（2026-10-26）

本轮范围经确认限定为“完成阶段一并归档”，阶段二至五尚未执行。

- 完成第 4 节四项工程修复；保留原 CUDA 算法及旧实验材料。
- Python 回归共 12 项通过，包含跨设备调用、异常恢复、close 幂等、
  close 后复用和自动回收；原有数值、CUDA Graph、PG-19 与分析测试通过。
- 独立 `kernels/build-sm120-debug` 构建使用 `-O2 -g`，不含 `NDEBUG`。
  CTest 通过，内部运行 2 个 GTest（100 组 CUDA 评分配置及新增 CPU 参考测试）。
  HND/NHD 下不同头选页与部分尾页的参考结果均通过。
- `results/baseline/stability-profile-smoke` 完成 128-token、两步 decode 的
  真实模型流程验证；普通测量和两个 profiling 配置分别运行，三组审计通过。
  该结果仅验证隔离机制，不作为新的性能结论。
- 归档入口：`docs/experiments/README.md`；固定模型权重、配置、tokenizer、
  源码、Release/Debug 二进制、依赖、输入及结果的 SHA256。
  本轮归档目录为 `results/baseline/frozen-stable`。
- baseline 实现、测试、复跑说明及现有报告纳入本轮 Git 提交；
  既有子模块工作区修改及 `quest/ops/CMakeLists.txt` 的修改另行保留。

后续起点仍为阶段二：预先固定多本书子集，对 Full / 1024 / 2048 / 4096
预算进行最长 30K 的逐目标配对评测，并补充进入稀疏状态后的配对区间统计。
本轮未新增 PG-19 精度结果，也未接入 H2O 或 Qwen。
