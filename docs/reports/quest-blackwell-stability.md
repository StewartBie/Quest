# Blackwell baseline 阶段一验收记录

日期：2026-10-26。范围：稳定当前基线、进程隔离与归档。

## 修复

`native.py` 在输入设备上下文内评分、创建计划和解码，并按计划记录的
设备释放资源。`weakref.finalize` 不保留目标对象，支持自动释放；显式
`close()` 幂等且可在之后重新创建计划，`with` 支持确定性清理。

`run.py --profile` 启动一个普通测量进程，再为每个长度、预算和后端
配置启动新 Python 解释器。父进程保存 `processes.json` 和完成/失败状态，
每个 worker 单独保存 manifest、精确输入及测量记录。新增 manifest 记录
进程 ID。审计工具兼容旧目录以及新的 profile-only worker。

CPU reference 移除不存在的断言变量，检查 MHA、每头等量选页以及尾页
有效长度。新增零 Q/K、非均匀 V 的解析期望测试，覆盖 HND/NHD、每头
不同选页、跳过中间页和单 token 尾页。

## 验证

```bash
/home/bzq/.local/envs/quest/bin/python -m unittest \
  evaluation.baseline.test_baseline evaluation.baseline.test_pg19 \
  evaluation.baseline.test_analysis evaluation.baseline.test_processes
ctest --test-dir kernels/build-sm120-debug --output-on-failure
```

Python 12 项通过。跨设备测试使用 GPU 1 输入、GPU 0 为当前设备；
资源释放包装记录每次实际释放的设备和次数，检查自动回收及复用。
Debug 编译使用 `-O2 -g`，没有 `NDEBUG`；CTest 通过，内部运行
2 个 GTest，其中 CUDA 测试包含 100 组评分配置。完整日志随归档保存。

真实模型进程隔离验证：

```bash
/home/bzq/.local/envs/quest/bin/python -m evaluation.baseline.run benchmark \
  --output results/baseline/stability-profile-smoke \
  --lengths 128 --budgets 0 4 --decode-steps 2 --repeats 1 \
  --backends native --profile --profile-level minimal
```

父进程状态 complete；`measure` 有两条 none 记录，两个 profile 子目录
各有一条 profile 记录，三个目录审计全部通过。两步短运行仅用于流程
验证，不用于报告加速比。复跑时必须使用新的输出目录。

## 归档

`results/baseline/frozen-stable` 保存源码和 Release/Debug 产物、Debug
测试日志、环境依赖、模型文件哈希及实验产物哈希。实际模型和结果载荷
保留在本地，需与归档一并复制；索引不代替数据备份。
清单及旧实验复跑入口见 [实验归档索引](../experiments/README.md)。

本轮未改变旧实验结论，未执行阶段二至五。硬件计数器权限限制仍然存在。
