# 代码审查与分析报告（2026-03-12）

## 1. 审查范围

- Step1 基线评测流程
- Step2 `Accelerate + DeepSpeed ZeRO-2` 训练流程
- Step3 分析流程
- 分布式评测、对比学习损失、启动脚本与运行文档

审查文件：

- `src/trainers/router_contrastive_trainer.py`
- `src/evaluation/evaluator.py`
- `src/losses/contrastive.py`
- `src/trainers/step1_baseline_trainer.py`
- `scripts/*.sh`
- `README.md`

## 2. 总体结论

- 当前实现已经完成了 Step1/Step2/Step3 的主流程闭环，结构上也符合“脚本仅入口、trainer 承载逻辑”的项目化要求。
- 但存在一个高优先级稳定性问题（分布式评测空分片崩溃），以及两个中优先级工程风险（分布式数据一致性与断点恢复能力）。

## 3. 发现的问题（按严重级别排序）

### [P1] 分布式评测在小验证集场景可能直接崩溃

- 位置：`src/evaluation/evaluator.py:67-95`
- 现象：
  - 在 `world_size > eval_samples` 或某些分片为空时，`local_anchor/local_positive` 可能为空列表。
  - 随后执行 `torch.cat(local_anchor, dim=0)` 会抛异常。
- 影响：
  - Step1 和 Step2 的 post-eval 在多机多卡小样本调试场景下不稳定，直接中断流程。
- 建议：
  - 在 `torch.cat` 前处理空分片：
    - 方案 A：先 all_gather 每个 rank 的样本数，允许空 rank 上报空 payload；
    - 方案 B：空分片时构造形状为 `(0, hidden_dim)` 的占位 tensor；
    - 方案 C：仅主进程汇总非空分片。

### [P2] Step2 使用 `device_specific=True` 设种子，存在跨进程数据采样不一致风险

- 位置：`src/trainers/router_contrastive_trainer.py:99`
- 现象：
  - 当前 DataLoader 使用 `shuffle=True`，但没有显式 DistributedSampler；
  - 同时每个进程使用不同随机种子，可能导致每个进程抽样顺序不同步。
- 影响：
  - 在多进程训练中可能出现重复样本或分片不均，影响统计效率与复现实验的一致性。
- 建议：
  - 对训练数据使用统一的 distributed sampler/sharding 机制，或改为全局一致随机种子并让框架负责分片；
  - 最少应在日志中输出每进程样本覆盖统计，便于快速确认是否存在重复覆盖。

### [P2] Checkpoint 仅保存 router 与 logit_scale，缺失恢复训练所需状态

- 位置：`src/trainers/router_contrastive_trainer.py:83-90`
- 现象：
  - 当前 checkpoint 不包含 optimizer/scheduler/global_step/RNG 状态。
- 影响：
  - 训练中断后无法无损恢复，学习率调度会漂移，影响实验可重复性和长任务稳定性。
- 建议：
  - 增加“训练态 checkpoint”格式：
    - `router_state_dict`
    - `logit_scale`
    - `optimizer_state`
    - `scheduler_state`
    - `global_step/epoch`
    - `rng_state`
  - 同时保留当前“轻量导出”格式用于部署。

## 4. 其他观察

- `README.md` 的多机示例可读性已较好，但 `--num_processes` 的使用建议在文档里补充“按当前集群配置核对 accelerate 参数语义（每机/总进程）”，避免误启动。
- `scripts/step2_train_router.sh` 的 `ACCELERATE_EXTRA_ARGS` 采用字符串分割，对复杂引号参数不够鲁棒；当前可用，但建议后续改为数组式传参文档规范。

## 5. 优先修复顺序建议

1. 修复分布式评测空分片崩溃（P1）。
2. 明确并固定分布式数据采样一致性策略（P2）。
3. 增加可恢复训练 checkpoint（P2）。

## 6. 测试缺口

建议补充以下自动化测试或最小集成验证：

- 多进程评测（`world_size > dataset_size`）不崩溃。
- Step2 在 `gradient_accumulation_steps > 1` 下的 loss/logit_scale 更新与日志一致性。
- 中断后恢复训练（resume）前后学习率和全局步数连续性。
