# moe2encoder 使用文档（Step1 / Step2 / Step3）

## 1. 环境准备

```bash
cd /Users/libozhou/Desktop/moe2encoder
pip install -r requirements.txt
```

默认 Step2 后端为 `Accelerate + DeepSpeed ZeRO-2`，配置文件：

- `configs/accelerate/deepspeed_zero2.yaml`
- `configs/deepspeed/zero2_router_only.json`

## 2. 数据准备

训练/评测数据为 `jsonl`，每行至少包含：

- `anchor_text`
- `positive_text`

可选字段：

- `group_id`
- `prompt_type`

完整格式参考：

- `configs/data/dataset_schema.md`

## 3. Step1：冻结基线评测

单机单卡：

```bash
./scripts/step1_baseline.sh \
  --config configs/train/router_only_qwen35a3b.yaml
```

单机多卡：

```bash
NPROC_PER_NODE=8 ./scripts/step1_baseline.sh \
  --config configs/train/router_only_qwen35a3b.yaml
```

输出目录（默认）：

- `outputs/.../step1_baseline/metrics.json`
- `outputs/.../step1_baseline/routing_stats.json`
- `outputs/.../step1_baseline/summary.json`

说明：

- 现已支持分布式评测空分片容错（例如 `world_size > eval_samples` 的场景）。

## 4. Step2：Router-only 对比学习训练

### 4.1 从头训练

```bash
ACCELERATE_CONFIG=configs/accelerate/deepspeed_zero2.yaml \
./scripts/step2_train_router.sh \
  --config configs/train/router_only_qwen35a3b.yaml \
  --run_post_eval
```

### 4.2 从 checkpoint 恢复训练（新增）

`--resume_from` 传入 `state_*` 目录路径：

```bash
ACCELERATE_CONFIG=configs/accelerate/deepspeed_zero2.yaml \
./scripts/step2_train_router.sh \
  --config configs/train/router_only_qwen35a3b.yaml \
  --resume_from outputs/qwen35a3b_router_only/step2_train/checkpoints/state_step_1000 \
  --run_post_eval
```

也可以从最终状态恢复继续训练：

```bash
ACCELERATE_CONFIG=configs/accelerate/deepspeed_zero2.yaml \
./scripts/step2_train_router.sh \
  --config configs/train/router_only_qwen35a3b.yaml \
  --resume_from outputs/qwen35a3b_router_only/step2_train/checkpoints/state_final
```

### 4.3 Step2 输出说明

训练中每到 `save_every_steps` 会生成：

- `checkpoints/state_step_<global_step>/`  
  含 Accelerate/DeepSpeed 训练态（模型、优化器、调度器、随机状态）与 `trainer_state.json`
- `checkpoints/router_step_<global_step>.pt`  
  轻量 router 导出（仅 router + logit_scale + meta）

训练结束会生成：

- `checkpoints/state_final/`
- `checkpoints/router_final.pt`
- `train_log.jsonl`
- `step2_post_eval/*.json`（开启 `--run_post_eval` 时）

### 4.4 `trainer_state.json` 字段

- `global_step`
- `epoch`
- `batch_in_epoch`
- `max_steps`
- `completed`（仅 `state_final`）

## 5. Step3：训练后分析

```bash
./scripts/step3_analysis.sh \
  --baseline_dir outputs/qwen35a3b_router_only/step1_baseline \
  --post_dir outputs/qwen35a3b_router_only/step2_train/step2_post_eval \
  --output_dir outputs/qwen35a3b_router_only/step3_analysis
```

输出：

- `metrics_diff.json`
- `routing_diff.json`
- `analysis_report.md`

## 6. 多机多卡启动（Step2）

示例：2 机 x 8 卡，主节点 IP 为 `10.0.0.1`。

node0：

```bash
ACCELERATE_CONFIG=configs/accelerate/deepspeed_zero2.yaml \
ACCELERATE_EXTRA_ARGS="--num_machines 2 --num_processes 16 --machine_rank 0 --main_process_ip 10.0.0.1 --main_process_port 29500" \
./scripts/step2_train_router.sh \
  --config configs/train/router_only_qwen35a3b.yaml
```

node1：

```bash
ACCELERATE_CONFIG=configs/accelerate/deepspeed_zero2.yaml \
ACCELERATE_EXTRA_ARGS="--num_machines 2 --num_processes 16 --machine_rank 1 --main_process_ip 10.0.0.1 --main_process_port 29500" \
./scripts/step2_train_router.sh \
  --config configs/train/router_only_qwen35a3b.yaml
```

## 7. 常见检查项

- `freeze_report.json` 中 `trainable_params` 必须大于 0。
- 训练日志 `train_log.jsonl` 的 `step`、`lr`、`logit_scale` 应连续变化。
- resume 后首条日志的 `step` 应大于或等于断点前最后 step。
