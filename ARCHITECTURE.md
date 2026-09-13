# ARCHITECTURE — banqi-engine 策略与推理引擎层

> **维护约定**：结构性变更（模块/公开类型/feature/bridge API）须同步更新本文并追加变更记录。

## 1. 定位

Banqi 的**策略与推理引擎层** crate：基础策略、MCTS+深度学习策略、TorchScript / ONNX 推理后端、NNUE 量化网络。独立编译、独立发布，向下依赖领域核心 `banqi-core`（`core::mcts::Evaluator` 与 `core::expectimax::nnue::{NnueEvaluate, NnueAccumulator}` 的实现方）。

上游消费方：`banqi-gui`（Tauri 桌面端）、`banqi-collector`（分布式训练数据采集 crate，2026-09-11 自 rust_4x8 拆出）、主仓库 `banqi_4x8`（切换中）。

## 2. 模块

| 模块 | 内容 | feature |
|---|---|---|
| `engine/` | `policies/`（`Policy` trait、`RandomPolicy`、`RevealFirstPolicy`）、`mcts_dl.rs`（`ModelWrapper` / `TchEvaluator<G>` / `MctsDlPolicy<G>`，Gumbel MCTS 落子） | `mcts_dl` 需 `torch` |
| `inference/` | `torchscript.rs`（`LocalEvaluator<G>`，Rust 侧批量推理，GIL-free）、`onnx/mod.rs`（`OnnxModel` / `OnnxEvaluator<G>` / `OnnxMctsPolicy<G>`，CUDA EP 可选） | 分别需 `torch` / `onnx` |
| `nnue/` | `feature.rs`（`Accumulator` / `DualAccumulator` / `FeatureDiff` / `compute_step_diff`）、`network.rs`（`NnueEvaluator` 量化前向 + `NnueBoard` 增量评估包装）、`adapter.rs`（trait 桥接 + `NnueEngineExt::from_nnue_file`） | 无（始终可用） |

## 3. feature 矩阵

`default = []`；`torch = ["tch"]`；`onnx = ["dep:ort"]`（download-binaries）；`onnx-cuda = ["onnx", "ort/cuda"]`。

构建：`cargo check --features torch` 需 `LIBTORCH` 环境变量（当前指向 miniconda 的 torch 包目录）。

## 4. 对接关系

```
banqi-core（领域核心）
   ├─ core::mcts::Evaluator<G>            ◄── TchEvaluator / OnnxEvaluator / LocalEvaluator
   └─ core::expectimax::nnue::NnueEvaluate ◄── NnueEvaluator（adapter.rs 桥接，NnueDualAcc 累加器）
```

- `ExpectimaxEngine::from_nnue_file` 不在 core（core 不感知权重格式），以扩展 trait `nnue::NnueEngineExt` 提供，调用处需 `use banqi_engine::nnue::NnueEngineExt;`。
- NNUE 增量协议：`init_accumulator(Arc<Self>) -> Box<dyn NnueAccumulator>`，累加器持有权重 Arc，`apply_step` 内部计算红黑双视角特征差分。

## 变更记录

- 2026-09-11：自 rust_4x8 拆分创立（engine/inference 迁入，NNUE 自 banqi-core 迁入并 trait 化），依赖 banqi-core path。
- 2026-09-13：`inference/torchscript.rs` 的 `LocalEvaluator` 特征维度改为从首个环境的运行时观测推导（与 `TchEvaluator` / `OnnxEvaluator` 一致），移除对 `GameEnv` 关联常量的依赖——后者已随 banqi-core 2026-09-13 变更删除，且对 4x4 / 4x2 变体不成立。
- 2026-09-13：`TchEvaluator` / `OnnxEvaluator` / `LocalEvaluator` 的动作空间宽度改为 `envs[0].action_space_size()`（随变体 352 / 112 / 40），替代 `G::action_space_size()`；模型输出与动作空间一致时不再补 `-inf`，不一致时保留原有补齐逻辑。
