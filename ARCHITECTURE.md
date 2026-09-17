# ARCHITECTURE — banqi-engine 策略与推理引擎层

> **维护约定**：结构性变更（模块/公开类型/feature/bridge API）须同步更新本文并追加变更记录。

## 1. 定位

Banqi 的**策略与推理引擎层** crate：基础策略、MCTS+深度学习策略、TorchScript / ONNX 推理后端、NNUE 量化网络。独立编译、独立发布，向下依赖领域核心 `banqi-core`（`core::mcts::Evaluator` 与 `core::expectimax::nnue::{NnueEvaluate, NnueAccumulator}` 的实现方）。

上游消费方：`banqi-gui`（Tauri 桌面端）、`banqi-collector`（分布式训练数据采集 crate，2026-09-11 自 rust_4x8 拆出）、主仓库 `banqi_4x8`（切换中）。

## 2. 模块

| 模块 | 内容 | feature |
|---|---|---|
| `engine/` | `policies/`（`Policy` trait、`RandomPolicy`、`RevealFirstPolicy`、`CaptureFirstPolicy`）、`mcts_dl.rs`（`ModelWrapper` / `TchEvaluator<G>` / `MctsDlPolicy<G>`，Gumbel MCTS 落子） | `mcts_dl` 需 `torch` |
| `inference/` | `batch.rs`（Evaluator 批量组装公共实现：维度推导 / 特征编码 / 输出对齐与 TorchScript 输出解包）、`torchscript.rs`（`LocalEvaluator<G>`，Rust 侧批量推理，GIL-free）、`onnx/mod.rs`（`OnnxModel` / `OnnxEvaluator<G>` / `OnnxMctsPolicy<G>`，CUDA EP 可选） | `batch.rs` 需 `torch` 或 `onnx`；后两者分别需 `torch` / `onnx` |
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
- 2026-09-15：Evaluator 去重与错误化：①新增 `inference/batch.rs` 收敛三后端共用的「维度推导 / 特征编码 / 输出对齐 / TorchScript 输出解包」样板（原为 4 份拷贝）；②`LocalEvaluator` / `TchEvaluator` 改用 `encode_resnet_features_flat_into` 复用缓冲（原 `TchEvaluator` 每环境两次 `get_resnet_state` 分配）并统一按 min 截断 + 补 `-inf`（宽度不一致时打印一次提示）；③三个 `evaluate` 改为 `Result<_, EvaluatorError>`：删除 torch 侧的 `expect`/5 处 `panic!`，并**删除 ONNX 侧「推理失败退化为均匀 logits」的静默 fallback**（改为 `Err`，由调用方决定重试/终止）；④`MctsDlPolicy::choose_action` / `choose_action_once` / `OnnxMctsPolicy::choose_action` / `onnx_choose_action_once` 改为返回 `Result<Option<usize>, EvaluatorError>`；⑤`ModelWrapper.gate` 锁中毒改为恢复而非 panic。
- 2026-09-17：`policies/reveal_first.rs` 新增 `CaptureFirstPolicy`（优先吃明子 → 无吃子时优先翻棋 → 否则随机静走），与 `RevealFirstPolicy` 同文件并通过私有 `bucket_actions` 复用 `DarkChessEnv::generate_moves` 的语义分桶；`RevealFirstPolicy` 行为不变（等价重构）。GUI 新增 `OpponentType::CaptureFirst`（对手 "CaptureFirst"）与前端下拉项「电脑 (优先吃子)」。
- 2026-09-15：`OnnxModel` 单会话改会话池：`session: Mutex<Session>` → `sessions: Vec<Mutex<Session>>` + `next: AtomicUsize`（轮转分配）；新增 `OnnxModel::with_sessions(path, device, n)` 与 `session_count()`，`new()` 保持单会话语义（GUI / 离线评估不受影响）。每个会话的 ORT intra-op 线程数取 `核数 / 会话数`（至少 1），避免多会话各自开满线程超额订阅；`build_session` / `build_cuda_session` 增加 `intra_threads` 参数（`SessionBuilder::with_*` 的错误类型携带 builder，故改为逐段 `map_err`，新增 `builder_with_intra_threads` 辅助）。动机：ORT 的 `Session::run` 需 `&mut self`，单会话会把所有并发推理挤成一条通道；4x2 批量自对弈 12 线程实测会话数 1→12 对应 18→160 局/s。
