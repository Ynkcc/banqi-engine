//! # Banqi Engine — 策略与推理引擎层
//!
//! - `engine`:    策略引擎（Random / RevealFirst / CaptureFirst 基础策略；MCTS+DL 策略需 `torch`）
//! - `inference`: 神经网络推理后端（TorchScript 需 `torch`；ONNX Runtime 需 `onnx`）
//! - `nnue`:      NNUE 量化网络（增量累加器 + 前向评估）及 `banqi-core`
//!   `NnueEvaluate`/`NnueAccumulator` trait 的实现桥接
//!
//! 领域核心（游戏环境 / MCTS / Expectimax / 特征提取）在上游 crate `banqi-core`。

pub mod engine;
pub mod inference;
pub mod nnue;
