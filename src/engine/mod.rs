//! 策略引擎层
//!
//! 基础策略（Random / RevealFirst）与 MCTS+深度学习策略（`torch` feature）。
//! 走子生成与 Expectimax 强引擎位于 `banqi-core`。

pub mod policies;

#[cfg(feature = "torch")]
pub mod mcts_dl;

pub use policies::{Policy, RandomPolicy, RevealFirstPolicy};

#[cfg(feature = "torch")]
pub use mcts_dl::{MctsDlPolicy, ModelWrapper, TchEvaluator};
