// src/local_evaluator.rs
//
// 本地模型评估器 - 独立模块（泛型化 G = 游戏环境）。
//
// 该评估器直接在 Rust 侧用 tch-rs 加载 TorchScript 模型（CModule），
// 推理不经过 Python / GIL，因此可以在多线程 / 批量自对弈中被安全共享：
//   - 模型只在构造时加载一次（单份内存，避免 spawn 多进程重复加载 libtorch + 权重）；
//   - `evaluate` 内部在 `no_grad` 下只读推理，`LocalEvaluator` 标记为 Send + Sync，
//     可跨线程并行评估，彻底规避「Python 侧 predict_fn 被 GIL 串行化」的瓶颈。
//
// 泛型 `G: GameEnv` 可为暗棋（DarkChessEnv）、4x2 迷你（MiniDarkChessEnv）或
// 4x4（Game4x4Env）；模型前向契约统一为：
//   forward(board[B, C, H, W], scalars[B, S]) -> (policy_logits[B, A], value[B])

use anyhow::Result;
use banqi_core::core::env::GameEnv;
use banqi_core::core::mcts::{Evaluator, EvaluatorError, EvaluatorOutput};
use std::marker::PhantomData;
use tch::{CModule, Device, Kind, Tensor};

use super::batch::{batch_dims, empty_output, encode_batch, torch};

// ============================================================================
// 本地模型评估器
// ============================================================================

/// 直接使用 tch-rs CModule 加载 TorchScript 模型的评估器。
///
/// 仅依赖 `CModule`（libtorch 的 TorchScript 模块），本身是纯 Rust 结构，
/// 推理不触碰 Python 解释器，因此天然不受 GIL 影响。
pub struct LocalEvaluator<G: GameEnv> {
    model: CModule,
    device: Device,
    /// 游戏环境类型标记
    _marker: PhantomData<G>,
}

// CModule 已在 tch 中实现 Send + Sync（libtorch 推理线程安全）；
// Device 与 PhantomData 亦为 Send + Sync，因此无需手写 unsafe impl。
impl<G: GameEnv> LocalEvaluator<G> {
    pub fn new(model_path: &str, device: Device) -> Result<Self> {
        let model = CModule::load(model_path)?;
        Ok(Self {
            model,
            device,
            _marker: PhantomData,
        })
    }
}

impl<G: GameEnv> Evaluator<G> for LocalEvaluator<G> {
    fn evaluate(&self, envs: &[G]) -> Result<EvaluatorOutput, EvaluatorError> {
        if envs.is_empty() {
            return Ok(empty_output());
        }

        let dims = batch_dims(envs);
        let action_space = envs[0].action_space_size();
        let (board_data, scalar_data) = encode_batch(envs, &dims);

        tch::no_grad(|| {
            let board_tensor = Tensor::from_slice(&board_data)
                .view([
                    dims.batch as i64,
                    dims.channels as i64,
                    dims.rows as i64,
                    dims.cols as i64,
                ])
                .to_device(self.device)
                .to_kind(Kind::Float);

            let scalar_tensor = Tensor::from_slice(&scalar_data)
                .view([dims.batch as i64, dims.scalars as i64])
                .to_device(self.device)
                .to_kind(Kind::Float);

            let outputs = self
                .model
                .forward_is(&[
                    tch::IValue::Tensor(board_tensor),
                    tch::IValue::Tensor(scalar_tensor),
                ])
                .map_err(|e| EvaluatorError::new(format!("TorchScript 前向失败: {e}")))?;

            let (policy_logits, value, health_t) = torch::unwrap_outputs(outputs)?;
            torch::assemble_output(policy_logits, value, health_t, &dims, action_space)
        })
    }
}
