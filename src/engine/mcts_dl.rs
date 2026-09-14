// src/ai/mcts_dl.rs
//! MCTS + 深度学习策略（支持搜索树复用）- 同步版本，基于 `GameConfig` 泛化。
//!
//! 特征维度由首个环境的运行时观测推导（`config` 驱动），动作空间取
//! `action_space_size()`，使同一份 TorchScript 推理代码服务所有变体（4x8 / 4x4 / 4x2）。
//!
//! 提供：
//! - `ModelWrapper`：加载 TorchScript `.pt` 模型（`CModule`）
//! - `TchEvaluator<G>`：实现 `Evaluator<G>`，批量前向 `(board, scalars) -> (logits, value)`
//! - `MctsDlPolicy<G>`：基于 Gumbel MCTS 的落子策略
//!
//! 使用流程：
//! 1. 加载模型 -> `ModelWrapper::load_from_file`
//! 2. 创建策略 -> `MctsDlPolicy::<G>::new(model, &env, sims)`
//! 3. 需要选择动作时调用 `choose_action(&env)`

use banqi_core::core::env::GameEnv;
use banqi_core::core::mcts::{
    Evaluator, EvaluatorError, EvaluatorOutput, GumbelConfig, GumbelMCTS,
};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use tch::{CModule, Device, Tensor};

use crate::inference::batch::{batch_dims, empty_output, encode_batch, torch};

// ---------------- Model 封装 ----------------

pub struct ModelWrapper {
    model: CModule,
    device: Device,
    gate: Mutex<()>, // 串行化前向以保线程安全
}

impl ModelWrapper {
    pub fn load_from_file(path: &str) -> Result<Self, String> {
        let device = Device::Cpu;
        let model = CModule::load(path).map_err(|e| format!("模型加载失败: {}", e))?;
        Ok(Self {
            model,
            device,
            gate: Mutex::new(()),
        })
    }

    pub fn get_device(&self) -> Device {
        self.device
    }
}

// 由于内部有互斥锁保护，允许跨线程共享
unsafe impl Send for ModelWrapper {}
unsafe impl Sync for ModelWrapper {}

// ---------------- Evaluator (泛型，基于 GameEnv) ----------------

/// 基于 `GameEnv` 关联常量适配任意变体的批量评估器。
pub struct TchEvaluator<G: GameEnv> {
    pub model: Arc<ModelWrapper>,
    pub _marker: PhantomData<G>,
}

impl<G: GameEnv> TchEvaluator<G> {
    pub fn new(model: Arc<ModelWrapper>) -> Self {
        Self {
            model,
            _marker: PhantomData,
        }
    }
}

impl<G: GameEnv> Evaluator<G> for TchEvaluator<G> {
    fn evaluate(&self, envs: &[G]) -> Result<EvaluatorOutput, EvaluatorError> {
        if envs.is_empty() {
            return Ok(empty_output());
        }

        let dims = batch_dims(envs);
        let action_space = envs[0].action_space_size();
        let (board_flat, scalars_flat) = encode_batch(envs, &dims);

        // 串行化前向：tch 的 CModule 未声明线程安全；批量自对弈靠合并大 batch 取并行度。
        let _guard = self.model.gate.lock().unwrap_or_else(|e| e.into_inner());
        tch::no_grad(|| {
            let board_t = Tensor::from_slice(&board_flat)
                .to_device(self.model.device)
                .view([
                    dims.batch as i64,
                    dims.channels as i64,
                    dims.rows as i64,
                    dims.cols as i64,
                ]);

            let scalars_t = Tensor::from_slice(&scalars_flat)
                .to_device(self.model.device)
                .view([dims.batch as i64, dims.scalars as i64]);

            let outputs = self
                .model
                .model
                .forward_is(&[tch::IValue::Tensor(board_t), tch::IValue::Tensor(scalars_t)])
                .map_err(|e| EvaluatorError::new(format!("TorchScript 前向失败: {e}")))?;

            let (policy_logits, value_t, health_t) = torch::unwrap_outputs(outputs)?;
            torch::assemble_output(policy_logits, value_t, health_t, &dims, action_space)
        })
    }

    fn evaluate_logits(&self, envs: &[G]) -> Result<EvaluatorOutput, EvaluatorError> {
        self.evaluate(envs)
    }
}

// ---------------- 策略对象（泛型，每次创建新 MCTS）----------------

/// MCTS + 深度学习策略
///
/// 为了避免生命周期问题，每次调用 choose_action 时创建新的 MCTS 实例
/// 虽然失去了搜索树复用的优势，但实现更简单可靠
pub struct MctsDlPolicy<G: GameEnv> {
    model: Arc<ModelWrapper>,
    num_simulations: usize,
    _marker: PhantomData<G>,
}

impl<G: GameEnv> MctsDlPolicy<G> {
    pub fn new(model: Arc<ModelWrapper>, _env: &G, num_simulations: usize) -> Self {
        Self {
            model,
            num_simulations,
            _marker: PhantomData,
        }
    }

    pub fn set_iterations(&mut self, sims: usize) {
        self.num_simulations = sims.max(1);
    }

    /// 选择动作（每次创建新 MCTS）；评估失败时返回 Err，由调用方决定重试或终止。
    pub fn choose_action(&self, env: &G) -> Result<Option<usize>, EvaluatorError> {
        choose_action_once(&self.model, env, self.num_simulations)
    }
}

// ---------------- 简化的一次性策略 ----------------

/// 为给定环境选择最佳动作（每次创建新 MCTS）
pub fn choose_action_once<G: GameEnv>(
    model: &Arc<ModelWrapper>,
    env: &G,
    num_simulations: usize,
) -> Result<Option<usize>, EvaluatorError> {
    let evaluator = TchEvaluator::<G>::new(model.clone());
    let config = GumbelConfig {
        num_simulations,
        max_considered_actions: 16,
        c_scale: 1.0,
        gumbel_scale: 1.0,
        ..Default::default()
    };

    let mut mcts = GumbelMCTS::new(env, &evaluator, config);
    // 只返回动作索引，忽略完整搜索结果
    Ok(mcts.run()?.map(|result| result.action))
}
