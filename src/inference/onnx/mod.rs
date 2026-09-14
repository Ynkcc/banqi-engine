// src/onnx/mod.rs
//
// ONNX Runtime 推理服务（feature = "onnx"）。
//
// 用途：
//   - 自对弈：经统一入口 `run_native_match`（record_episodes=True）在 Rust 侧
//     持有 ONNX 模型，推理不经过 Python / GIL。
//   - banqi-tauri：`OnnxMctsPolicy` 作为「MCTS + ONNX」对手，无需 libtorch。
//
// 模型前向契约与 TorchScript 一致：
//   forward(board[B, C, H, W], scalars[B, S]) -> (policy_logits[B, A], value[B, 1])
// 输入名固定为 "board" / "scalars"，输出名固定为 "policy_logits" / "value"
// （由 Python 侧 banqi/checkpoint.py 的 export_onnx 导出时指定）。

use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use ort::session::Session;
use ort::value::Tensor;

use banqi_core::core::env::GameEnv;
use banqi_core::core::mcts::{
    Evaluator, EvaluatorError, EvaluatorOutput, GumbelConfig, GumbelMCTS,
};

use super::batch::{batch_dims, empty_output, encode_batch, output_from_row_logits};

// ============================================================================
// ONNX 模型封装
// ============================================================================

/// 加载并持有 ONNX 模型的推理服务。
///
/// - `Mutex<Session>`：onnxruntime 的 `Session::run` 需要 `&mut self`（内部 EP 非
///   线程安全），用互斥锁串行化推理；批量自对弈通过「合并大 batch」获得并行度。
/// - 结构为 `Send + Sync`，可跨线程共享（`Arc<OnnxModel>`）。
pub struct OnnxModel {
    session: Mutex<Session>,
    model_path: String,
}

impl OnnxModel {
    /// 加载 ONNX 模型。
    ///
    /// `device`: "cpu" 强制 CPU；"cuda" / "auto" 在启用 `onnx-cuda` feature 时
    /// 尝试 CUDA EP（失败自动回退 CPU），否则直接使用 CPU。
    pub fn new(model_path: &str, device: &str) -> Result<Self, String> {
        let prefer_gpu = matches!(device, "cuda" | "auto");
        let session = build_session(model_path, prefer_gpu)?;
        Ok(Self {
            session: Mutex::new(session),
            model_path: model_path.to_string(),
        })
    }

    pub fn model_path(&self) -> &str {
        &self.model_path
    }

    /// 批量前向推理。
    ///
    /// 参数为扁平特征（由 `GameEnv::encode_resnet_features_flat_into` 填充）：
    ///   - board_data:   batch * channels * rows * cols
    ///   - scalars_data: batch * scalar_count
    ///
    /// 返回 `(policy_logits[B, A_model], values[B], health[B, K] | None)`
    /// （A_model 为模型输出动作维度，可能小于环境动作空间，由 `OnnxEvaluator` 负责补齐；
    /// health 仅在模型带血量差异头时返回 Some，否则 None）。
    pub fn run(
        &self,
        board_data: &[f32],
        scalars_data: &[f32],
        batch_size: usize,
        board_channels: usize,
        board_rows: usize,
        board_cols: usize,
        scalar_count: usize,
    ) -> Result<(Vec<Vec<f32>>, Vec<f32>, Option<Vec<Vec<f32>>>), String> {
        let board_tensor = Tensor::from_array((
            [batch_size, board_channels, board_rows, board_cols],
            board_data.to_vec().into_boxed_slice(),
        ))
        .map_err(|e| format!("构建 board 张量失败: {e}"))?;

        let scalars_tensor = Tensor::from_array((
            [batch_size, scalar_count],
            scalars_data.to_vec().into_boxed_slice(),
        ))
        .map_err(|e| format!("构建 scalars 张量失败: {e}"))?;

        // SessionOutputs 借用自 Session，需让互斥锁守卫存活到提取完输出为止。
        let mut session = self
            .session
            .lock()
            .map_err(|e| format!("ONNX 会话锁中毒: {e}"))?;
        let outputs = session
            .run(ort::inputs![
                "board" => board_tensor,
                "scalars" => scalars_tensor,
            ])
            .map_err(|e| format!("ONNX 推理失败: {e}"))?;

        // 输出顺序与 export_onnx 的 output_names 一致：policy_logits, value
        let model_action = extract_dim(&outputs[0], 1, "policy_logits")?;
        let mut logits_flat = vec![0.0f32; batch_size * model_action];
        copy_tensor(&outputs[0], &mut logits_flat)?;
        let logits: Vec<Vec<f32>> = logits_flat
            .chunks(model_action)
            .map(|c| c.to_vec())
            .collect();

        let mut values = vec![0.0f32; batch_size];
        copy_tensor(&outputs[1], &mut values)?;

        // 血量差异头：模型输出第三个输出时解析为 [B, K] 分桶 logits，否则 None。
        let health = if outputs.len() >= 3 {
            let k = extract_dim(&outputs[2], 1, "health")?;
            let mut health_flat = vec![0.0f32; batch_size * k];
            copy_tensor(&outputs[2], &mut health_flat)?;
            Some(
                health_flat
                    .chunks(k)
                    .map(|c| c.to_vec())
                    .collect::<Vec<Vec<f32>>>(),
            )
        } else {
            None
        };

        Ok((logits, values, health))
    }
}

// ============================================================================
// 会话构建（CUDA EP 为可选项，失败自动回退 CPU）
// ============================================================================

#[cfg(feature = "onnx-cuda")]
fn build_cuda_session(model_path: &str) -> Result<Session, String> {
    use ort::execution_providers::CUDAExecutionProvider;
    let provider = CUDAExecutionProvider::default()
        .build()
        .map_err(|e| format!("CUDA EP 构建失败: {e}"))?;
    Session::builder()
        .and_then(|mut b| b.with_execution_providers([provider]))
        .and_then(|mut b| b.commit_from_file(model_path))
        .map_err(|e| format!("加载 ONNX 模型（CUDA EP）失败 ({model_path}): {e}"))
}

fn build_session(model_path: &str, prefer_gpu: bool) -> Result<Session, String> {
    #[cfg(feature = "onnx-cuda")]
    if prefer_gpu {
        match build_cuda_session(model_path) {
            Ok(s) => {
                println!("[onnx] 已使用 CUDA EP: {model_path}");
                return Ok(s);
            }
            Err(e) => eprintln!("[onnx] CUDA EP 不可用，回退 CPU: {e}"),
        }
    }
    #[cfg(not(feature = "onnx-cuda"))]
    let _ = prefer_gpu;
    Session::builder()
        .and_then(|mut b| b.commit_from_file(model_path))
        .map_err(|e| format!("加载 ONNX 模型失败 ({model_path}): {e}"))
}

// ============================================================================
// 输出张量辅助函数
// ============================================================================

fn extract_dim(value: &ort::value::Value, dim: usize, name: &str) -> Result<usize, String> {
    let (shape, _data) = value
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("提取输出 {name} 失败: {e}"))?;
    shape
        .get(dim)
        .copied()
        .map(|v| v as usize)
        .ok_or_else(|| format!("输出 {name} 维度不足: {:?}", &shape[..]))
}

fn copy_tensor(value: &ort::value::Value, out: &mut [f32]) -> Result<(), String> {
    let (_shape, data) = value
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("提取输出张量失败: {e}"))?;
    let n = out.len().min(data.len());
    out[..n].copy_from_slice(&data[..n]);
    Ok(())
}

// ============================================================================
// ONNX 评估器（泛型，适配任意变体）
// ============================================================================

/// 基于 ONNX 模型的批量评估器，与 `crate::inference::torchscript::LocalEvaluator` /
/// `TchEvaluator` 等价（后端为 ONNX Runtime，不依赖 libtorch）。
pub struct OnnxEvaluator<G: GameEnv> {
    pub model: Arc<OnnxModel>,
    pub _marker: PhantomData<G>,
}

impl<G: GameEnv> OnnxEvaluator<G> {
    pub fn new(model: Arc<OnnxModel>) -> Self {
        Self {
            model,
            _marker: PhantomData,
        }
    }
}

impl<G: GameEnv> Evaluator<G> for OnnxEvaluator<G> {
    fn evaluate(&self, envs: &[G]) -> Result<EvaluatorOutput, EvaluatorError> {
        if envs.is_empty() {
            return Ok(empty_output());
        }

        let dims = batch_dims(envs);
        let action_space = envs[0].action_space_size();
        let (board_data, scalars_data) = encode_batch(envs, &dims);

        let (raw_logits, values, health) = self
            .model
            .run(
                &board_data,
                &scalars_data,
                dims.batch,
                dims.channels,
                dims.rows,
                dims.cols,
                dims.scalars,
            )
            .map_err(EvaluatorError::from)?;

        output_from_row_logits(&raw_logits, dims.batch, action_space, values, health)
    }

    fn evaluate_logits(&self, envs: &[G]) -> Result<EvaluatorOutput, EvaluatorError> {
        self.evaluate(envs)
    }
}

// ============================================================================
// MCTS + ONNX 策略（供 banqi-tauri 等单步决策使用）
// ============================================================================

/// MCTS + ONNX 深度学习策略，每次调用创建新 MCTS 实例（与 MctsDlPolicy 一致）。
pub struct OnnxMctsPolicy<G: GameEnv> {
    model: Arc<OnnxModel>,
    num_simulations: usize,
    _marker: PhantomData<G>,
}

impl<G: GameEnv> OnnxMctsPolicy<G> {
    pub fn new(model: Arc<OnnxModel>, _env: &G, num_simulations: usize) -> Self {
        Self {
            model,
            num_simulations,
            _marker: PhantomData,
        }
    }

    pub fn set_iterations(&mut self, sims: usize) {
        self.num_simulations = sims.max(1);
    }

    pub fn choose_action(&self, env: &G) -> Result<Option<usize>, EvaluatorError> {
        onnx_choose_action_once(&self.model, env, self.num_simulations)
    }
}

/// 为给定环境选择最佳动作（每次创建新 MCTS）；评估失败返回 Err。
pub fn onnx_choose_action_once<G: GameEnv>(
    model: &Arc<OnnxModel>,
    env: &G,
    num_simulations: usize,
) -> Result<Option<usize>, EvaluatorError> {
    let evaluator = OnnxEvaluator::<G>::new(model.clone());
    let config = GumbelConfig {
        num_simulations,
        max_considered_actions: 16,
        c_scale: 1.0,
        gumbel_scale: 1.0,
        ..Default::default()
    };
    let mut mcts = GumbelMCTS::new(env, &evaluator, config);
    Ok(mcts.run()?.map(|result| result.action))
}
