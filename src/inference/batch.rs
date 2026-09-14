//! Evaluator 批量组装公共实现：特征编码、输出对齐与 TorchScript 输出解包。
//!
//! 三个推理后端（`LocalEvaluator` / `TchEvaluator` / `OnnxEvaluator`）此前各自复制了
//! 「维度推导 → 特征编码 → 输出拆分 → `EvaluatorOutput` 组装」样板；本模块收敛为唯一实现，
//! 并把推理期错误统一为 `Result<_, EvaluatorError>`（不再 panic，也不静默退化）。

use banqi_core::core::env::GameEnv;
use banqi_core::core::mcts::{EvaluatorError, EvaluatorOutput};

/// 批量特征维度（由首个环境的运行时观测推导，适配 4x8 / 4x4 / 4x2）。
#[derive(Debug, Clone, Copy)]
pub(crate) struct BatchDims {
    pub batch: usize,
    pub channels: usize,
    pub rows: usize,
    pub cols: usize,
    pub scalars: usize,
}

/// 由首个环境推导批量特征维度。
pub(crate) fn batch_dims<G: GameEnv>(envs: &[G]) -> BatchDims {
    let obs = envs[0].get_resnet_state();
    BatchDims {
        batch: envs.len(),
        channels: obs.board.shape()[0],
        rows: obs.board.shape()[1],
        cols: obs.board.shape()[2],
        scalars: obs.scalars.len(),
    }
}

/// 将环境批次编码为扁平特征 `(board, scalars)`；复用临时缓冲，避免逐环境堆分配。
pub(crate) fn encode_batch<G: GameEnv>(envs: &[G], dims: &BatchDims) -> (Vec<f32>, Vec<f32>) {
    let mut board_data = Vec::with_capacity(dims.batch * dims.channels * dims.rows * dims.cols);
    let mut scalars_data = Vec::with_capacity(dims.batch * dims.scalars);
    let mut board_buf = Vec::new();
    let mut scalar_buf = Vec::new();
    for env in envs {
        env.encode_resnet_features_flat_into(&mut board_buf, &mut scalar_buf);
        board_data.extend_from_slice(&board_buf);
        scalars_data.extend_from_slice(&scalar_buf);
    }
    (board_data, scalars_data)
}

/// 空批次的统一早退输出。
pub(crate) fn empty_output() -> EvaluatorOutput {
    EvaluatorOutput {
        logits: Vec::new(),
        values: Vec::new(),
        health: None,
    }
}

/// 把一行原始 logits 对齐到 `action_space`：宽度不一致时按 min 截断并以 -inf 补齐
/// （-inf 位置在合法动作掩码下无效，不影响搜索）。宽度不一致会打印一次提示，
/// 便于定位「模型与变体不匹配」。
fn align_row(
    row: &[f32],
    action_space: usize,
    model_action: usize,
    mismatch_warned: &mut bool,
) -> Vec<f32> {
    if model_action != action_space && !*mismatch_warned {
        *mismatch_warned = true;
        eprintln!(
            "⚠️ [evaluator] 模型动作维度 {model_action} 与环境动作空间 {action_space} 不一致：按 min 截断并补 -inf"
        );
    }
    let mut padded = vec![f32::NEG_INFINITY; action_space];
    let n = model_action.min(action_space);
    padded[..n].copy_from_slice(&row[..n]);
    padded
}

/// 由**扁平** logits（`batch * model_action`）组装输出（TorchScript / Tch 路径）。
#[cfg(feature = "torch")]
pub(crate) fn output_from_flat_logits(
    logits_flat: &[f32],
    model_action: usize,
    batch: usize,
    action_space: usize,
    values: Vec<f32>,
    health: Option<Vec<Vec<f32>>>,
) -> Result<EvaluatorOutput, EvaluatorError> {
    if model_action == 0 {
        return Err(EvaluatorError::new("模型 policy 输出缺少动作维度"));
    }
    if values.len() != batch {
        return Err(EvaluatorError::new(format!(
            "模型 value 输出长度 {} 与 batch {batch} 不符",
            values.len()
        )));
    }
    let mut mismatch_warned = false;
    let logits = logits_flat
        .chunks(model_action)
        .take(batch)
        .map(|row| align_row(row, action_space, model_action, &mut mismatch_warned))
        .collect();
    Ok(EvaluatorOutput {
        logits,
        values,
        health,
    })
}

/// 由**逐行** logits 组装输出（ONNX 路径：模型输出已按行拆分）。
#[cfg(feature = "onnx")]
pub(crate) fn output_from_row_logits(
    rows: &[Vec<f32>],
    batch: usize,
    action_space: usize,
    values: Vec<f32>,
    health: Option<Vec<Vec<f32>>>,
) -> Result<EvaluatorOutput, EvaluatorError> {
    let model_action = rows.first().map_or(0, |r| r.len());
    if model_action == 0 {
        return Err(EvaluatorError::new("模型 policy 输出缺少动作维度"));
    }
    if rows.len() != batch || values.len() != batch {
        return Err(EvaluatorError::new(format!(
            "模型输出行数 {} / value 长度 {} 与 batch {batch} 不符",
            rows.len(),
            values.len()
        )));
    }
    let mut mismatch_warned = false;
    let logits = rows
        .iter()
        .map(|row| align_row(row, action_space, model_action, &mut mismatch_warned))
        .collect();
    Ok(EvaluatorOutput {
        logits,
        values,
        health,
    })
}

#[cfg(feature = "torch")]
pub(crate) mod torch {
    use banqi_core::core::mcts::{EvaluatorError, EvaluatorOutput};
    use tch::{Device, IValue, Tensor};

    use super::{output_from_flat_logits, BatchDims};

    /// 从 tuple 尾部弹出一个 Tensor，类型不符时报错（不 panic）。
    fn pop_tensor(tensors: &mut Vec<IValue>, what: &str) -> Result<Tensor, EvaluatorError> {
        match tensors.pop() {
            Some(IValue::Tensor(t)) => Ok(t),
            Some(_) => Err(EvaluatorError::new(format!(
                "TorchScript 输出 {what} 不是 Tensor"
            ))),
            None => Err(EvaluatorError::new(format!("TorchScript 输出缺少 {what}"))),
        }
    }

    /// 解包 TorchScript 前向结果：兼容 2 输出（旧模型）与 3 输出（带血量差异头）。
    ///
    /// 输出顺序约定：`(policy_logits, [health], value)`，与旧实现一致（从尾部弹出）。
    pub(crate) fn unwrap_outputs(
        outputs: IValue,
    ) -> Result<(Tensor, Tensor, Option<Tensor>), EvaluatorError> {
        let mut tensors = match outputs {
            IValue::Tuple(t) => t,
            _ => {
                return Err(EvaluatorError::new(
                    "TorchScript 输出应为 tuple(policy_logits[, health], value)",
                ));
            }
        };
        let n = tensors.len();
        if n != 2 && n != 3 {
            return Err(EvaluatorError::new(format!(
                "TorchScript 输出 tuple 长度应为 2 或 3，实际 {n}"
            )));
        }
        let health = if n == 3 {
            Some(pop_tensor(&mut tensors, "health")?)
        } else {
            None
        };
        let value = pop_tensor(&mut tensors, "value")?;
        let policy_logits = pop_tensor(&mut tensors, "policy_logits")?;
        Ok((policy_logits, value, health))
    }

    /// 把 TorchScript 的 `(policy_logits, value, health)` 张量搬回 CPU 并组装 `EvaluatorOutput`。
    pub(crate) fn assemble_output(
        policy_logits: Tensor,
        value: Tensor,
        health_t: Option<Tensor>,
        dims: &BatchDims,
        action_space: usize,
    ) -> Result<EvaluatorOutput, EvaluatorError> {
        let model_action = policy_logits.size().get(1).copied().unwrap_or(0) as usize;
        if model_action == 0 {
            return Err(EvaluatorError::new("TorchScript policy 输出缺少动作维度"));
        }
        let mut logits_flat = vec![0.0f32; dims.batch * model_action];
        let logits_len = logits_flat.len();
        policy_logits
            .to_device(Device::Cpu)
            .copy_data(&mut logits_flat, logits_len);

        let mut values = vec![0.0f32; dims.batch];
        let values_len = values.len();
        value
            .to_device(Device::Cpu)
            .view([dims.batch as i64])
            .copy_data(&mut values, values_len);

        // 血量差异头：[B, K] 分桶 logits；旧模型为 None。
        let health = match health_t {
            Some(h) => {
                let k = h.size().get(1).copied().unwrap_or(0) as usize;
                if k == 0 {
                    return Err(EvaluatorError::new("TorchScript health 输出维度不足"));
                }
                let mut health_flat = vec![0.0f32; dims.batch * k];
                let n = health_flat.len();
                h.to_device(Device::Cpu).copy_data(&mut health_flat, n);
                Some(health_flat.chunks(k).map(|c| c.to_vec()).collect())
            }
            None => None,
        };

        output_from_flat_logits(
            &logits_flat,
            model_action,
            dims.batch,
            action_space,
            values,
            health,
        )
    }
}
