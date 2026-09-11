//! NNUE 与 `banqi-core` 评估抽象（`NnueEvaluate` / `NnueAccumulator`）的桥接。
//!
//! - `NnueEvaluator` 实现全量叶评估 trait；
//! - `NnueDualAcc` 包装双视角累加器，实现搜索用的增量评估协议；
//! - `NnueEngineExt` 为 `ExpectimaxEngine` 提供 `from_nnue_file` 构造扩展
//!   （core 侧只认 trait，不感知具体权重格式）。

use std::sync::Arc;

use banqi_core::core::env::types::Player;
use banqi_core::core::env::DarkChessEnv;
use banqi_core::core::expectimax::ExpectimaxEngine;
use banqi_core::core::expectimax::nnue::{NnueAccumulator, NnueEvaluate};

use super::feature::{compute_step_diff, DualAccumulator};
use super::network::NnueEvaluator;

impl NnueEvaluate for NnueEvaluator {
    fn evaluate(&self, env: &DarkChessEnv) -> f32 {
        NnueEvaluator::evaluate(self, env)
    }

    fn validate_feature_dim(&self, expected: usize) -> Result<(), String> {
        NnueEvaluator::validate_feature_dim(self, expected)
    }

    fn init_accumulator(self: Arc<Self>, env: &DarkChessEnv) -> Box<dyn NnueAccumulator> {
        Box::new(NnueDualAcc {
            dual: DualAccumulator::init_from_env(env, &self),
            evaluator: self,
        })
    }
}

/// 持有权重引用的红黑双视角累加器（增量更新需要特征权重）。
#[derive(Clone)]
struct NnueDualAcc {
    dual: DualAccumulator,
    evaluator: Arc<NnueEvaluator>,
}

impl NnueAccumulator for NnueDualAcc {
    fn clone_box(&self) -> Box<dyn NnueAccumulator> {
        Box::new(self.clone())
    }

    fn apply_step(&mut self, before: &DarkChessEnv, after: &DarkChessEnv, action: usize) {
        let (diff_red, diff_black) = compute_step_diff(before, after, action);
        self.dual.apply_diffs(&diff_red, &diff_black, &self.evaluator);
    }

    fn evaluate(&self, player: Player) -> f32 {
        self.evaluator.forward_accumulator(self.dual.get(player))
    }
}

/// `ExpectimaxEngine` 的 NNUE 构造扩展（从 `.nnue` 权重量化文件加载）。
pub trait NnueEngineExt {
    fn from_nnue_file(path: &str) -> Result<Self, String>
    where
        Self: Sized;
}

impl NnueEngineExt for ExpectimaxEngine {
    fn from_nnue_file(path: &str) -> Result<Self, String> {
        let evaluator = NnueEvaluator::load_from_file(path)
            .map_err(|e| format!("加载 NNUE 权重文件失败 {}: {}", path, e))?;
        Ok(ExpectimaxEngine::with_nnue(Arc::new(evaluator)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use banqi_core::core::expectimax::{SearchConfig, search};

    /// 迁自 banqi-core expectimax 测试：NNUE 增量评估与全量重算一致。
    #[test]
    fn engine_incremental_nnue_matches_full_recompute() {
        let feature_dim = DarkChessEnv::default().config.nnue_feature_dim();
        let eval = NnueEvaluator::new_dummy(feature_dim);
        let mut cfg = SearchConfig::default();
        cfg.node_budget = 50_000;
        cfg.max_depth = 5;
        cfg.nnue_evaluator = Some(Arc::new(eval));

        for seed in [7u64, 33u64] {
            let mut env = DarkChessEnv::new();
            env.seed = Some(seed);
            env.reset();
            let res = search(&env, &cfg).expect("NNUE 增量搜索应返回动作");
            let mut masks = vec![0i32; env.config.action_space_size];
            env.action_masks_into(&mut masks);
            assert_eq!(masks[res.action], 1, "Seed {}: 引擎返回非法动作 {}", seed, res.action);
            assert!(res.value.abs() <= 1.0, "Seed {}: 评估值越界 {}", seed, res.value);
        }
    }

    /// NnueEngineExt 构造路径可用（dummy 权重写入临时文件后加载）。
    #[test]
    fn from_nnue_file_roundtrip() {
        let dim = DarkChessEnv::default().config.nnue_feature_dim();
        let evaluator = NnueEvaluator::new_dummy(dim);

        let mut buf = Vec::new();
        for v in evaluator
            .feature_weights
            .iter()
            .chain(&evaluator.feature_bias)
            .chain(&evaluator.fc1_weights)
            .chain(&evaluator.fc1_bias)
            .chain(&evaluator.fc2_weights)
        {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf.extend_from_slice(&evaluator.fc2_bias.to_le_bytes());

        let path = std::env::temp_dir().join(format!("banqi_engine_nnue_{}.nnue", std::process::id()));
        std::fs::write(&path, &buf).expect("写测试文件失败");
        let engine = ExpectimaxEngine::from_nnue_file(path.to_string_lossy().as_ref());
        std::fs::remove_file(&path).ok();

        let engine = engine.expect("from_nnue_file 应成功");
        let mut env = DarkChessEnv::new();
        env.seed = Some(11);
        env.reset();
        assert!(engine.search(&env).is_some(), "加载的引擎应能搜索");
    }
}
