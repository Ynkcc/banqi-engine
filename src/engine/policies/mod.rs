//! 引擎策略层

pub mod random;
pub mod reveal_first;

pub use random::RandomPolicy;
pub use reveal_first::{CaptureFirstPolicy, RevealFirstPolicy};

use banqi_core::core::env::DarkChessEnv;

pub trait Policy {
    fn choose_action(env: &DarkChessEnv) -> Option<usize>;
}
