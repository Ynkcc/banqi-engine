use banqi_core::core::env::DarkChessEnv;
use rand::seq::SliceRandom;

use super::Policy;

/// 当前玩家的合法动作按语义分桶：`(吃明子, 翻棋, 其余)`。
///
/// 「其余」包含空位移动、炮击以及吃暗子（机会动作）。语义标记由
/// `DarkChessEnv::generate_moves` 统一给出，避免各处重复推导规则。
fn bucket_actions(env: &DarkChessEnv) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let (mut captures, mut flips, mut others) = (Vec::new(), Vec::new(), Vec::new());
    for mv in env.generate_moves(env.get_current_player()) {
        if mv.is_capture {
            captures.push(mv.action);
        } else if mv.is_flip {
            flips.push(mv.action);
        } else {
            others.push(mv.action);
        }
    }
    (captures, flips, others)
}

/// 优先翻棋策略：
/// - 若存在“翻棋”类有效动作，随机选其一
/// - 否则在剩余所有有效动作（吃子 / 移动 / 炮击）中随机选择
pub struct RevealFirstPolicy;

impl Policy for RevealFirstPolicy {
    fn choose_action(env: &DarkChessEnv) -> Option<usize> {
        let (captures, flips, others) = bucket_actions(env);
        let mut rng = rand::thread_rng();

        // 优先：随机翻一个暗子
        if !flips.is_empty() {
            return flips.choose(&mut rng).copied();
        }

        // 退化：在其余所有合法动作中随机
        let mut fallback = captures;
        fallback.extend(others);
        fallback.choose(&mut rng).copied()
    }
}

/// 优先吃子策略：
/// - 若存在“吃明子”类有效动作，随机选其一
/// - 否则退化为优先翻棋（随机翻一个暗子）
/// - 再无可翻动作时，在剩余合法动作（移动 / 炮击 / 吃暗子）中随机选择
pub struct CaptureFirstPolicy;

impl Policy for CaptureFirstPolicy {
    fn choose_action(env: &DarkChessEnv) -> Option<usize> {
        let (captures, flips, others) = bucket_actions(env);
        let mut rng = rand::thread_rng();

        // 优先：随机吃一个明子
        if !captures.is_empty() {
            return captures.choose(&mut rng).copied();
        }

        // 其次：复用“优先翻棋”的逻辑
        if !flips.is_empty() {
            return flips.choose(&mut rng).copied();
        }

        // 最后：随机走其余合法动作
        others.choose(&mut rng).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use banqi_core::core::env::DarkChessEnv;

    /// 只要存在吃明子动作，`CaptureFirstPolicy` 必须选择吃子（否则退化为翻棋/静走）。
    #[test]
    fn capture_first_prefers_captures() {
        let mut checked = 0;
        for seed in 1..=64u64 {
            let mut env = DarkChessEnv::new();
            env.seed = Some(seed);
            env.reset();

            for _ in 0..200 {
                let moves = env.generate_moves(env.get_current_player());
                if moves.iter().any(|m| m.is_capture) {
                    let action = CaptureFirstPolicy::choose_action(&env)
                        .expect("存在吃子动作时不应返回 None");
                    let mv = moves
                        .iter()
                        .find(|m| m.action == action)
                        .expect("策略返回的动作应合法");
                    assert!(mv.is_capture, "seed={seed}: 有吃子动作却选择了非吃子动作");
                    checked += 1;
                }

                let legal = env.legal_action_indices();
                if legal.is_empty() {
                    break;
                }
                let pick = legal[(seed as usize * 7 + legal.len()) % legal.len()];
                match env.step(pick, None) {
                    Ok((_, terminated, _, _)) if !terminated => {}
                    _ => break,
                }
            }
        }
        assert!(checked > 0, "未构造出任何存在吃子的局面");
    }
}
