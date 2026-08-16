//! 压缩管理器：按策略顺序自动选择或手动指定。
//!
//! 参见 `06-compaction.html §4`。

use agent_contract::{CompactionStrategy, Message, Session};
use agent_error::{Error, Result};

/// 压缩管理器。
pub struct CompactionManager {
    pub strategies: Vec<Box<dyn CompactionStrategy>>,
    pub max_tokens: u32,
}

impl CompactionManager {
    /// 默认装 sliding_window + smart 策略。
    pub fn new(max_tokens: u32) -> Self {
        let provider: std::sync::Arc<dyn agent_contract::LlmProvider> =
            std::sync::Arc::new(crate::strategies::NoopProvider);
        let smart = crate::strategies::SmartCompaction::new(provider);
        let sliding = crate::strategies::SlidingWindowCompaction::new(2, 10);
        Self {
            strategies: vec![Box::new(sliding), Box::new(smart)],
            max_tokens,
        }
    }

    /// 自定义策略集。
    pub fn with_strategies(max_tokens: u32, strategies: Vec<Box<dyn CompactionStrategy>>) -> Self {
        Self {
            strategies,
            max_tokens,
        }
    }

    /// 自动选择第一个 `should_compact=true` 的策略执行。
    pub async fn auto_compact(&self, session: &Session) -> Result<Vec<Message>> {
        for strategy in &self.strategies {
            if strategy.should_compact(session, self.max_tokens) {
                tracing::info!("Compacting with strategy: {}", strategy.name());
                return strategy.compact(session).await;
            }
        }
        Ok(session.messages.clone())
    }

    /// 手动指定策略名压缩。
    pub async fn compact_with(
        &self,
        session: &Session,
        strategy_name: &str,
    ) -> Result<Vec<Message>> {
        let strategy = self
            .strategies
            .iter()
            .find(|s| s.name() == strategy_name)
            .ok_or_else(|| Error::StrategyNotFound(strategy_name.to_string()))?;
        strategy.compact(session).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategies::SlidingWindowCompaction;
    use agent_contract::SessionMeta;

    fn make_session(n: usize) -> Session {
        let mut meta = SessionMeta::new("m", "p");
        meta.id = "t".into();
        let mut session = Session::new(meta);
        for i in 0..n {
            session.push_user(format!("message number {i} with some text"));
        }
        session
    }

    #[tokio::test]
    async fn compact_with_unknown_strategy_errors() {
        let m = CompactionManager::new(180_000);
        let session = make_session(5);
        let r = m.compact_with(&session, "nonexistent").await;
        assert!(matches!(r, Err(Error::StrategyNotFound(_))));
    }

    #[tokio::test]
    async fn auto_compact_no_trigger_returns_clone() {
        let m = CompactionManager::with_strategies(
            1_000_000,
            vec![Box::new(SlidingWindowCompaction::new(2, 5))],
        );
        let session = make_session(3);
        let result = m.auto_compact(&session).await.unwrap();
        assert_eq!(result.len(), 3);
    }
}
