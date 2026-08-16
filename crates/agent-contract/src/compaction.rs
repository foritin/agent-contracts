//! 上下文压缩策略抽象。
//!
//! 参见 `06-compaction.html §2`。trait 定义在 `agent-contract`；具体策略
//! （滑动窗口 / LLM 摘要 / 智能选择）与 `CompactionManager` 在 `agent-compaction`。

use crate::message::Message;
use crate::session::Session;
use crate::Result;

/// 压缩策略抽象。
#[async_trait::async_trait]
pub trait CompactionStrategy: Send + Sync {
    /// 判断是否需要压缩。
    fn should_compact(&self, session: &Session, max_tokens: u32) -> bool;

    /// 执行压缩，返回压缩后的消息列表。
    async fn compact(&self, session: &Session) -> Result<Vec<Message>>;

    /// 策略名称。
    fn name(&self) -> &str;
}
