//! `agent-compaction` -- 上下文压缩。
//!
//! 参见 `06-compaction.html`。提供滑动窗口、LLM 摘要、智能选择三种策略，
//! 以及按顺序自动选择的 `CompactionManager`。

pub mod manager;
pub mod strategies;

pub use manager::CompactionManager;
pub use strategies::{
    LlmSummaryCompaction, NoopProvider, SlidingWindowCompaction, SmartCompaction,
};
