//! `agent-store` -- JSONL 会话持久化。
//!
//! 参见 `02-session-management.html`。提供 `SessionStore`，支持创建、追加、
//! 加载、列出、删除、崩溃恢复（V-STORE-01）与归档（gzip）。原子写入保证
//! 中断后旧/新文件二选一完整可读（V-STORE-02）。

pub mod session_store;

pub use session_store::{
    SessionStore, DURABLE_USER_MESSAGE_CANCEL_EVENT, DURABLE_USER_MESSAGE_EVENT,
};
