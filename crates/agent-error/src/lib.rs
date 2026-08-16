//! `agent-error` —— agent-core 共享错误类型。
//!
//! 基于 `thiserror` 定义统一 `Error` 枚举，支持跨模块传播与上下文携带。
//! 所有公共 crate 统一返回 `Result<T, Error>`，避免错误语义分叉。
//!
//! 参见文档 `07-error-handling.html`。

mod context;
mod recovery;

use std::path::PathBuf;

pub use context::{ErrorContext, ResultExt};
pub use recovery::{strategy_for, RecoveryStrategy};

/// 统一错误类型。
///
/// 每个变体对应一个可分类的故障域：会话 / Provider / 工具 / MCP / 配置 / 存储 /
/// 序列化 / IPC / 压缩。配合 `recovery` 模块中的策略表决定是否可恢复。
#[derive(Debug, thiserror::Error)]
pub enum Error {
    // ── 会话相关 ──────────────────────────────────────────────
    #[error("session not found: {0}")]
    SessionNotFound(String),

    #[error("session is locked: {0}")]
    SessionLocked(String),

    #[error("invalid session format: {0}")]
    InvalidSession(String),

    #[error("session recovery failed: {0}")]
    SessionRecoveryFailed(String),

    // ── Provider 相关 ────────────────────────────────────────
    #[error("provider error: {0}")]
    Provider(String),

    #[error("API error: {status} - {message}")]
    ApiError { status: u16, message: String },

    #[error("rate limited, retry after {retry_after}s")]
    RateLimited { retry_after: u64 },

    #[error("authentication failed: {0}")]
    AuthFailed(String),

    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("provider not found: {0}")]
    ProviderNotFound(String),

    // ── 工具相关 ──────────────────────────────────────────────
    #[error("tool error: {0}")]
    ToolHost(String),

    #[error("tool not found: {0}")]
    ToolNotFound(String),

    #[error("tool call failed (tool={tool}): {message}")]
    ToolCallFailed { tool: String, message: String },

    #[error("permission denied: {0}")]
    PermissionDenied(String),

    #[error("risk level too high: {level} exceeds threshold {threshold}")]
    RiskTooHigh { level: String, threshold: String },

    // ── MCP 相关 ──────────────────────────────────────────────
    #[error("MCP error: {0}")]
    Mcp(String),

    #[error("MCP server not connected: {0}")]
    McpNotConnected(String),

    #[error("MCP server not found: {0}")]
    McpServerNotFound(String),

    // ── 配置相关 ──────────────────────────────────────────────
    #[error("config error: {0}")]
    Config(String),

    #[error("config file not found: {0}")]
    ConfigNotFound(PathBuf),

    // ── 存储相关 ──────────────────────────────────────────────
    #[error("storage error: {0}")]
    Storage(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    // ── 序列化 ────────────────────────────────────────────────
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("YAML error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    #[error("TOML error: {0}")]
    Toml(#[from] toml::de::Error),

    // ── IPC ───────────────────────────────────────────────────
    #[error("IPC error: {0}")]
    Ipc(String),

    #[error("IPC timeout")]
    IpcTimeout,

    // ── 压缩 ──────────────────────────────────────────────────
    #[error("compaction failed: {0}")]
    Compaction(String),

    #[error("strategy not found: {0}")]
    StrategyNotFound(String),

    // ── 其他 ──────────────────────────────────────────────────
    #[error("internal error: {0}")]
    Internal(String),

    #[error("not implemented: {0}")]
    NotImplemented(String),

    #[error("{0}")]
    Other(String),
}

/// `serde_json` / `serde_yaml` / `toml` 的 `#[from]` 转换在枚举中直接声明，
/// 便于序列化失败时统一向上传播。这些 crate 作为本 crate 的常规依赖引入，
/// 因为几乎所有公共 crate 都会序列化数据。
///
/// 公共 Result 别名。
pub type Result<T> = std::result::Result<T, Error>;

/// 判断错误是否「可恢复」（用于驱动重试 / 重连 / 截断恢复等策略）。
///
/// 参见 `07-error-handling.html §4`。
pub fn is_recoverable(err: &Error) -> bool {
    matches!(
        err,
        Error::RateLimited { .. }
            | Error::SessionRecoveryFailed(_)
            | Error::McpNotConnected(_)
            | Error::IpcTimeout
            | Error::Io(_)
    )
}

/// 将任意错误字符串包装为 `Error::Other`，便于在边界处快速转换。
pub fn other(msg: impl Into<String>) -> Error {
    Error::Other(msg.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_roundtrip() {
        let e = Error::SessionNotFound("s-1".into());
        assert_eq!(e.to_string(), "session not found: s-1");
    }

    #[test]
    fn from_io_error() {
        let io = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");
        let e: Error = io.into();
        assert!(matches!(e, Error::Io(_)));
    }

    #[test]
    fn recoverable_classification() {
        assert!(is_recoverable(&Error::RateLimited { retry_after: 5 }));
        assert!(!is_recoverable(&Error::AuthFailed("bad key".into())));
    }
}
