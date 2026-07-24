//! 错误恢复策略映射表。
//!
//! 参见 `07-error-handling.html §4`。仅描述策略本身（可重试 / 需用户介入 /
//! 需重连 / 需截断恢复），具体重试 / 重连实现分散到各业务模块。
//!
//! 本模块提供「策略枚举」与「分类函数」，使上层 Agent 循环可以据此决定
//! 下一步动作，而不必在每个调用点重复 if-else。

use crate::Error;

/// 针对单个错误的推荐恢复策略。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryStrategy {
    /// 指数退避重试（如 RateLimited）。
    RetryWithBackoff,
    /// 提示用户更新凭据 / 重新授权。
    PromptUserAuth,
    /// 尝试截断恢复（如 JSONL 坏尾行）。
    TruncateAndRecover,
    /// 自动重连（如 MCP 断开）。
    Reconnect,
    /// 向 LLM 报告错误，让模型决定下一步。
    ReportToModel,
    /// 重试并通知用户（如 IPC 超时）。
    RetryAndNotify,
    /// 提示用户手动审批（如风险等级过高）。
    RequireManualApproval,
    /// 不可恢复，终止当前操作。
    Fatal,
}

/// 根据错误变体返回推荐恢复策略。
pub fn strategy_for(err: &Error) -> RecoveryStrategy {
    match err {
        Error::RateLimited { .. } => RecoveryStrategy::RetryWithBackoff,
        Error::AuthFailed(_) => RecoveryStrategy::PromptUserAuth,
        Error::SessionRecoveryFailed(_) => RecoveryStrategy::TruncateAndRecover,
        Error::McpNotConnected(_) | Error::McpServerNotFound(_) => RecoveryStrategy::Reconnect,
        Error::ToolCallFailed { .. } | Error::ToolNotFound(_) => RecoveryStrategy::ReportToModel,
        Error::IpcTimeout => RecoveryStrategy::RetryAndNotify,
        Error::RiskTooHigh { .. } | Error::PermissionDenied(_) => {
            RecoveryStrategy::RequireManualApproval
        }
        Error::Io(_) => RecoveryStrategy::RetryWithBackoff,
        _ => RecoveryStrategy::Fatal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_limited_retries() {
        assert_eq!(
            strategy_for(&Error::RateLimited { retry_after: 3 }),
            RecoveryStrategy::RetryWithBackoff
        );
    }

    #[test]
    fn mcp_disconnect_reconnects() {
        assert_eq!(
            strategy_for(&Error::McpNotConnected("fs".into())),
            RecoveryStrategy::Reconnect
        );
    }

    #[test]
    fn risk_requires_approval() {
        assert_eq!(
            strategy_for(&Error::RiskTooHigh {
                level: "R3".into(),
                threshold: "R2".into()
            }),
            RecoveryStrategy::RequireManualApproval
        );
    }
}
