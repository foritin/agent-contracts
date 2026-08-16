//! MCP 错误类型。
//!
//! 参见 `05-mcp-client.html §6`。`McpError` 为本 crate 内部错误，通过
//! `From<McpError> for agent_error::Error` 统一向上传播。

use std::path::PathBuf;

/// MCP 客户端错误。
#[derive(Debug, Clone, thiserror::Error)]
pub enum McpError {
    #[error("server not connected: {0}")]
    NotConnected(String),

    #[error("server not found: {0}")]
    ServerNotFound(String),

    #[error("tool not found: {0}")]
    ToolNotFound(String),

    #[error("invalid tool name format: {0}")]
    InvalidToolName(String),

    #[error("call failed: {0}")]
    CallFailed(String),

    #[error("transport error: {0}")]
    Transport(String),

    #[error("initialize failed: {0}")]
    InitializeFailed(String),

    #[error("config error: {0}")]
    Config(String),

    #[error("config file not found: {0}")]
    ConfigNotFound(PathBuf),

    #[error("timeout")]
    Timeout,

    #[error("parse error: {0}")]
    Parse(String),
}

impl From<serde_json::Error> for McpError {
    fn from(e: serde_json::Error) -> Self {
        McpError::Parse(e.to_string())
    }
}

impl From<tokio::io::Error> for McpError {
    fn from(e: tokio::io::Error) -> Self {
        McpError::Transport(e.to_string())
    }
}

/// 将 `McpError` 转为公共 `agent_error::Error`。
impl From<McpError> for agent_error::Error {
    fn from(e: McpError) -> Self {
        match e {
            McpError::NotConnected(s) => agent_error::Error::McpNotConnected(s),
            McpError::ServerNotFound(s) => agent_error::Error::McpServerNotFound(s),
            McpError::ToolNotFound(s) => agent_error::Error::ToolNotFound(s),
            McpError::InvalidToolName(s) => {
                agent_error::Error::ToolHost(format!("invalid tool name format: {s}"))
            }
            McpError::CallFailed(s) => agent_error::Error::ToolHost(s),
            McpError::Timeout => agent_error::Error::IpcTimeout,
            other => agent_error::Error::Mcp(other.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_to_public_error() {
        let e: agent_error::Error = McpError::NotConnected("fs".into()).into();
        assert!(matches!(e, agent_error::Error::McpNotConnected(_)));
    }

    #[test]
    fn invalid_tool_name_maps_to_tool_host() {
        let e: agent_error::Error = McpError::InvalidToolName("bad".into()).into();
        assert!(matches!(e, agent_error::Error::ToolHost(_)));
    }
}
