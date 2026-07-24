//! 错误上下文与结果扩展。
//!
//! 参见 `07-error-handling.html §3`。允许在错误传播链上附加
//! `module` / `operation` / `session_id` / `tool_name` 等上下文，
//! 便于诊断与日志归因，而不改变错误本身的分类。

use crate::{Error, Result};
use std::fmt;

/// 错误上下文快照（非错误本身的变体，仅用于诊断附加信息）。
#[derive(Debug)]
pub struct ErrorContext {
    pub module: String,
    pub operation: String,
    pub session_id: Option<String>,
    pub tool_name: Option<String>,
    pub source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.module, self.operation)?;
        if let Some(sid) = &self.session_id {
            write!(f, " (session={})", sid)?;
        }
        if let Some(tool) = &self.tool_name {
            write!(f, " (tool={})", tool)?;
        }
        Ok(())
    }
}

impl ErrorContext {
    pub fn new(module: impl Into<String>, operation: impl Into<String>) -> Self {
        Self {
            module: module.into(),
            operation: operation.into(),
            session_id: None,
            tool_name: None,
            source: None,
        }
    }

    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    pub fn with_tool(mut self, tool_name: impl Into<String>) -> Self {
        self.tool_name = Some(tool_name.into());
        self
    }
}

/// 为 `Result<T>` 附加上下文的扩展 trait。
///
/// 将底层错误包装为更具体的变体（会话 / 工具），保留原始消息用于诊断。
pub trait ResultExt<T> {
    /// 附加会话上下文：失败时转为 `Error::SessionNotFound`（含原消息）。
    fn with_session(self, session_id: &str) -> Result<T>;

    /// 附加工具上下文：失败时转为 `Error::ToolCallFailed`（含工具名与原消息）。
    fn with_tool(self, tool_name: &str) -> Result<T>;
}

impl<T> ResultExt<T> for Result<T> {
    fn with_session(self, session_id: &str) -> Result<T> {
        self.map_err(|e| Error::SessionNotFound(format!("{}: {}", session_id, e)))
    }

    fn with_tool(self, tool_name: &str) -> Result<T> {
        self.map_err(|e| Error::ToolCallFailed {
            tool: tool_name.to_string(),
            message: e.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_display_includes_fields() {
        let ctx = ErrorContext::new("store", "append")
            .with_session("s-1")
            .with_tool("read_file");
        let s = ctx.to_string();
        assert!(s.contains("[store] append"));
        assert!(s.contains("session=s-1"));
        assert!(s.contains("tool=read_file"));
    }

    #[test]
    fn with_tool_wraps_error() {
        let r: Result<()> = Err(Error::Io(std::io::Error::other("boom")));
        let wrapped = r.with_tool("read_file");
        match wrapped {
            Err(Error::ToolCallFailed { tool, message }) => {
                assert_eq!(tool, "read_file");
                assert!(message.contains("boom"));
            }
            _ => panic!("expected ToolCallFailed"),
        }
    }
}
