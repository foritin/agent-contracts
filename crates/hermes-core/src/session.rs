//! 会话数据模型。
//!
//! 参见 `02-session-management.html §2` 与 `12-api-contracts.html §2`。
//! `SessionMeta` 作为 JSONL 第一行；`SessionEvent` 为每一行事件；
//! `Session` 为内存视图，由 `hermes-store` 在加载时重建。

use crate::message::Message;
use crate::usage::Usage;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

/// 会话元数据（JSONL 第一行）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMeta {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub model: String,
    pub provider: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

impl SessionMeta {
    /// 生成新元数据（随机 id + 当前时间）。
    pub fn new(model: impl Into<String>, provider: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            created_at: Utc::now(),
            model: model.into(),
            provider: provider.into(),
            title: None,
        }
    }
}

/// JSONL 中的每一行事件。
///
/// 注意 `ToolResult` 含 `is_error` 字段（见 `12-api-contracts §2.2`），
/// 用于区分正常结果与失败 / 取消结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionEvent {
    Meta(SessionMeta),
    Message(Message),
    Usage(Usage),
    ToolCall {
        name: String,
        input: Value,
    },
    ToolResult {
        call_id: String,
        output: Value,
        #[serde(default)]
        is_error: bool,
    },
    /// 当前运行工作集的无损快照。
    ///
    /// 工具调用同时会保留为独立审计事件；快照专用于在应用重启、运行时重建或
    /// 会话分叉后恢复 provider 所需的完整 Message/ToolUse/ToolResult 配对。
    HistorySnapshot {
        messages: Vec<Message>,
    },
    System {
        event: String,
        data: Value,
    },
}

/// 会话状态。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionStatus {
    Active,
    Paused,
    Completed,
    Archived,
    Error(String),
}

/// 内存中的会话视图，由持久化层重建。
#[derive(Debug, Clone)]
pub struct Session {
    pub meta: SessionMeta,
    pub messages: Vec<Message>,
    pub total_input_tokens: u32,
    pub total_output_tokens: u32,
    pub total_tool_calls: u32,
    pub status: SessionStatus,
}

impl Session {
    /// 由元数据构造空会话。
    pub fn new(meta: SessionMeta) -> Self {
        Self {
            meta,
            messages: Vec::new(),
            total_input_tokens: 0,
            total_output_tokens: 0,
            total_tool_calls: 0,
            status: SessionStatus::Active,
        }
    }

    /// 推入一条用户文本消息。
    pub fn push_user(&mut self, text: impl Into<String>) {
        self.messages.push(Message::user_text(text));
    }

    /// 推入一条 assistant 文本消息。
    pub fn push_assistant(&mut self, text: impl Into<String>) {
        self.messages.push(Message::assistant_text(text));
    }

    /// 累加 usage。
    pub fn add_usage(&mut self, usage: &Usage) {
        self.total_input_tokens += usage.input_tokens;
        self.total_output_tokens += usage.output_tokens;
    }

    /// 粗略估算当前消息总 token。
    pub fn estimate_tokens(&self) -> u32 {
        self.messages.iter().map(|m| m.estimate_tokens()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meta_new_generates_id() {
        let m = SessionMeta::new("claude-sonnet-4", "anthropic");
        assert!(!m.id.is_empty());
        assert_eq!(m.model, "claude-sonnet-4");
    }

    #[test]
    fn event_serde_roundtrip() {
        let ev = SessionEvent::ToolResult {
            call_id: "t1".into(),
            output: serde_json::json!({"ok": true}),
            is_error: false,
        };
        let json = serde_json::to_string(&ev).unwrap();
        assert!(json.contains(r#""tool_result""#));
        let back: SessionEvent = serde_json::from_str(&json).unwrap();
        match back {
            SessionEvent::ToolResult { is_error, .. } => assert!(!is_error),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn history_snapshot_roundtrips_tool_protocol_messages() {
        let ev = SessionEvent::HistorySnapshot {
            messages: vec![
                Message {
                    role: crate::Role::Assistant,
                    content: vec![crate::ContentBlock::ToolUse {
                        id: "t1".into(),
                        name: "read_file".into(),
                        input: serde_json::json!({"path": "src/lib.rs"}),
                    }],
                },
                Message {
                    role: crate::Role::User,
                    content: vec![crate::ContentBlock::ToolResult {
                        tool_use_id: "t1".into(),
                        content: "contents".into(),
                        is_error: false,
                    }],
                },
            ],
        };

        let encoded = serde_json::to_string(&ev).unwrap();
        let decoded: SessionEvent = serde_json::from_str(&encoded).unwrap();
        let SessionEvent::HistorySnapshot { messages } = decoded else {
            panic!("wrong event variant");
        };
        assert_eq!(messages.len(), 2);
        assert!(messages[0].content[0].is_tool_use());
        assert!(messages[1].content[0].is_tool_result());
    }

    #[test]
    fn event_unknown_variant_does_not_crash_when_skipping() {
        // 外部 tagged：未知变体无法反序列化为已知变体，但已知变体仍可解析。
        // 这里验证已知 Meta 行可被解析。
        let meta = SessionMeta::new("m", "p");
        let json = serde_json::to_string(&SessionEvent::Meta(meta)).unwrap();
        let _: SessionEvent = serde_json::from_str(&json).unwrap();
    }
}
