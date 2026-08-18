//! 会话数据模型。
//!
//! 参见 `02-session-management.html §2` 与 `12-api-contracts.html §2`。
//! `SessionMeta` 作为 JSONL 第一行；`SessionEvent` 为每一行事件；
//! `Session` 为内存视图，由 `agent-store` 在加载时重建。

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
    /// 仅供模型请求使用的压缩投影；完整历史继续由 `HistorySnapshot` 保存。
    ///
    /// `None` 会清除既有投影，使后续请求重新使用 canonical transcript。该事件是
    /// 向后兼容的增量格式：旧 JSONL 没有此行时等同于 `None`。
    ModelProjection {
        messages: Option<Vec<Message>>,
    },
    /// 模型请求信封快照（docs/harness-migration.md §1.3）。每次派发前追加；
    /// `reason` 区分 initial / resume / change。只存哈希不存全文（体积考虑：
    /// 全文每轮落盘会让 JSONL 随轮数平方膨胀）。
    ///
    /// 审计 / 重建自检专用：`agent-store` 加载时对它做 no-op，不进入
    /// messages / usage 投影。
    RequestHeader {
        /// system 文本指纹（serde_json 规范化字节流的 SHA-256 十六进制）。
        system_sha256: String,
        /// tools 列表指纹（同上；哈希覆盖完整 ToolSpec）。
        tools_sha256: String,
        /// 派发消息列表（已排除头部 memory 注入与 `excluded_tails` 登记 的
        /// 尾部注入）的指纹。
        messages_sha256: String,
        /// 派发原因："initial"（会话/运行首轮）| "resume"（恢复类重放）|
        /// "change"（其余常规轮）。
        reason: String,
        /// 尾部注入清单标签（本地时钟 / task_context / plan mode 等按轮注入、
        /// 不落盘的尾部 user 消息），重建自检时按登记排除，不算不一致。
        #[serde(default)]
        excluded_tails: Vec<String>,
        /// 本轮 tools 数组的名字清单（按派发顺序，含 hosted 工具别名后的名字）。
        /// 与 tools_sha256 互补：哈希负责字节级身份判等，名字清单负责 jq 级
        /// 人可读审计（docs/request-audit-and-anchoring.md A1）。
        #[serde(default)]
        tool_names: Vec<String>,
        /// 本轮 hosted 工具名（summary_only 轮为空）。
        #[serde(default)]
        hosted_tool_names: Vec<String>,
        /// 本轮实际派发的 max_tokens（钳制后）。0 表示旧版本写入的行。
        /// dsh issue #11 的教训：adapterDefaults 可能静默覆盖配置封顶，
        /// 该字段让「模型看到的输出预算」直接可审计。
        #[serde(default)]
        max_tokens: u32,
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
    /// 完整、可恢复的协议历史。压缩不得改写此字段。
    pub messages: Vec<Message>,
    /// 可选的模型可见投影。`None` 表示模型直接使用完整历史。
    ///
    /// 该字段只描述请求工作集，不参与 UI 时间线；摘要压缩可以替换投影，但必须
    /// 保留 `messages` 作为 canonical transcript，避免长会话证据永久丢失。
    pub model_projection: Option<Vec<Message>>,
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
            model_projection: None,
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
    fn new_session_starts_without_a_model_projection() {
        let session = Session::new(SessionMeta::new("m", "p"));
        assert!(session.model_projection.is_none());
    }

    #[test]
    fn model_projection_event_roundtrips_independently() {
        let ev = SessionEvent::ModelProjection {
            messages: Some(vec![Message::user_text("summary")]),
        };
        let encoded = serde_json::to_string(&ev).unwrap();
        let decoded: SessionEvent = serde_json::from_str(&encoded).unwrap();
        let SessionEvent::ModelProjection { messages } = decoded else {
            panic!("wrong event variant");
        };
        assert_eq!(messages.unwrap()[0].text_content(), "summary");
    }

    #[test]
    fn request_header_roundtrips_with_snake_case_tag() {
        // 1.3：RequestHeader 必须与其他变体保持一致的外部 tagged + snake_case
        // 表示（`{"request_header":{...}}`），否则旧 jq 抽取脚本与新写入互不认识。
        let ev = SessionEvent::RequestHeader {
            system_sha256: "aa".into(),
            tools_sha256: "bb".into(),
            messages_sha256: "cc".into(),
            reason: "initial".into(),
            excluded_tails: vec!["local_clock".into(), "plan_mode".into()],
            tool_names: vec!["read_file".into(), "edit".into()],
            hosted_tool_names: vec!["web_search".into()],
            max_tokens: 8_192,
        };
        let encoded = serde_json::to_string(&ev).unwrap();
        assert!(
            encoded.contains(r#""request_header""#),
            "encoded: {encoded}"
        );
        assert!(encoded.contains(r#""system_sha256":"aa""#));
        assert!(encoded.contains(r#""excluded_tails":["local_clock","plan_mode"]"#));
        assert!(encoded.contains(r#""tool_names":["read_file","edit"]"#));
        assert!(encoded.contains(r#""hosted_tool_names":["web_search"]"#));
        assert!(encoded.contains(r#""max_tokens":8192"#));
        let decoded: SessionEvent = serde_json::from_str(&encoded).unwrap();
        let SessionEvent::RequestHeader {
            system_sha256,
            tools_sha256,
            messages_sha256,
            reason,
            excluded_tails,
            tool_names,
            hosted_tool_names,
            max_tokens,
        } = decoded
        else {
            panic!("wrong event variant");
        };
        assert_eq!(system_sha256, "aa");
        assert_eq!(tools_sha256, "bb");
        assert_eq!(messages_sha256, "cc");
        assert_eq!(reason, "initial");
        assert_eq!(
            excluded_tails,
            vec!["local_clock".to_string(), "plan_mode".to_string()]
        );
        assert_eq!(
            tool_names,
            vec!["read_file".to_string(), "edit".to_string()]
        );
        assert_eq!(hosted_tool_names, vec!["web_search".to_string()]);
        assert_eq!(max_tokens, 8_192);
        // excluded_tails 及 A1 新字段缺省（旧读取器 / 手写行）时反序列化为
        // 默认值而非报错：tool_names/hosted_tool_names 为空清单，max_tokens 为 0。
        let without_tails = r#"{"request_header":{"system_sha256":"a","tools_sha256":"b","messages_sha256":"c","reason":"change"}}"#;
        let decoded: SessionEvent = serde_json::from_str(without_tails).unwrap();
        let SessionEvent::RequestHeader {
            excluded_tails,
            tool_names,
            hosted_tool_names,
            max_tokens,
            ..
        } = decoded
        else {
            panic!("wrong event variant");
        };
        assert!(excluded_tails.is_empty());
        assert!(tool_names.is_empty());
        assert!(hosted_tool_names.is_empty());
        assert_eq!(max_tokens, 0);
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
