//! LLM Provider 抽象 trait 与请求 / 响应类型。
//!
//! 参见 `01-llm-provider.html`、`12-api-contracts.html §1`。
//! trait 定义在 `hermes-core`，具体实现（Anthropic / OpenAI / DeepSeek / Mock）
//! 在 `hermes-llm` crate 中。

use crate::message::{ContentBlock, Message};
use crate::tool_host::ToolSpec;
use crate::usage::Usage;
use crate::Result;
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// 统一 LLM Provider 抽象。
#[async_trait::async_trait]
pub trait LlmProvider: Send + Sync {
    /// 非流式完成。
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse>;

    /// 流式完成，返回事件流。
    async fn stream(&self, request: CompletionRequest) -> Result<BoxStream<'static, StreamEvent>>;

    /// 声明能力。
    fn capabilities(&self) -> Capabilities;

    /// Provider 名称。
    fn name(&self) -> &str;
}

/// 完成请求。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    /// 系统提示（独立顶层字段，与 Anthropic API 一致；非消息角色）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub tools: Vec<ToolSpec>,
    pub max_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub enable_caching: bool,
}

/// 完成响应。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub content: Vec<ContentBlock>,
    pub stop_reason: StopReason,
    pub usage: Usage,
}

impl CompletionResponse {
    /// 取首个文本块的文本（便于简单用例）。
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|b| b.as_text())
            .collect::<Vec<_>>()
            .join("")
    }
}

/// 流式事件。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    TextDelta { text: String },
    ToolUseStart { id: String, name: String },
    ToolUseDelta { id: String, input_json: String },
    ToolUseComplete { id: String, input: Value },
    Stop { reason: StopReason },
    Usage(Usage),
}

/// 停止原因。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    StopSequence,
    Other(String),
}

/// Provider 能力声明。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capabilities {
    pub supports_streaming: bool,
    pub supports_tool_use: bool,
    pub supports_vision: bool,
    pub supports_prompt_caching: bool,
    pub max_context_tokens: u32,
}

impl Capabilities {
    pub fn can_use_tools(&self) -> bool {
        self.supports_tool_use
    }

    pub fn cache_key(&self) -> &'static str {
        if self.supports_prompt_caching {
            "cached"
        } else {
            "uncached"
        }
    }
}
