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
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct InferenceOptions {
    /// Provider 原生思考模式：enabled / disabled / adaptive。None 表示服务默认。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    /// 推理强度：none / minimal / low / medium / high / xhigh / max。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    /// 输出详略：low / medium / high。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<String>,
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
    /// 由模型服务执行的托管工具。它们与客户端 `ToolSpec` 不同：Provider 会在
    /// 服务端完成调用，Agent 不能再次通过本地 ToolHost 执行。
    #[serde(default)]
    pub hosted_tools: Vec<HostedToolSpec>,
    pub max_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub enable_caching: bool,
    /// 会话级模型推理参数；全部为空时完全沿用 Provider 默认行为。
    #[serde(default, skip_serializing_if = "InferenceOptions::is_default")]
    pub inference: InferenceOptions,
}

/// 跨 Provider 的服务端托管工具声明。
///
/// 每个协议适配器负责把它映射为厂商自己的请求结构；不支持的适配器应忽略或
/// 显式拒绝，而不能把它伪装成需要客户端执行的函数调用。
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HostedToolFormat {
    /// 协议原生格式：Anthropic 的版本化工具，或 Responses 的 `web_search`。
    #[default]
    Standard,
    /// 阿里百炼 Responses：网页读取使用 `web_extractor`。
    DashScope,
    /// OpenRouter Server Tools：使用 `openrouter:*` 类型。
    OpenRouter,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HostedToolSpec {
    WebSearch {
        #[serde(default, skip_serializing_if = "is_standard_hosted_tool_format")]
        format: HostedToolFormat,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_uses: Option<u32>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        allowed_domains: Vec<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        blocked_domains: Vec<String>,
    },
    WebFetch {
        #[serde(default, skip_serializing_if = "is_standard_hosted_tool_format")]
        format: HostedToolFormat,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_uses: Option<u32>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        allowed_domains: Vec<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        blocked_domains: Vec<String>,
    },
}

fn is_standard_hosted_tool_format(format: &HostedToolFormat) -> bool {
    *format == HostedToolFormat::Standard
}

impl HostedToolSpec {
    /// 默认的按需联网搜索：模型自行决定是否调用，每个请求最多搜索五次。
    pub fn web_search() -> Self {
        Self::web_search_with_format(HostedToolFormat::Standard)
    }

    pub fn web_search_with_format(format: HostedToolFormat) -> Self {
        Self::WebSearch {
            format,
            max_uses: Some(5),
            allowed_domains: Vec::new(),
            blocked_domains: Vec::new(),
        }
    }

    /// 默认的按需网页读取：模型自行决定是否访问 URL，每个请求最多读取五次。
    pub fn web_fetch() -> Self {
        Self::web_fetch_with_format(HostedToolFormat::Standard)
    }

    pub fn web_fetch_with_format(format: HostedToolFormat) -> Self {
        Self::WebFetch {
            format,
            max_uses: Some(5),
            allowed_domains: Vec::new(),
            blocked_domains: Vec::new(),
        }
    }

    pub fn is_web_search(&self) -> bool {
        matches!(self, Self::WebSearch { .. })
    }

    pub fn is_web_fetch(&self) -> bool {
        matches!(self, Self::WebFetch { .. })
    }
}

impl InferenceOptions {
    pub fn is_default(&self) -> bool {
        self.thinking.is_none() && self.reasoning_effort.is_none() && self.verbosity.is_none()
    }
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
    TextDelta {
        text: String,
    },
    ToolUseStart {
        id: String,
        name: String,
    },
    ToolUseDelta {
        id: String,
        input_json: String,
    },
    ToolUseComplete {
        id: String,
        input: Value,
    },
    /// A tool executed by the model provider itself. Consumers may display it, but must not
    /// dispatch it to the local ToolHost.
    HostedToolUse {
        id: String,
        name: String,
        input: Value,
        /// Provider-native block retained only for protocol continuation. Product event logs
        /// should use the public fields above and must not expose this payload directly.
        #[serde(skip)]
        provider_content: Option<Value>,
    },
    /// Public, sanitized result metadata for a provider-hosted tool.
    HostedToolResult {
        id: String,
        name: String,
        output: Value,
        is_error: bool,
        /// Provider-native result retained for mixed client/server tool turns and pause/resume.
        /// It may contain encrypted search content, so UI consumers must use `output` instead.
        #[serde(skip)]
        provider_content: Option<Value>,
    },
    Stop {
        reason: StopReason,
    },
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
