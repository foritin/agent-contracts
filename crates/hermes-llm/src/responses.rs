//! OpenAI Responses API Provider（`POST {base}/responses`）。
//!
//! 与 Chat Completions 是**两套协议**，不是同一协议的两种写法：
//!
//! | | Chat Completions | Responses |
//! | --- | --- | --- |
//! | 输入 | `messages[]` | `input[]`（异构 item 数组） |
//! | 系统提示 | `messages[0].role="system"` | 顶层 `instructions` |
//! | 输出上限 | `max_tokens` | `max_output_tokens` |
//! | 输出 | `choices[0].message` | `output[]`（message / reasoning / function_call 平级） |
//! | 工具定义 | `tools[].function.{...}` 嵌套 | `tools[].{type,name,parameters}` 扁平 |
//! | 工具回传 | `role:"tool"` + `tool_call_id` | `function_call_output` item + **`call_id`** |
//! | 流式 | 单一 chunk 靠字段判类型，`[DONE]` 收尾 | 类型化事件，`response.completed` 收尾，**无 `[DONE]`** |
//!
//! 采用 **无状态**策略（`store: false`，不使用 `previous_response_id`）：整段历史
//! 每轮重放。原因是我们的会话历史保存在本地 SQLite，服务端会话反而会引入
//! 双份真相；而且火山方舟这类实现只有服务端会话、没有加密回传，两条路无法统一。
//!
//! 两个高频 400 由 [`sanitize_input_items`] 在构造请求时静态挡掉：
//! - `Item 'rs_…' of type 'reasoning' was provided without its required following item.`
//! - `No tool call found for function call output with call_id …`

use hermes_core::{
    Capabilities, CompletionRequest, CompletionResponse, ContentBlock, HostedToolFormat,
    HostedToolSpec, LlmProvider, Message, Role, StopReason, StreamEvent, ToolSpec, Usage,
};
use hermes_error::{Error, Result};
use serde_json::{json, Value};

use crate::openai::{map_api_error, sanitize_transport_error};
use crate::url::openai_api_root;

/// 推理内容的处理策略。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReasoningMode {
    /// 不索取、也不回传 reasoning item。兼容性最好。
    ///
    /// 代价是多轮工具调用之间模型的思维链不连续；好处是永远不会触发
    /// "孤儿 reasoning" 400，且对不支持 `include` 的实现（如火山方舟）同样可用。
    #[default]
    Drop,
    /// 索取 `reasoning.encrypted_content` 并在下一轮原样回传。
    ///
    /// 仅 OpenAI 官方与 xAI 支持。加密块存放在 [`ContentBlock::Thinking::signature`] 里，
    /// 编解码见 [`encode_reasoning_signature`] / [`decode_reasoning_signature`]。
    EncryptedReplay,
}

pub struct ResponsesProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    /// 配置的默认模型（请求可覆盖）。
    #[allow(dead_code)]
    model: String,
    reasoning: ReasoningMode,
    max_context_tokens: u32,
}

impl ResponsesProvider {
    pub fn new(api_key: String, model: String, base_url: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            base_url,
            model,
            reasoning: ReasoningMode::default(),
            max_context_tokens: 200_000,
        }
    }

    /// 开启加密 reasoning 回传。只对 OpenAI / xAI 有意义。
    pub fn with_reasoning(mut self, mode: ReasoningMode) -> Self {
        self.reasoning = mode;
        self
    }

    /// 覆盖能力声明里的上下文窗口。
    pub fn with_max_context_tokens(mut self, tokens: u32) -> Self {
        self.max_context_tokens = tokens;
        self
    }

    fn responses_url(&self) -> String {
        format!("{}/responses", openai_api_root(&self.base_url))
    }

    fn build_body(&self, request: &CompletionRequest, stream: bool) -> Value {
        let mut input: Vec<Value> = Vec::new();
        for message in &request.messages {
            input.extend(message_to_items(message, self.reasoning));
        }
        let input = sanitize_input_items(input);

        let mut body = json!({
            "model": request.model,
            "input": input,
            "max_output_tokens": request.max_tokens,
            // 本地保存历史 + 每轮全量重放，不依赖服务端会话
            "store": false,
        });
        if let Some(system) = &request.system {
            // Responses 的 instructions 不会被上一轮继承，每次都要带
            body["instructions"] = json!(system);
        }
        if let Some(temp) = request.temperature {
            body["temperature"] = json!(temp);
        }
        if let Some(effort) = request.inference.reasoning_effort.as_deref() {
            body["reasoning"] = json!({ "effort": effort });
        }
        if let Some(verbosity) = request.inference.verbosity.as_deref() {
            body["text"] = json!({ "verbosity": verbosity });
        }
        if stream {
            body["stream"] = json!(true);
        }
        let mut tools = request
            .tools
            .iter()
            .map(tool_to_responses)
            .collect::<Vec<_>>();
        tools.extend(
            request
                .hosted_tools
                .iter()
                .filter_map(hosted_tool_to_responses),
        );
        if !tools.is_empty() {
            body["tools"] = json!(tools);
            body["tool_choice"] = json!("auto");
        }
        if self.reasoning == ReasoningMode::EncryptedReplay {
            body["include"] = json!(["reasoning.encrypted_content"]);
        }
        body
    }
}

#[async_trait::async_trait]
impl LlmProvider for ResponsesProvider {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let body = self.build_body(&request, false);
        let resp = self
            .client
            .post(self.responses_url())
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                Error::Provider(sanitize_transport_error(&e.to_string(), &self.api_key))
            })?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(map_api_error(status.as_u16(), &text, &self.api_key));
        }
        let value: Value = resp
            .json()
            .await
            .map_err(|e| Error::Provider(format!("invalid response json: {e}")))?;
        parse_responses_response(&value, self.reasoning)
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<futures::stream::BoxStream<'static, StreamEvent>> {
        let body = self.build_body(&request, true);
        let resp = self
            .client
            .post(self.responses_url())
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                Error::Provider(sanitize_transport_error(&e.to_string(), &self.api_key))
            })?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(map_api_error(status.as_u16(), &text, &self.api_key));
        }
        Ok(Box::pin(parse_responses_sse(resp.bytes_stream())))
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            supports_streaming: true,
            supports_tool_use: true,
            supports_vision: true,
            supports_prompt_caching: false,
            max_context_tokens: self.max_context_tokens,
        }
    }

    fn name(&self) -> &str {
        "openai_responses"
    }
}

// ── reasoning 签名编解码 ──────────────────────────────────────────
//
// 内部消息模型没有"reasoning item"这一类内容块，复用 `ContentBlock::Thinking`：
// `thinking` 放摘要文本（可能为空），`signature` 放回传所需的 id + 加密块。
// 用前缀 + 冒号分隔而不是 JSON，避免二次转义；id 形如 `rs_xxx`、加密块是
// base64 字符集，两者都不含冒号。

const REASONING_SIG_PREFIX: &str = "resp-reasoning:v1";

pub fn encode_reasoning_signature(id: &str, encrypted: &str) -> String {
    format!("{REASONING_SIG_PREFIX}:{id}:{encrypted}")
}

pub fn decode_reasoning_signature(signature: &str) -> Option<(String, String)> {
    let rest = signature
        .strip_prefix(REASONING_SIG_PREFIX)?
        .strip_prefix(':')?;
    let (id, encrypted) = rest.split_once(':')?;
    if encrypted.is_empty() {
        return None;
    }
    Some((id.to_string(), encrypted.to_string()))
}

// ── 请求转换 ──────────────────────────────────────────────────

/// 一条内部消息 → 若干个 Responses input item。**保持块顺序**，
/// reasoning 与其配对产物的相邻关系依赖于此。
pub fn message_to_items(msg: &Message, reasoning: ReasoningMode) -> Vec<Value> {
    let mut items = Vec::new();

    match msg.role {
        Role::User => {
            // 工具结果：每个 ToolResult 是一个独立 item，不是 message
            let mut has_tool_result = false;
            for block in &msg.content {
                if let ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    ..
                } = block
                {
                    has_tool_result = true;
                    items.push(json!({
                        "type": "function_call_output",
                        "call_id": tool_use_id,
                        "output": content,
                    }));
                }
            }
            if !has_tool_result {
                let content = msg
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text { text } if !text.is_empty() => {
                            Some(json!({ "type": "input_text", "text": text }))
                        }
                        ContentBlock::Image { source } => Some(json!({
                            "type": "input_image",
                            "image_url": format!("data:{};base64,{}", source.media_type, source.data),
                        })),
                        ContentBlock::File { source }
                            if source.media_type.starts_with("image/") && source.data.is_some() =>
                        {
                            Some(json!({
                                "type": "input_image",
                                "image_url": format!(
                                    "data:{};base64,{}",
                                    source.media_type,
                                    source.data.as_deref().unwrap_or_default()
                                ),
                            }))
                        }
                        ContentBlock::File { source }
                            if source.media_type == "application/pdf" && source.data.is_some() =>
                        {
                            Some(json!({
                                "type": "input_file",
                                "filename": source.name,
                                "file_data": format!(
                                    "data:{};base64,{}",
                                    source.media_type,
                                    source.data.as_deref().unwrap_or_default()
                                ),
                            }))
                        }
                        ContentBlock::File { source } if source.kind == "text" => Some(json!({
                            "type": "input_text",
                            "text": format!(
                                "\n\n--- Attached file: {} ({}) ---\n{}\n--- End attached file: {} ---",
                                source.name,
                                source.media_type,
                                source.text.as_deref().unwrap_or_default(),
                                source.name,
                            ),
                        })),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                items.push(json!({
                    "type": "message",
                    "role": "user",
                    "content": content,
                }));
            }
        }
        Role::Assistant => {
            let mut text_buffer = String::new();
            for block in &msg.content {
                match block {
                    ContentBlock::Thinking {
                        thinking,
                        signature,
                    } => {
                        if reasoning != ReasoningMode::EncryptedReplay {
                            continue;
                        }
                        let Some((id, encrypted)) =
                            signature.as_deref().and_then(decode_reasoning_signature)
                        else {
                            continue;
                        };
                        // 文本先落盘，保证 reasoning 排在它的配对产物之前
                        flush_assistant_text(&mut text_buffer, &mut items);
                        let _ = thinking;
                        items.push(json!({
                            "type": "reasoning",
                            "id": id,
                            "summary": [],
                            "encrypted_content": encrypted,
                        }));
                    }
                    ContentBlock::Text { text } => text_buffer.push_str(text),
                    ContentBlock::ToolUse { id, name, input } => {
                        flush_assistant_text(&mut text_buffer, &mut items);
                        items.push(json!({
                            "type": "function_call",
                            "call_id": id,
                            "name": name,
                            "arguments": input.to_string(),
                        }));
                    }
                    ContentBlock::Custom { type_name, data }
                        if is_hosted_web_item_type(type_name) =>
                    {
                        // Responses hosted tools are output items, not function calls. DeepSeek's
                        // stateless endpoint asks clients to replay the completed item as-is so it
                        // can restore the provider-side search context on later turns.
                        flush_assistant_text(&mut text_buffer, &mut items);
                        let mut item = data.as_object().cloned().unwrap_or_default();
                        item.insert("type".to_string(), Value::String(type_name.clone()));
                        items.push(Value::Object(item));
                    }
                    _ => {}
                }
            }
            flush_assistant_text(&mut text_buffer, &mut items);
        }
    }

    items
}

fn flush_assistant_text(buffer: &mut String, items: &mut Vec<Value>) {
    if buffer.is_empty() {
        return;
    }
    items.push(json!({
        "type": "message",
        "role": "assistant",
        "content": [{ "type": "output_text", "text": buffer }],
    }));
    buffer.clear();
}

/// 静态挡掉两个高频 400。
///
/// 1. `function_call_output` 的 `call_id` 必须在前面出现过同 id 的 `function_call`，
///    否则整个请求 400。历史被压缩、或上一轮工具调用被中止时会出现落单的结果。
/// 2. `reasoning` item 后面必须紧跟它的配对产物（`function_call` 或 assistant
///    `message`）。落在数组末尾、或后面只剩 user 消息的 reasoning 一律丢弃。
pub fn sanitize_input_items(items: Vec<Value>) -> Vec<Value> {
    let mut seen_calls: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut without_orphan_outputs = Vec::with_capacity(items.len());

    for item in items {
        match item_type(&item) {
            "function_call" => {
                if let Some(call_id) = item.get("call_id").and_then(|v| v.as_str()) {
                    seen_calls.insert(call_id.to_string());
                }
                without_orphan_outputs.push(item);
            }
            "function_call_output" => {
                let paired = item
                    .get("call_id")
                    .and_then(|v| v.as_str())
                    .is_some_and(|call_id| seen_calls.contains(call_id));
                if paired {
                    without_orphan_outputs.push(item);
                }
            }
            _ => without_orphan_outputs.push(item),
        }
    }

    let mut result = Vec::with_capacity(without_orphan_outputs.len());
    for (index, item) in without_orphan_outputs.iter().enumerate() {
        if item_type(item) == "reasoning" {
            let followed_by_product = without_orphan_outputs
                .get(index + 1)
                .is_some_and(is_reasoning_product);
            if !followed_by_product {
                continue;
            }
        }
        result.push(item.clone());
    }
    result
}

fn item_type(item: &Value) -> &str {
    item.get("type").and_then(|v| v.as_str()).unwrap_or("")
}

fn is_hosted_web_item_type(kind: &str) -> bool {
    matches!(
        kind,
        "web_search_call" | "web_fetch_call" | "web_extractor_call"
    )
}

fn hosted_web_tool_name(kind: &str) -> &'static str {
    match kind {
        "web_fetch_call" | "web_extractor_call" => "web_fetch",
        _ => "web_search",
    }
}

/// reasoning 的合法后继：函数调用，或助手消息。
fn is_reasoning_product(item: &Value) -> bool {
    match item_type(item) {
        kind if kind == "function_call" || is_hosted_web_item_type(kind) => true,
        "message" => item.get("role").and_then(|v| v.as_str()) == Some("assistant"),
        _ => false,
    }
}

/// 工具定义。Responses 是扁平结构，没有 `function` 这一层嵌套。
fn tool_to_responses(tool: &ToolSpec) -> Value {
    json!({
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.input_schema,
        // 我们的工具 schema 没有按 strict 模式写（缺 additionalProperties:false），
        // 显式关掉，避免服务端按严格模式校验后拒绝。
        "strict": false,
    })
}

fn hosted_tool_to_responses(tool: &HostedToolSpec) -> Option<Value> {
    match tool {
        // OpenAI, xAI, Azure, DeepSeek and Ark use the standard Responses schema. DashScope uses
        // the same search type but a dedicated extractor type for full-page reads.
        HostedToolSpec::WebSearch {
            format: HostedToolFormat::Standard | HostedToolFormat::DashScope,
            ..
        } => Some(json!({"type": "web_search"})),
        HostedToolSpec::WebFetch {
            format: HostedToolFormat::DashScope,
            ..
        } => Some(json!({"type": "web_extractor"})),
        HostedToolSpec::WebSearch {
            format: HostedToolFormat::OpenRouter,
            max_uses,
            allowed_domains,
            blocked_domains,
        } => Some(openrouter_server_tool(
            "openrouter:web_search",
            *max_uses,
            allowed_domains,
            blocked_domains,
        )),
        HostedToolSpec::WebFetch {
            format: HostedToolFormat::OpenRouter,
            max_uses,
            allowed_domains,
            blocked_domains,
        } => Some(openrouter_server_tool(
            "openrouter:web_fetch",
            *max_uses,
            allowed_domains,
            blocked_domains,
        )),
        // Standard Responses does not define a standalone web-fetch type. OpenAI/xAI/Azure
        // expose page opening through `web_search`; Anthropic maps WebFetch in its own adapter.
        HostedToolSpec::WebFetch {
            format: HostedToolFormat::Standard,
            ..
        } => None,
    }
}

fn openrouter_server_tool(
    tool_type: &str,
    max_uses: Option<u32>,
    allowed_domains: &[String],
    blocked_domains: &[String],
) -> Value {
    let mut parameters = serde_json::Map::new();
    if let Some(max_uses) = max_uses {
        parameters.insert("max_uses".into(), json!(max_uses));
    }
    if !allowed_domains.is_empty() {
        parameters.insert("allowed_domains".into(), json!(allowed_domains));
    }
    if !blocked_domains.is_empty() {
        let key = if tool_type == "openrouter:web_search" {
            "excluded_domains"
        } else {
            "blocked_domains"
        };
        parameters.insert(key.into(), json!(blocked_domains));
    }

    let mut tool = json!({"type": tool_type});
    if !parameters.is_empty() {
        tool["parameters"] = Value::Object(parameters);
    }
    tool
}

// ── 响应解析 ──────────────────────────────────────────────────

pub fn parse_responses_response(
    value: &Value,
    reasoning: ReasoningMode,
) -> Result<CompletionResponse> {
    let output = value
        .get("output")
        .and_then(|v| v.as_array())
        .ok_or_else(|| Error::Provider("response missing output".into()))?;

    let mut content = Vec::new();
    let mut saw_tool_call = false;

    for item in output {
        match item_type(item) {
            "reasoning" => {
                if reasoning != ReasoningMode::EncryptedReplay {
                    continue;
                }
                let Some(encrypted) = item.get("encrypted_content").and_then(|v| v.as_str()) else {
                    continue;
                };
                let id = item.get("id").and_then(|v| v.as_str()).unwrap_or_default();
                content.push(ContentBlock::Thinking {
                    thinking: reasoning_summary_text(item),
                    signature: Some(encode_reasoning_signature(id, encrypted)),
                });
            }
            "function_call" => {
                saw_tool_call = true;
                // 回传时必须用 call_id，不是 item 的 id
                let id = item
                    .get("call_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let name = item
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                content.push(ContentBlock::ToolUse {
                    id,
                    name,
                    input: parse_arguments(item.get("arguments").and_then(|v| v.as_str())),
                });
            }
            kind if is_hosted_web_item_type(kind) => {
                if let Some(block) = responses_hosted_item_to_custom(item) {
                    content.push(block);
                }
            }
            "message" => {
                let text = output_text_of(item);
                if !text.is_empty() {
                    content.push(ContentBlock::Text { text });
                }
            }
            _ => {}
        }
    }

    Ok(CompletionResponse {
        content,
        stop_reason: response_stop_reason(value, saw_tool_call),
        usage: parse_usage(value.get("usage")),
    })
}

/// `arguments` 永远是字符串化 JSON，不是对象。
fn parse_arguments(raw: Option<&str>) -> Value {
    raw.filter(|s| !s.trim().is_empty())
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or(Value::Null)
}

fn reasoning_summary_text(item: &Value) -> String {
    item.get("summary")
        .and_then(|v| v.as_array())
        .map(|parts| {
            parts
                .iter()
                .filter_map(|part| part.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("")
        })
        .unwrap_or_default()
}

fn output_text_of(item: &Value) -> String {
    item.get("content")
        .and_then(|v| v.as_array())
        .map(|parts| {
            parts
                .iter()
                .filter(|part| item_type(part) == "output_text")
                .filter_map(|part| part.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("")
        })
        .unwrap_or_default()
}

fn response_stop_reason(value: &Value, saw_tool_call: bool) -> StopReason {
    if saw_tool_call {
        return StopReason::ToolUse;
    }
    match value.get("status").and_then(|v| v.as_str()) {
        Some("completed") | None => StopReason::EndTurn,
        Some("incomplete") => {
            let reason = value
                .pointer("/incomplete_details/reason")
                .and_then(|v| v.as_str())
                .unwrap_or("incomplete");
            if reason == "max_output_tokens" {
                StopReason::MaxTokens
            } else {
                StopReason::Other(reason.to_string())
            }
        }
        Some(other) => StopReason::Other(other.to_string()),
    }
}

fn parse_usage(usage: Option<&Value>) -> Usage {
    let Some(usage) = usage else {
        return Usage::default();
    };
    Usage {
        input_tokens: usage
            .get("input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32,
        output_tokens: usage
            .get("output_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32,
        cache_read_tokens: usage
            .pointer("/input_tokens_details/cached_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32),
        cache_write_tokens: None,
    }
}

// ── SSE 流解析 ────────────────────────────────────────────────

/// 流内可变状态。
#[derive(Default)]
struct StreamState {
    /// `item_id` → `(call_id, name)`。delta 事件只带 item_id，回传要 call_id。
    calls: std::collections::HashMap<String, (String, String)>,
    /// 已发出 ToolUseComplete 的 call_id，防止 `.done` 与 `response.completed` 重复发。
    completed_calls: std::collections::HashSet<String>,
    /// 已发出 HostedToolUse / HostedToolResult 的 Responses web-search item id。
    announced_hosted_searches: std::collections::HashSet<String>,
    completed_hosted_searches: std::collections::HashSet<String>,
    /// 上一个 `sequence_number`，用于断线重连后的去重。
    last_sequence: Option<u64>,
    saw_tool_call: bool,
    stopped: bool,
}

fn parse_responses_sse(
    byte_stream: impl futures::Stream<Item = std::result::Result<bytes::Bytes, reqwest::Error>>,
) -> impl futures::Stream<Item = StreamEvent> {
    use futures::StreamExt;
    let mut buffer = String::new();
    let mut state = StreamState::default();

    byte_stream
        .map(move |chunk| {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(_) => return Vec::new(),
            };
            buffer.push_str(&String::from_utf8_lossy(&chunk));
            let mut events = Vec::new();
            while let Some(pos) = buffer.find('\n') {
                let line: String = buffer.drain(..pos + 1).collect();
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                // `event:` 行可以忽略：负载 JSON 自带 "type" 字段
                let Some(data) = line.strip_prefix("data:") else {
                    continue;
                };
                let data = data.trim();
                if data.is_empty() {
                    continue;
                }
                // Responses 规范里没有 [DONE]，但部分网关照抄了 Chat Completions
                if data == "[DONE]" {
                    if !state.stopped {
                        state.stopped = true;
                        events.push(StreamEvent::Stop {
                            reason: if state.saw_tool_call {
                                StopReason::ToolUse
                            } else {
                                StopReason::EndTurn
                            },
                        });
                    }
                    continue;
                }
                events.extend(parse_one_responses_event(data, &mut state));
            }
            events
        })
        .flat_map(futures::stream::iter)
}

fn parse_one_responses_event(data: &str, state: &mut StreamState) -> Vec<StreamEvent> {
    let Ok(value) = serde_json::from_str::<Value>(data) else {
        return Vec::new();
    };

    // 断线重连会重放已处理的事件；sequence_number 单调递增，据此去重
    if let Some(sequence) = value.get("sequence_number").and_then(|v| v.as_u64()) {
        if state.last_sequence.is_some_and(|last| sequence <= last) {
            return Vec::new();
        }
        state.last_sequence = Some(sequence);
    }

    let event_type = value.get("type").and_then(|v| v.as_str()).unwrap_or("");
    let mut events = Vec::new();

    match event_type {
        "response.output_text.delta" => {
            if let Some(delta) = value.get("delta").and_then(|v| v.as_str()) {
                if !delta.is_empty() {
                    events.push(StreamEvent::TextDelta {
                        text: delta.to_string(),
                    });
                }
            }
        }
        "response.output_item.added" => {
            if let Some(item) = value.get("item") {
                match item_type(item) {
                    "function_call" => {
                        state.saw_tool_call = true;
                        let item_id = item
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string();
                        let call_id = item
                            .get("call_id")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string();
                        let name = item
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string();
                        state.calls.insert(item_id, (call_id.clone(), name.clone()));
                        events.push(StreamEvent::ToolUseStart { id: call_id, name });
                    }
                    kind if is_hosted_web_item_type(kind) => {
                        announce_hosted_web_item(item, state, &mut events)
                    }
                    _ => {}
                }
            }
        }
        "response.function_call_arguments.delta" => {
            if let (Some(item_id), Some(delta)) = (
                value.get("item_id").and_then(|v| v.as_str()),
                value.get("delta").and_then(|v| v.as_str()),
            ) {
                if let Some((call_id, _)) = state.calls.get(item_id) {
                    events.push(StreamEvent::ToolUseDelta {
                        id: call_id.clone(),
                        input_json: delta.to_string(),
                    });
                }
            }
        }
        // `.done` 携带完整字符串，作为唯一可信来源；delta 只用于 UI 增量显示
        "response.output_item.done" => {
            if let Some(item) = value.get("item") {
                match item_type(item) {
                    "function_call" => {
                        let call_id = item
                            .get("call_id")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string();
                        if !call_id.is_empty() && state.completed_calls.insert(call_id.clone()) {
                            events.push(StreamEvent::ToolUseComplete {
                                id: call_id,
                                input: parse_arguments(
                                    item.get("arguments").and_then(|v| v.as_str()),
                                ),
                            });
                        }
                    }
                    kind if is_hosted_web_item_type(kind) => {
                        complete_hosted_web_item(item, state, &mut events)
                    }
                    _ => {}
                }
            }
        }
        "response.completed" | "response.incomplete" => {
            if let Some(response) = value.get("response") {
                // 少数实现不发 output_item.done，只在终帧给出完整 output
                events.extend(drain_final_output(response, state));
                events.push(StreamEvent::Usage(parse_usage(response.get("usage"))));
                if !state.stopped {
                    state.stopped = true;
                    events.push(StreamEvent::Stop {
                        reason: response_stop_reason(response, state.saw_tool_call),
                    });
                }
            } else if !state.stopped {
                state.stopped = true;
                events.push(StreamEvent::Stop {
                    reason: StopReason::EndTurn,
                });
            }
        }
        "response.failed" | "error" if !state.stopped => {
            state.stopped = true;
            let message = value
                .pointer("/response/error/message")
                .or_else(|| value.pointer("/error/message"))
                .or_else(|| value.get("message"))
                .and_then(|v| v.as_str())
                .unwrap_or("response failed");
            events.push(StreamEvent::Stop {
                reason: StopReason::Other(message.to_string()),
            });
        }
        // 未知的 response.* 事件一律忽略：OpenAI 持续新增事件类型，
        // 各家兼容实现也只做子集，报错会让流白白中断。
        _ => {}
    }

    events
}

/// 终帧兜底：补发没有通过 `output_item.done` 走完的工具调用。
fn drain_final_output(response: &Value, state: &mut StreamState) -> Vec<StreamEvent> {
    let Some(output) = response.get("output").and_then(|v| v.as_array()) else {
        return Vec::new();
    };
    let mut events = Vec::new();
    for item in output {
        match item_type(item) {
            "function_call" => {
                let Some(call_id) = item.get("call_id").and_then(|v| v.as_str()) else {
                    continue;
                };
                if call_id.is_empty() || !state.completed_calls.insert(call_id.to_string()) {
                    continue;
                }
                state.saw_tool_call = true;
                let name = item
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                events.push(StreamEvent::ToolUseStart {
                    id: call_id.to_string(),
                    name,
                });
                events.push(StreamEvent::ToolUseComplete {
                    id: call_id.to_string(),
                    input: parse_arguments(item.get("arguments").and_then(|v| v.as_str())),
                });
            }
            kind if is_hosted_web_item_type(kind) => {
                complete_hosted_web_item(item, state, &mut events)
            }
            _ => {}
        }
    }
    events
}

fn responses_hosted_item_to_custom(item: &Value) -> Option<ContentBlock> {
    let mut data = item.as_object()?.clone();
    let type_name = data.remove("type")?.as_str()?.to_string();
    is_hosted_web_item_type(&type_name).then_some(ContentBlock::Custom {
        type_name,
        data: Value::Object(data),
    })
}

fn hosted_web_item_id(item: &Value) -> Option<String> {
    item.get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .map(ToOwned::to_owned)
}

fn hosted_web_item_input(item: &Value) -> Value {
    if item_type(item) == "web_extractor_call" {
        return json!({
            "goal": item.get("goal").cloned().unwrap_or(Value::Null),
            "urls": item.get("urls").cloned().unwrap_or_else(|| json!([])),
        });
    }
    let action = item.get("action").unwrap_or(&Value::Null);
    if let Some(query) = action.get("query").and_then(Value::as_str) {
        return json!({"query": query});
    }
    if let Some(url) = action.get("url").and_then(Value::as_str) {
        return json!({
            "action": action.get("type").and_then(Value::as_str).unwrap_or("open_page"),
            "url": url,
        });
    }
    json!({
        "action": action.get("type").and_then(Value::as_str).unwrap_or("search")
    })
}

fn announce_hosted_web_item(item: &Value, state: &mut StreamState, events: &mut Vec<StreamEvent>) {
    let Some(id) = hosted_web_item_id(item) else {
        return;
    };
    if state.announced_hosted_searches.insert(id.clone()) {
        events.push(StreamEvent::HostedToolUse {
            id,
            name: hosted_web_tool_name(item_type(item)).to_string(),
            input: hosted_web_item_input(item),
            // The completed output item is retained once in HostedToolResult below.
            provider_content: None,
        });
    }
}

fn complete_hosted_web_item(item: &Value, state: &mut StreamState, events: &mut Vec<StreamEvent>) {
    let Some(id) = hosted_web_item_id(item) else {
        return;
    };
    announce_hosted_web_item(item, state, events);
    if !state.completed_hosted_searches.insert(id.clone()) {
        return;
    }
    let status = item
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("completed");
    let is_error = matches!(status, "failed" | "incomplete");
    events.push(StreamEvent::HostedToolResult {
        id,
        name: hosted_web_tool_name(item_type(item)).to_string(),
        output: if item_type(item) == "web_extractor_call" {
            json!({
                "status": status,
                "urls": item.get("urls").cloned().unwrap_or_else(|| json!([])),
            })
        } else {
            json!({"status": status})
        },
        is_error,
        provider_content: Some(item.clone()),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider(base_url: &str) -> ResponsesProvider {
        ResponsesProvider::new("sk-test".into(), "gpt-5.6-sol".into(), base_url.into())
    }

    fn request(messages: Vec<Message>) -> CompletionRequest {
        CompletionRequest {
            model: "gpt-5.6-sol".into(),
            system: Some("be nice".into()),
            messages,
            tools: vec![],
            hosted_tools: vec![],
            max_tokens: 1024,
            temperature: None,
            enable_caching: false,
            inference: Default::default(),
        }
    }

    #[test]
    fn responses_url_respects_custom_version_segment() {
        assert_eq!(
            provider("https://api.openai.com").responses_url(),
            "https://api.openai.com/v1/responses"
        );
        assert_eq!(
            provider("https://api.x.ai/v1").responses_url(),
            "https://api.x.ai/v1/responses"
        );
        assert_eq!(
            provider("https://ark.cn-beijing.volces.com/api/v3").responses_url(),
            "https://ark.cn-beijing.volces.com/api/v3/responses"
        );
    }

    #[test]
    fn system_goes_to_instructions_not_messages() {
        let body = provider("https://api.openai.com").build_body(&request(vec![]), false);
        assert_eq!(body["instructions"], "be nice");
        assert!(body.get("messages").is_none());
        assert_eq!(body["max_output_tokens"], 1024);
        assert!(body.get("max_tokens").is_none());
        assert_eq!(body["store"], false);
    }

    #[test]
    fn inference_options_use_responses_reasoning_and_text_objects() {
        let mut req = request(vec![]);
        req.inference = hermes_core::InferenceOptions {
            thinking: None,
            reasoning_effort: Some("high".into()),
            verbosity: Some("low".into()),
        };

        let body = provider("https://api.openai.com").build_body(&req, false);
        assert_eq!(body["reasoning"]["effort"], "high");
        assert_eq!(body["text"]["verbosity"], "low");
    }

    #[test]
    fn tools_are_flat_not_nested_under_function() {
        let mut req = request(vec![]);
        req.tools = vec![ToolSpec {
            name: "read_file".into(),
            description: "read a file".into(),
            input_schema: json!({"type": "object"}),
            source: hermes_core::ToolSource::Builtin,
            requires_confirmation: false,
        }];
        let body = provider("https://api.openai.com").build_body(&req, false);
        let tool = &body["tools"][0];
        assert_eq!(tool["type"], "function");
        assert_eq!(tool["name"], "read_file");
        assert!(tool.get("function").is_none());
        assert_eq!(body["tool_choice"], "auto");
    }

    #[test]
    fn hosted_web_search_uses_the_responses_server_tool_schema() {
        let mut req = request(vec![]);
        req.hosted_tools = vec![hermes_core::HostedToolSpec::web_search()];

        let body = provider("https://api.deepseek.com").build_body(&req, false);

        assert_eq!(body["tools"].as_array().map(Vec::len), Some(1));
        assert_eq!(body["tools"][0]["type"], "web_search");
        assert!(body["tools"][0].get("name").is_none());
        assert_eq!(body["tool_choice"], "auto");
    }

    #[test]
    fn provider_specific_responses_web_tools_use_their_documented_types() {
        let mut dashscope = request(vec![]);
        dashscope.hosted_tools = vec![
            HostedToolSpec::web_search_with_format(HostedToolFormat::DashScope),
            HostedToolSpec::web_fetch_with_format(HostedToolFormat::DashScope),
        ];
        let dashscope_body = provider("https://dashscope.aliyuncs.com/compatible-mode/v1")
            .build_body(&dashscope, false);
        assert_eq!(dashscope_body["tools"][0]["type"], "web_search");
        assert_eq!(dashscope_body["tools"][1]["type"], "web_extractor");

        let mut openrouter = request(vec![]);
        openrouter.hosted_tools = vec![
            HostedToolSpec::web_search_with_format(HostedToolFormat::OpenRouter),
            HostedToolSpec::web_fetch_with_format(HostedToolFormat::OpenRouter),
        ];
        let openrouter_body =
            provider("https://openrouter.ai/api/v1").build_body(&openrouter, false);
        assert_eq!(openrouter_body["tools"][0]["type"], "openrouter:web_search");
        assert_eq!(openrouter_body["tools"][0]["parameters"]["max_uses"], 5);
        assert_eq!(openrouter_body["tools"][1]["type"], "openrouter:web_fetch");
        assert_eq!(openrouter_body["tools"][1]["parameters"]["max_uses"], 5);
    }

    #[test]
    fn standard_responses_does_not_invent_a_web_fetch_type() {
        let mut req = request(vec![]);
        req.hosted_tools = vec![HostedToolSpec::web_fetch()];

        let body = provider("https://api.openai.com/v1").build_body(&req, false);

        assert!(body.get("tools").is_none());
    }

    #[test]
    fn include_only_requested_in_encrypted_replay_mode() {
        let plain = provider("https://api.openai.com").build_body(&request(vec![]), false);
        assert!(plain.get("include").is_none());

        let replay = provider("https://api.openai.com")
            .with_reasoning(ReasoningMode::EncryptedReplay)
            .build_body(&request(vec![]), false);
        assert_eq!(replay["include"][0], "reasoning.encrypted_content");
    }

    #[test]
    fn tool_use_becomes_function_call_with_call_id() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "call_abc".into(),
                name: "read_file".into(),
                input: json!({"path": "/a"}),
            }],
        };
        let items = message_to_items(&msg, ReasoningMode::Drop);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["type"], "function_call");
        assert_eq!(items[0]["call_id"], "call_abc");
        // arguments 必须是字符串化 JSON
        assert!(items[0]["arguments"].is_string());
    }

    #[test]
    fn assistant_web_search_call_replays_as_a_responses_input_item() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Custom {
                type_name: "web_search_call".into(),
                data: json!({
                    "id": "ws_1",
                    "status": "completed",
                    "action": {"type": "search", "query": "Milvus hybrid search"}
                }),
            }],
        };

        let items = message_to_items(&msg, ReasoningMode::Drop);

        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["type"], "web_search_call");
        assert_eq!(items[0]["id"], "ws_1");
        assert_eq!(items[0]["action"]["query"], "Milvus hybrid search");
    }

    #[test]
    fn tool_result_becomes_function_call_output_not_message() {
        let msg = Message {
            role: Role::User,
            content: vec![
                ContentBlock::ToolResult {
                    tool_use_id: "call_abc".into(),
                    content: "ok".into(),
                    is_error: false,
                },
                ContentBlock::ToolResult {
                    tool_use_id: "call_def".into(),
                    content: "also ok".into(),
                    is_error: false,
                },
            ],
        };
        let items = message_to_items(&msg, ReasoningMode::Drop);
        // 一条内部消息里的多个 ToolResult 必须全部展开
        assert_eq!(items.len(), 2);
        assert_eq!(items[0]["type"], "function_call_output");
        assert_eq!(items[1]["call_id"], "call_def");
    }

    #[test]
    fn user_image_is_emitted_as_input_image() {
        let message = Message {
            role: Role::User,
            content: vec![
                ContentBlock::Text {
                    text: "look".into(),
                },
                ContentBlock::Image {
                    source: hermes_core::ImageSource {
                        kind: "base64".into(),
                        media_type: "image/webp".into(),
                        data: "UklGRg==".into(),
                    },
                },
            ],
        };
        let items = message_to_items(&message, ReasoningMode::Drop);
        assert_eq!(items[0]["content"][0]["type"], "input_text");
        assert_eq!(items[0]["content"][1]["type"], "input_image");
        assert_eq!(
            items[0]["content"][1]["image_url"],
            "data:image/webp;base64,UklGRg=="
        );
    }

    #[test]
    fn orphan_function_call_output_is_dropped() {
        let items = vec![
            json!({"type": "function_call", "call_id": "call_1", "name": "a", "arguments": "{}"}),
            json!({"type": "function_call_output", "call_id": "call_1", "output": "ok"}),
            // 上一轮被中止，只剩结果没有调用
            json!({"type": "function_call_output", "call_id": "call_ghost", "output": "?"}),
        ];
        let sanitized = sanitize_input_items(items);
        assert_eq!(sanitized.len(), 2);
        assert!(sanitized.iter().all(|item| item["call_id"] != "call_ghost"));
    }

    #[test]
    fn orphan_reasoning_is_dropped_but_paired_reasoning_survives() {
        let items = vec![
            json!({"type": "reasoning", "id": "rs_1", "encrypted_content": "x"}),
            json!({"type": "function_call", "call_id": "call_1", "name": "a", "arguments": "{}"}),
            // 末尾落单的 reasoning：留着必 400
            json!({"type": "reasoning", "id": "rs_2", "encrypted_content": "y"}),
        ];
        let sanitized = sanitize_input_items(items);
        assert_eq!(sanitized.len(), 2);
        assert_eq!(sanitized[0]["id"], "rs_1");
        assert_eq!(sanitized[1]["type"], "function_call");
    }

    #[test]
    fn reasoning_followed_only_by_user_message_is_dropped() {
        let items = vec![
            json!({"type": "reasoning", "id": "rs_1", "encrypted_content": "x"}),
            json!({"type": "message", "role": "user", "content": []}),
        ];
        assert_eq!(sanitize_input_items(items).len(), 1);
    }

    #[test]
    fn reasoning_signature_roundtrip() {
        let sig = encode_reasoning_signature("rs_1", "gAAAAAB+base64==");
        assert_eq!(
            decode_reasoning_signature(&sig),
            Some(("rs_1".into(), "gAAAAAB+base64==".into()))
        );
        // Anthropic 的 thinking signature 不该被误读成 Responses 的
        assert_eq!(decode_reasoning_signature("ErUBCkYIBRgCKk"), None);
    }

    #[test]
    fn reasoning_block_replays_before_its_product() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Thinking {
                    thinking: "think".into(),
                    signature: Some(encode_reasoning_signature("rs_1", "enc")),
                },
                ContentBlock::ToolUse {
                    id: "call_1".into(),
                    name: "a".into(),
                    input: json!({}),
                },
            ],
        };
        let items = sanitize_input_items(message_to_items(&msg, ReasoningMode::EncryptedReplay));
        assert_eq!(items[0]["type"], "reasoning");
        assert_eq!(items[0]["encrypted_content"], "enc");
        assert_eq!(items[1]["type"], "function_call");
    }

    #[test]
    fn reasoning_block_omitted_in_drop_mode() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Thinking {
                thinking: "think".into(),
                signature: Some(encode_reasoning_signature("rs_1", "enc")),
            }],
        };
        assert!(message_to_items(&msg, ReasoningMode::Drop).is_empty());
    }

    #[test]
    fn parses_heterogeneous_output_array() {
        let value = json!({
            "status": "completed",
            "output": [
                {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "enc"},
                {"type": "function_call", "id": "fc_1", "call_id": "call_1",
                 "name": "read_file", "arguments": "{\"path\":\"/a\"}"},
                {"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": "done"}]}
            ],
            "usage": {"input_tokens": 10, "output_tokens": 3,
                      "input_tokens_details": {"cached_tokens": 4}}
        });
        let parsed = parse_responses_response(&value, ReasoningMode::EncryptedReplay).unwrap();
        assert_eq!(parsed.content.len(), 3);
        assert!(matches!(parsed.content[0], ContentBlock::Thinking { .. }));
        // 工具调用要取 call_id 而不是 item 的 id
        assert_eq!(parsed.content[1].tool_id(), Some("call_1"));
        assert_eq!(parsed.text(), "done");
        assert_eq!(parsed.stop_reason, StopReason::ToolUse);
        assert_eq!(parsed.usage.input_tokens, 10);
        assert_eq!(parsed.usage.cache_read_tokens, Some(4));
    }

    #[test]
    fn incomplete_due_to_length_maps_to_max_tokens() {
        let value = json!({
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": []
        });
        let parsed = parse_responses_response(&value, ReasoningMode::Drop).unwrap();
        assert_eq!(parsed.stop_reason, StopReason::MaxTokens);
    }

    #[test]
    fn missing_output_is_an_error_not_a_silent_empty_turn() {
        assert!(
            parse_responses_response(&json!({"status": "completed"}), ReasoningMode::Drop).is_err()
        );
    }

    // ── 流式 ──────────────────────────────────────────────────

    fn run_stream(frames: &[&str]) -> Vec<StreamEvent> {
        let mut state = StreamState::default();
        frames
            .iter()
            .flat_map(|frame| parse_one_responses_event(frame, &mut state))
            .collect()
    }

    #[test]
    fn typed_events_map_to_stream_events() {
        let events = run_stream(&[
            r#"{"type":"response.created","sequence_number":0}"#,
            r#"{"type":"response.output_text.delta","sequence_number":1,"delta":"He"}"#,
            r#"{"type":"response.output_text.delta","sequence_number":2,"delta":"llo"}"#,
            r#"{"type":"response.completed","sequence_number":3,
                "response":{"status":"completed","output":[],
                            "usage":{"input_tokens":5,"output_tokens":2}}}"#,
        ]);
        let text: String = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::TextDelta { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "Hello");
        assert!(matches!(events.last(), Some(StreamEvent::Stop { .. })));
    }

    #[test]
    fn tool_call_stream_uses_call_id_throughout() {
        let events = run_stream(&[
            r#"{"type":"response.output_item.added","sequence_number":1,
                "item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"read_file"}}"#,
            r#"{"type":"response.function_call_arguments.delta","sequence_number":2,
                "item_id":"fc_1","delta":"{\"path\":"}"#,
            r#"{"type":"response.function_call_arguments.delta","sequence_number":3,
                "item_id":"fc_1","delta":"\"/a\"}"}"#,
            r#"{"type":"response.output_item.done","sequence_number":4,
                "item":{"type":"function_call","id":"fc_1","call_id":"call_1",
                        "name":"read_file","arguments":"{\"path\":\"/a\"}"}}"#,
        ]);
        assert!(matches!(&events[0], StreamEvent::ToolUseStart { id, .. } if id == "call_1"));
        assert!(matches!(&events[1], StreamEvent::ToolUseDelta { id, .. } if id == "call_1"));
        match events.last().unwrap() {
            StreamEvent::ToolUseComplete { id, input } => {
                assert_eq!(id, "call_1");
                // 以 .done 的完整 arguments 为准，而不是 delta 拼接
                assert_eq!(input["path"], "/a");
            }
            other => panic!("expected ToolUseComplete, got {other:?}"),
        }
    }

    #[test]
    fn web_search_stream_emits_hosted_events_and_never_a_local_tool_call() {
        let events = run_stream(&[
            r#"{"type":"response.output_item.added","sequence_number":1,
                "item":{"type":"web_search_call","id":"ws_1","status":"in_progress",
                        "action":{"type":"search","query":"Milvus hybrid search"}}}"#,
            r#"{"type":"response.output_item.done","sequence_number":2,
                "item":{"type":"web_search_call","id":"ws_1","status":"completed",
                        "action":{"type":"search","query":"Milvus hybrid search"}}}"#,
            r#"{"type":"response.completed","sequence_number":3,
                "response":{"status":"completed",
                 "output":[{"type":"web_search_call","id":"ws_1","status":"completed",
                            "action":{"type":"search","query":"Milvus hybrid search"}}],
                 "usage":{"input_tokens":2,"output_tokens":1}}}"#,
        ]);

        assert!(matches!(
            &events[0],
            StreamEvent::HostedToolUse { id, name, input, .. }
                if id == "ws_1"
                    && name == "web_search"
                    && input["query"] == "Milvus hybrid search"
        ));
        assert!(matches!(
            &events[1],
            StreamEvent::HostedToolResult { id, name, output, is_error: false, .. }
                if id == "ws_1"
                    && name == "web_search"
                    && output["status"] == "completed"
        ));
        assert!(!events.iter().any(|event| matches!(
            event,
            StreamEvent::ToolUseStart { .. }
                | StreamEvent::ToolUseDelta { .. }
                | StreamEvent::ToolUseComplete { .. }
        )));
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, StreamEvent::HostedToolUse { .. }))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, StreamEvent::HostedToolResult { .. }))
                .count(),
            1
        );
        assert!(matches!(
            events.last(),
            Some(StreamEvent::Stop {
                reason: StopReason::EndTurn
            })
        ));
    }

    #[test]
    fn web_extractor_stream_is_reported_as_hosted_web_fetch() {
        let events = run_stream(&[
            r#"{"type":"response.output_item.added","sequence_number":1,
                "item":{"type":"web_extractor_call","id":"we_1","status":"in_progress",
                        "goal":"summarize","urls":["https://example.com"]}}"#,
            r#"{"type":"response.output_item.done","sequence_number":2,
                "item":{"type":"web_extractor_call","id":"we_1","status":"completed",
                        "goal":"summarize","urls":["https://example.com"],"output":"summary"}}"#,
        ]);

        assert!(matches!(
            &events[0],
            StreamEvent::HostedToolUse { id, name, input, .. }
                if id == "we_1" && name == "web_fetch"
                    && input["urls"][0] == "https://example.com"
        ));
        assert!(matches!(
            &events[1],
            StreamEvent::HostedToolResult { id, name, output, is_error: false, .. }
                if id == "we_1" && name == "web_fetch"
                    && output["urls"][0] == "https://example.com"
        ));
    }

    #[test]
    fn replayed_sequence_numbers_are_deduped() {
        let mut state = StreamState::default();
        let frame = r#"{"type":"response.output_text.delta","sequence_number":1,"delta":"x"}"#;
        assert_eq!(parse_one_responses_event(frame, &mut state).len(), 1);
        // 断线重连重放同一帧
        assert_eq!(parse_one_responses_event(frame, &mut state).len(), 0);
    }

    #[test]
    fn unknown_events_are_ignored_not_fatal() {
        let events = run_stream(&[
            r#"{"type":"response.some_future_event","sequence_number":1,"whatever":true}"#,
            r#"{"type":"response.output_text.delta","sequence_number":2,"delta":"ok"}"#,
        ]);
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn final_frame_backfills_tool_calls_never_announced() {
        // 部分兼容实现不发 output_item.added/done，只在终帧给出完整 output
        let events = run_stream(&[r#"{"type":"response.completed","sequence_number":1,
                "response":{"status":"completed",
                 "output":[{"type":"function_call","id":"fc_1","call_id":"call_1",
                            "name":"a","arguments":"{}"}],
                 "usage":{"input_tokens":1,"output_tokens":1}}}"#]);
        assert!(matches!(&events[0], StreamEvent::ToolUseStart { id, .. } if id == "call_1"));
        assert!(matches!(&events[1], StreamEvent::ToolUseComplete { .. }));
        assert!(matches!(
            events.last(),
            Some(StreamEvent::Stop {
                reason: StopReason::ToolUse
            })
        ));
    }

    #[test]
    fn tool_call_is_not_completed_twice() {
        let events = run_stream(&[
            r#"{"type":"response.output_item.done","sequence_number":1,
                "item":{"type":"function_call","id":"fc_1","call_id":"call_1",
                        "name":"a","arguments":"{}"}}"#,
            r#"{"type":"response.completed","sequence_number":2,
                "response":{"status":"completed",
                 "output":[{"type":"function_call","id":"fc_1","call_id":"call_1",
                            "name":"a","arguments":"{}"}],"usage":{}}}"#,
        ]);
        let completes = events
            .iter()
            .filter(|event| matches!(event, StreamEvent::ToolUseComplete { .. }))
            .count();
        assert_eq!(completes, 1);
    }

    #[test]
    fn failure_frame_stops_the_stream() {
        let events = run_stream(&[r#"{"type":"response.failed","sequence_number":1,
                "response":{"error":{"message":"upstream exploded"}}}"#]);
        assert!(matches!(
            &events[0],
            StreamEvent::Stop { reason: StopReason::Other(message) } if message == "upstream exploded"
        ));
    }

    #[test]
    fn error_frames_support_compatible_message_shapes() {
        let cases = [
            (
                r#"{"type":"error","error":{"message":"nested error"}}"#,
                "nested error",
            ),
            (
                r#"{"type":"error","message":"top-level error"}"#,
                "top-level error",
            ),
            (r#"{"type":"error"}"#, "response failed"),
        ];

        for (frame, expected) in cases {
            let events = run_stream(&[frame]);
            assert_eq!(events.len(), 1);
            assert!(matches!(
                &events[0],
                StreamEvent::Stop { reason: StopReason::Other(message) } if message == expected
            ));
        }
    }

    #[test]
    fn terminal_frames_emit_stop_only_once() {
        let completed_then_errors = run_stream(&[
            r#"{"type":"response.completed","sequence_number":1,
                "response":{"status":"completed","output":[],"usage":{}}}"#,
            r#"{"type":"response.failed","sequence_number":2,
                "response":{"error":{"message":"late failure"}}}"#,
            r#"{"type":"error","sequence_number":3,"message":"late error"}"#,
        ]);
        assert_eq!(
            completed_then_errors
                .iter()
                .filter(|event| matches!(event, StreamEvent::Stop { .. }))
                .count(),
            1
        );

        let failure_then_errors = run_stream(&[
            r#"{"type":"response.failed","sequence_number":1,
                "response":{"error":{"message":"first failure"}}}"#,
            r#"{"type":"error","sequence_number":2,"message":"late error"}"#,
            r#"{"type":"response.failed","sequence_number":3,
                "response":{"error":{"message":"later failure"}}}"#,
        ]);
        let stops: Vec<_> = failure_then_errors
            .iter()
            .filter(|event| matches!(event, StreamEvent::Stop { .. }))
            .collect();
        assert_eq!(stops.len(), 1);
        assert!(matches!(
            stops[0],
            StreamEvent::Stop { reason: StopReason::Other(message) } if message == "first failure"
        ));
    }

    #[test]
    fn capabilities_and_name() {
        let provider = provider("https://api.openai.com");
        assert_eq!(provider.name(), "openai_responses");
        assert!(provider.capabilities().supports_tool_use);
        assert_eq!(
            provider
                .with_max_context_tokens(400_000)
                .capabilities()
                .max_context_tokens,
            400_000
        );
    }

    #[test]
    fn pdf_attachment_becomes_an_input_file() {
        let message = Message {
            role: Role::User,
            content: vec![ContentBlock::File {
                source: hermes_core::FileSource {
                    kind: "base64".into(),
                    name: "spec.pdf".into(),
                    media_type: "application/pdf".into(),
                    text: None,
                    data: Some("JVBERi0=".into()),
                },
            }],
        };
        let items = message_to_items(&message, ReasoningMode::Drop);
        assert_eq!(items[0]["content"][0]["type"], "input_file");
        assert_eq!(items[0]["content"][0]["filename"], "spec.pdf");
    }
}
