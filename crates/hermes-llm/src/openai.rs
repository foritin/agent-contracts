//! OpenAI 兼容 Provider（Chat Completions API）。
//!
//! 参见 `01-llm-provider.html §4.2`。system 作为消息角色注入；不支持 prompt
//! caching。错误信息不含 api_key。

use hermes_core::{
    Capabilities, CompletionRequest, CompletionResponse, ContentBlock, LlmProvider, Message, Role,
    StopReason, StreamEvent, ToolSpec, Usage,
};
use hermes_error::{Error, Result};
use serde_json::{json, Value};

use crate::url::openai_api_root;

const MAX_ERROR_MESSAGE_CHARS: usize = 240;
const MAX_ERROR_METADATA_CHARS: usize = 64;

pub struct OpenAiProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    /// 配置的默认模型（请求可覆盖）。
    #[allow(dead_code)]
    model: String,
}

impl OpenAiProvider {
    pub fn new(api_key: String, model: String, base_url: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            base_url,
            model,
        }
    }

    fn completions_url(&self) -> String {
        // 版本段由 base_url 自带（/v1、/v3、/v4、/compatible-mode/v1 …），
        // 只有裸域名才补 /v1。规则与理由见 `crate::url`。
        format!("{}/chat/completions", openai_api_root(&self.base_url))
    }

    /// 官方 OpenAI 端点支持在流末额外返回 usage；兼容接口不假定支持该扩展。
    fn supports_stream_usage(&self) -> bool {
        reqwest::Url::parse(self.base_url.trim())
            .ok()
            .and_then(|url| {
                url.host_str()
                    .map(|host| host.eq_ignore_ascii_case("api.openai.com"))
            })
            .unwrap_or(false)
    }

    fn build_body(&self, request: &CompletionRequest, stream: bool) -> Value {
        let mut messages: Vec<Value> = Vec::new();

        // system 作为角色注入
        if let Some(system) = &request.system {
            messages.push(json!({ "role": "system", "content": system }));
        }

        for message in &request.messages {
            messages.extend(messages_to_openai(message));
        }

        let mut body = json!({
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
        });
        if let Some(temp) = request.temperature {
            body["temperature"] = json!(temp);
        }
        if stream {
            body["stream"] = json!(true);
            if self.supports_stream_usage() {
                body["stream_options"] = json!({ "include_usage": true });
            }
        }
        if !request.tools.is_empty() {
            body["tools"] = json!(request.tools.iter().map(tool_to_openai).collect::<Vec<_>>());
        }
        body
    }
}

#[async_trait::async_trait]
impl LlmProvider for OpenAiProvider {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let body = self.build_body(&request, false);
        let resp = self
            .client
            .post(self.completions_url())
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
        let v: Value = resp
            .json()
            .await
            .map_err(|e| Error::Provider(format!("invalid response json: {e}")))?;
        parse_openai_response(&v)
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<futures::stream::BoxStream<'static, StreamEvent>> {
        let body = self.build_body(&request, true);
        let resp = self
            .client
            .post(self.completions_url())
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
        Ok(Box::pin(parse_openai_sse(resp.bytes_stream())))
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            supports_streaming: true,
            supports_tool_use: true,
            supports_vision: true,
            supports_prompt_caching: false,
            max_context_tokens: 128_000,
        }
    }

    fn name(&self) -> &str {
        "openai"
    }
}

// ── 请求转换 ──────────────────────────────────────────────────

pub fn messages_to_openai(msg: &Message) -> Vec<Value> {
    // OpenAI 要求每个 tool_call 都有对应 role=tool 回复。一条内部 User 消息
    // 可以承载多个 ToolResult，因而必须展开为多条 API 消息，不能只取第一条。
    let tool_results: Vec<Value> = msg
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } => Some(json!({
                "role": "tool",
                "tool_call_id": tool_use_id,
                "content": content,
            })),
            _ => None,
        })
        .collect();
    if !tool_results.is_empty() {
        return tool_results;
    }

    // 若含 ToolUse 块 -> assistant + tool_calls
    let tool_uses: Vec<&ContentBlock> = msg.content.iter().filter(|b| b.is_tool_use()).collect();
    if !tool_uses.is_empty() {
        let tool_calls: Vec<Value> = tool_uses
            .iter()
            .map(|b| match b {
                ContentBlock::ToolUse { id, name, input } => json!({
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": input.to_string(),
                    }
                }),
                _ => json!({}),
            })
            .collect();
        return vec![json!({
            "role": "assistant",
            "content": null,
            "tool_calls": tool_calls,
        })];
    }

    // 普通文本消息
    let role = match msg.role {
        Role::User => "user",
        Role::Assistant => "assistant",
    };
    vec![json!({ "role": role, "content": msg.text_content() })]
}

/// 将单条内部消息转换为一个 OpenAI 消息。
///
/// 保留这个便捷入口以兼容只含文本、ToolUse 或单个 ToolResult 的调用方；构建完整
/// 请求时必须使用 [`messages_to_openai`]，以免丢失同一消息中的后续 ToolResult。
pub fn message_to_openai(msg: &Message) -> Value {
    messages_to_openai(msg)
        .into_iter()
        .next()
        .unwrap_or_else(|| json!({ "role": "user", "content": "" }))
}

fn tool_to_openai(t: &ToolSpec) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.input_schema,
        }
    })
}

// ── 响应解析 ──────────────────────────────────────────────────

fn parse_openai_response(v: &Value) -> Result<CompletionResponse> {
    let choice = v
        .get("choices")
        .and_then(|c| c.get(0))
        .ok_or_else(|| Error::Provider("response missing choices".into()))?;

    let mut content = Vec::new();
    if let Some(msg) = choice.get("message") {
        if let Some(text) = msg.get("content").and_then(|t| t.as_str()) {
            if !text.is_empty() {
                content.push(ContentBlock::Text {
                    text: text.to_string(),
                });
            }
        }
        if let Some(tool_calls) = msg.get("tool_calls").and_then(|t| t.as_array()) {
            for tc in tool_calls {
                let id = tc
                    .get("id")
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string();
                let name = tc
                    .pointer("/function/name")
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string();
                let args = tc
                    .pointer("/function/arguments")
                    .and_then(|t| t.as_str())
                    .and_then(|s| serde_json::from_str(s).ok())
                    .unwrap_or(Value::Null);
                content.push(ContentBlock::ToolUse {
                    id,
                    name,
                    input: args,
                });
            }
        }
    }

    let stop_reason = choice
        .get("finish_reason")
        .and_then(|t| t.as_str())
        .map(parse_finish_reason)
        .unwrap_or(StopReason::EndTurn);

    let usage = v
        .get("usage")
        .map(|u| Usage {
            input_tokens: u.get("prompt_tokens").and_then(|t| t.as_u64()).unwrap_or(0) as u32,
            output_tokens: u
                .get("completion_tokens")
                .and_then(|t| t.as_u64())
                .unwrap_or(0) as u32,
            cache_read_tokens: None,
            cache_write_tokens: None,
        })
        .unwrap_or_default();

    Ok(CompletionResponse {
        content,
        stop_reason,
        usage,
    })
}

fn parse_finish_reason(s: &str) -> StopReason {
    match s {
        "stop" => StopReason::EndTurn,
        "tool_calls" => StopReason::ToolUse,
        "length" => StopReason::MaxTokens,
        other => StopReason::Other(other.to_string()),
    }
}

#[derive(Default)]
struct OpenAiErrorDetails {
    message: Option<String>,
    error_type: Option<String>,
    code: Option<String>,
    param: Option<String>,
}

/// 将 OpenAI 风格的错误负载压缩为可展示文本，避免把原始 JSON 交给调用方。
///
/// Responses provider 复用同一套错误归一化与密钥脱敏，不另写一份。
pub(crate) fn map_api_error(status: u16, body: &str, api_key: &str) -> Error {
    let message = normalize_api_error(body, api_key);
    match status {
        401 | 403 => Error::AuthFailed("authentication failed (check api_key)".into()),
        429 => Error::RateLimited { retry_after: 0 },
        404 => Error::ModelNotFound(format!("model not found (status {status}): {message}")),
        _ => Error::ApiError { status, message },
    }
}

fn normalize_api_error(body: &str, api_key: &str) -> String {
    let parsed = serde_json::from_str::<Value>(body).ok();
    let details = parsed
        .as_ref()
        .map(|value| extract_openai_error_details(value, api_key))
        .unwrap_or_default();

    let message = details
        .message
        .filter(|message| !message.is_empty())
        .unwrap_or_else(|| {
            if parsed.is_some() {
                "request failed".into()
            } else {
                let fallback = sanitize_error_value(body, api_key, MAX_ERROR_MESSAGE_CHARS);
                if fallback.is_empty() {
                    "request failed".into()
                } else {
                    fallback
                }
            }
        });

    let mut metadata = Vec::new();
    if let Some(error_type) = details.error_type.filter(|value| !value.is_empty()) {
        metadata.push(format!("type={error_type}"));
    }
    if let Some(code) = details.code.filter(|value| !value.is_empty()) {
        metadata.push(format!("code={code}"));
    }
    if let Some(param) = details.param.filter(|value| !value.is_empty()) {
        metadata.push(format!("param={param}"));
    }

    if metadata.is_empty() {
        message
    } else {
        format!("{message} ({})", metadata.join(", "))
    }
}

fn extract_openai_error_details(value: &Value, api_key: &str) -> OpenAiErrorDetails {
    let error = value.get("error").unwrap_or(value);
    let field = |name| error.get(name).or_else(|| value.get(name));
    let sanitize_message = |value: &Value| {
        json_scalar_to_string(value)
            .map(|value| sanitize_error_value(&value, api_key, MAX_ERROR_MESSAGE_CHARS))
    };
    let sanitize_metadata = |value: &Value| {
        json_scalar_to_string(value)
            .map(|value| sanitize_error_value(&value, api_key, MAX_ERROR_METADATA_CHARS))
    };

    OpenAiErrorDetails {
        message: error
            .as_str()
            .map(|value| sanitize_error_value(value, api_key, MAX_ERROR_MESSAGE_CHARS))
            .or_else(|| field("message").and_then(sanitize_message)),
        error_type: field("type").and_then(sanitize_metadata),
        code: field("code").and_then(sanitize_metadata),
        param: field("param").and_then(sanitize_metadata),
    }
}

fn json_scalar_to_string(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        Value::Number(_) | Value::Bool(_) => Some(value.to_string()),
        Value::Null | Value::Array(_) | Value::Object(_) => None,
    }
}

pub(crate) fn sanitize_transport_error(message: &str, api_key: &str) -> String {
    sanitize_error_value(message, api_key, MAX_ERROR_MESSAGE_CHARS)
}

fn sanitize_error_value(value: &str, api_key: &str, max_chars: usize) -> String {
    let value = if api_key.is_empty() {
        value.to_string()
    } else {
        value.replace(api_key, "***")
    };
    truncate_chars(&redact_common_secrets(&value), max_chars)
}

fn redact_common_secrets(value: &str) -> String {
    const ASSIGNMENT_PREFIXES: [&str; 5] = [
        "api_key=",
        "api-key=",
        "apikey=",
        "access_token=",
        "authorization=",
    ];

    let mut remaining = value;
    let mut redacted = String::with_capacity(value.len());

    while !remaining.is_empty() {
        if starts_with_ascii_ignore_case(remaining, "bearer ") {
            let prefix_len = "bearer ".len();
            let token_len = secret_token_len(&remaining[prefix_len..]);
            if token_len > 0 {
                redacted.push_str(&remaining[..prefix_len]);
                redacted.push_str("***");
                remaining = &remaining[prefix_len + token_len..];
                continue;
            }
        }

        if starts_with_ascii_ignore_case(remaining, "sk-") {
            let token_len = secret_token_len(remaining);
            if token_len > "sk-".len() {
                redacted.push_str("***");
                remaining = &remaining[token_len..];
                continue;
            }
        }

        if let Some(prefix) = ASSIGNMENT_PREFIXES
            .iter()
            .find(|prefix| starts_with_ascii_ignore_case(remaining, prefix))
        {
            let token_len = secret_token_len(&remaining[prefix.len()..]);
            if token_len > 0 {
                redacted.push_str(&remaining[..prefix.len()]);
                redacted.push_str("***");
                remaining = &remaining[prefix.len() + token_len..];
                continue;
            }
        }

        let ch = remaining.chars().next().expect("remaining is not empty");
        redacted.push(ch);
        remaining = &remaining[ch.len_utf8()..];
    }

    redacted
}

fn starts_with_ascii_ignore_case(value: &str, prefix: &str) -> bool {
    value
        .get(..prefix.len())
        .is_some_and(|candidate| candidate.eq_ignore_ascii_case(prefix))
}

fn secret_token_len(value: &str) -> usize {
    value
        .char_indices()
        .take_while(|(_, ch)| {
            ch.is_ascii_alphanumeric()
                || matches!(ch, '-' | '_' | '.' | '~' | '+' | '/' | '=' | '%')
        })
        .map(|(index, ch)| index + ch.len_utf8())
        .last()
        .unwrap_or(0)
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

// ── SSE 流解析 ────────────────────────────────────────────────

fn parse_openai_sse(
    byte_stream: impl futures::Stream<Item = std::result::Result<bytes::Bytes, reqwest::Error>>,
) -> impl futures::Stream<Item = StreamEvent> {
    use futures::StreamExt;
    let mut buffer = String::new();
    let mut tool_args: std::collections::HashMap<u32, (String, String)> =
        std::collections::HashMap::new();
    let mut stopped = false;

    byte_stream
        .map(move |chunk| {
            let chunk = match chunk {
                Ok(c) => c,
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
                let Some(data) = line.strip_prefix("data: ") else {
                    continue;
                };
                if data.trim() == "[DONE]" {
                    // 少数兼容接口不会发送带 finish_reason 的最后一个 choice。若仍有
                    // 未完成工具调用，在 EOF 前将它们和 Stop 一并补齐。
                    if !stopped {
                        let reason = if tool_args.is_empty() {
                            StopReason::EndTurn
                        } else {
                            StopReason::ToolUse
                        };
                        events.extend(finish_openai_turn(&mut tool_args, reason));
                        stopped = true;
                    }
                    continue;
                }
                let parsed = parse_one_openai(data, &mut tool_args);
                if parsed
                    .iter()
                    .any(|event| matches!(event, StreamEvent::Stop { .. }))
                {
                    stopped = true;
                }
                events.extend(parsed);
            }
            events
        })
        .flat_map(futures::stream::iter)
}

fn parse_one_openai(
    data: &str,
    tool_args: &mut std::collections::HashMap<u32, (String, String)>,
) -> Vec<StreamEvent> {
    let Ok(value) = serde_json::from_str::<Value>(data) else {
        return Vec::new();
    };
    let mut events = Vec::new();

    if let Some(choice) = value.get("choices").and_then(|choices| choices.get(0)) {
        if let Some(delta) = choice.get("delta") {
            if let Some(content) = delta.get("content").and_then(|text| text.as_str()) {
                if !content.is_empty() {
                    events.push(StreamEvent::TextDelta {
                        text: content.to_string(),
                    });
                }
            }

            if let Some(tool_calls) = delta.get("tool_calls").and_then(|calls| calls.as_array()) {
                for tool_call in tool_calls {
                    let index = tool_call
                        .get("index")
                        .and_then(|value| value.as_u64())
                        .unwrap_or(0) as u32;
                    let entry = tool_args
                        .entry(index)
                        .or_insert_with(|| (String::new(), String::new()));
                    if let Some(id) = tool_call.get("id").and_then(|value| value.as_str()) {
                        entry.0 = id.to_string();
                    }
                    if let Some(name) = tool_call
                        .pointer("/function/name")
                        .and_then(|value| value.as_str())
                    {
                        if !name.is_empty() {
                            events.push(StreamEvent::ToolUseStart {
                                id: entry.0.clone(),
                                name: name.to_string(),
                            });
                        }
                    }
                    if let Some(arguments) = tool_call
                        .pointer("/function/arguments")
                        .and_then(|value| value.as_str())
                    {
                        entry.1.push_str(arguments);
                        events.push(StreamEvent::ToolUseDelta {
                            id: entry.0.clone(),
                            input_json: arguments.to_string(),
                        });
                    }
                }
            }
        }

        if let Some(reason) = choice.get("finish_reason").and_then(|value| value.as_str()) {
            events.extend(finish_openai_turn(tool_args, parse_finish_reason(reason)));
        }
    }

    if let Some(usage) = value.get("usage") {
        events.push(StreamEvent::Usage(Usage {
            input_tokens: usage
                .get("prompt_tokens")
                .and_then(|tokens| tokens.as_u64())
                .unwrap_or(0) as u32,
            output_tokens: usage
                .get("completion_tokens")
                .and_then(|tokens| tokens.as_u64())
                .unwrap_or(0) as u32,
            cache_read_tokens: None,
            cache_write_tokens: None,
        }));
    }

    events
}

/// 将一个结束帧归一化为“所有工具完成，再停止”。OpenAI Chat Completions 在
/// `finish_reason=tool_calls` 时可以在同一个 SSE 帧中同时表达全部工具调用；
/// 不能只 drain 第一个，否则后续请求会缺少成对的 `role=tool` 消息。
fn finish_openai_turn(
    tool_args: &mut std::collections::HashMap<u32, (String, String)>,
    reason: StopReason,
) -> Vec<StreamEvent> {
    let mut tool_calls: Vec<(u32, (String, String))> = tool_args.drain().collect();
    tool_calls.sort_by_key(|(index, _)| *index);

    let mut events: Vec<StreamEvent> = tool_calls
        .into_iter()
        .map(|(_, (id, arguments))| StreamEvent::ToolUseComplete {
            id,
            input: serde_json::from_str(&arguments).unwrap_or(Value::Null),
        })
        .collect();
    events.push(StreamEvent::Stop { reason });
    events
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completions_url_accepts_root_or_v1_base_url() {
        let from_root = OpenAiProvider::new(
            "sk-test".into(),
            "test-model".into(),
            "https://api.example.com".into(),
        );
        let from_v1 = OpenAiProvider::new(
            "sk-test".into(),
            "test-model".into(),
            "https://api.example.com/v1/".into(),
        );

        assert_eq!(
            from_root.completions_url(),
            "https://api.example.com/v1/chat/completions"
        );
        assert_eq!(
            from_v1.completions_url(),
            "https://api.example.com/v1/chat/completions"
        );
    }

    #[test]
    fn completions_url_keeps_vendor_version_segments() {
        // 智谱 Coding Plan 用 /v4，火山方舟套餐用 /api/coding/v3，
        // 百炼用 /compatible-mode/v1。旧的"结尾不是 /v1 就补 /v1"会把前两个
        // 拼成 .../v4/v1/chat/completions → 404。
        for base in [
            "https://open.bigmodel.cn/api/coding/paas/v4",
            "https://ark.cn-beijing.volces.com/api/coding/v3",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ] {
            let provider = OpenAiProvider::new("k".into(), "m".into(), base.into());
            assert_eq!(
                provider.completions_url(),
                format!("{base}/chat/completions")
            );
        }
    }

    #[test]
    fn system_role_injected() {
        let p = OpenAiProvider::new("k".into(), "gpt".into(), "https://api.openai.com".into());
        let req = CompletionRequest {
            model: "gpt".into(),
            system: Some("be nice".into()),
            messages: vec![Message::user_text("hi")],
            tools: vec![],
            max_tokens: 16,
            temperature: None,
            enable_caching: false,
        };
        let body = p.build_body(&req, false);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "be nice");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(body["max_tokens"], 16);
    }

    #[test]
    fn stream_usage_is_requested_only_for_official_openai_endpoint() {
        let req = CompletionRequest {
            model: "gpt".into(),
            system: None,
            messages: vec![Message::user_text("hi")],
            tools: vec![],
            max_tokens: 16,
            temperature: None,
            enable_caching: false,
        };
        let official = OpenAiProvider::new(
            "sk-test".into(),
            "gpt".into(),
            "https://api.openai.com/v1".into(),
        );
        let compatible = OpenAiProvider::new(
            "key".into(),
            "model".into(),
            "https://openrouter.ai/api/v1".into(),
        );

        let official_body = official.build_body(&req, true);
        let compatible_body = compatible.build_body(&req, true);

        assert_eq!(official_body["stream"], true);
        assert_eq!(official_body["stream_options"]["include_usage"], true);
        assert_eq!(compatible_body["stream"], true);
        assert!(compatible_body.get("stream_options").is_none());
    }

    #[test]
    fn tool_use_becomes_tool_calls() {
        let m = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "t1".into(),
                name: "read".into(),
                input: json!({"x": 1}),
            }],
        };
        let v = message_to_openai(&m);
        assert_eq!(v["role"], "assistant");
        assert_eq!(v["tool_calls"][0]["function"]["name"], "read");
    }

    #[test]
    fn tool_result_becomes_tool_role() {
        let m = Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "t1".into(),
                content: "ok".into(),
                is_error: false,
            }],
        };
        let v = message_to_openai(&m);
        assert_eq!(v["role"], "tool");
        assert_eq!(v["tool_call_id"], "t1");
    }

    #[test]
    fn multiple_tool_results_expand_to_multiple_tool_messages() {
        let message = Message {
            role: Role::User,
            content: vec![
                ContentBlock::ToolResult {
                    tool_use_id: "t1".into(),
                    content: "first".into(),
                    is_error: false,
                },
                ContentBlock::ToolResult {
                    tool_use_id: "t2".into(),
                    content: "second".into(),
                    is_error: false,
                },
            ],
        };

        let messages = messages_to_openai(&message);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "tool");
        assert_eq!(messages[0]["tool_call_id"], "t1");
        assert_eq!(messages[1]["role"], "tool");
        assert_eq!(messages[1]["tool_call_id"], "t2");
    }

    #[test]
    fn tool_calls_finish_with_all_completions_and_stop() {
        let mut tool_args = std::collections::HashMap::from([
            (0, ("call-a".to_string(), r#"{"a":1}"#.to_string())),
            (1, ("call-b".to_string(), r#"{"b":2}"#.to_string())),
        ]);

        let events = parse_one_openai(
            r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
            &mut tool_args,
        );

        assert!(matches!(
            events.as_slice(),
            [
                StreamEvent::ToolUseComplete { id: first, .. },
                StreamEvent::ToolUseComplete { id: second, .. },
                StreamEvent::Stop {
                    reason: StopReason::ToolUse
                }
            ] if first == "call-a" && second == "call-b"
        ));
        assert!(tool_args.is_empty());
    }

    #[test]
    fn parse_response_text() {
        let v = json!({
            "choices": [{
                "message": {"content": "hi"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1}
        });
        let r = parse_openai_response(&v).unwrap();
        assert_eq!(r.text(), "hi");
        assert_eq!(r.usage.input_tokens, 2);
    }

    #[test]
    fn api_error_sanitizes_key() {
        let e = map_api_error(401, "Bearer sk-SECRET", "sk-SECRET");
        let msg = e.to_string();
        assert!(!msg.contains("sk-SECRET"));
        assert!(matches!(e, Error::AuthFailed(_)));
    }

    #[test]
    fn api_error_extracts_service_message_from_json() {
        let e = map_api_error(
            400,
            r#"{"error":{"message":"Invalid max_tokens value","type":"invalid_request_error","code":"invalid_value","param":"max_tokens"}}"#,
            "sk-test",
        );
        let Error::ApiError { status, message } = e else {
            panic!("expected API error");
        };
        assert_eq!(status, 400);
        assert_eq!(
            message,
            "Invalid max_tokens value (type=invalid_request_error, code=invalid_value, param=max_tokens)"
        );
    }

    #[test]
    fn api_error_redacts_secrets_and_does_not_dump_unknown_json() {
        let e = map_api_error(
            400,
            r#"{"error":{"message":"invalid authorization: Bearer custom-secret sk-SECRET","type":"invalid_request_error","param":"api_key"}}"#,
            "custom-secret",
        );
        let msg = e.to_string();
        assert!(!msg.contains("custom-secret"));
        assert!(!msg.contains("sk-SECRET"));
        assert!(msg.contains("Bearer ***"));

        let e = map_api_error(
            500,
            r#"{"error":{"unexpected":"request payload should not be exposed"}}"#,
            "",
        );
        assert_eq!(e.to_string(), "API error: 500 - request failed");
    }
}
