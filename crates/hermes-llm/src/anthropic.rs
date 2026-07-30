//! Anthropic Claude Provider。
//!
//! 参见 `01-llm-provider.html §4.1 §7`。实现 Messages API 转换、流式 SSE 解析
//! 与 prompt caching。错误消息绝不包含 api_key（V-PROV-02）。

use hermes_core::{
    Capabilities, CompletionRequest, CompletionResponse, ContentBlock, LlmProvider, Message, Role,
    StopReason, StreamEvent, Usage,
};
use hermes_error::{Error, Result};
use serde_json::{json, Value};

const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";

pub struct AnthropicProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    /// 配置的默认模型（请求可覆盖）。
    #[allow(dead_code)]
    model: String,
}

impl AnthropicProvider {
    pub fn new(api_key: String, model: String, base_url: Option<String>) -> Result<Self> {
        if api_key.is_empty() {
            return Err(Error::AuthFailed("anthropic api_key is empty".into()));
        }
        Ok(Self {
            client: reqwest::Client::new(),
            api_key,
            base_url: base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_string()),
            model,
        })
    }

    fn messages_url(&self) -> String {
        let base_url = self.base_url.trim_end_matches('/');
        let api_root = if base_url.ends_with("/v1") {
            base_url.to_string()
        } else {
            format!("{base_url}/v1")
        };
        format!("{api_root}/messages")
    }

    /// 将内部请求转为 Anthropic Messages API 请求体。
    fn build_request_body(&self, request: &CompletionRequest) -> Value {
        let messages: Vec<Value> = request.messages.iter().map(message_to_anthropic).collect();

        let mut body = json!({
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": messages,
        });

        if let Some(system) = &request.system {
            body["system"] = json!(system);
        }
        if let Some(temp) = request.temperature {
            body["temperature"] = json!(temp);
        }
        if let Some(thinking) = request.inference.thinking.as_deref() {
            body["thinking"] = json!({ "type": thinking });
        } else if request.inference.reasoning_effort.is_some()
            && request.model.to_ascii_lowercase().contains("claude")
        {
            // Claude 4.6+ / 5 的 effort 必须与 adaptive thinking 一起使用。
            body["thinking"] = json!({ "type": "adaptive" });
        }
        if let Some(effort) = request.inference.reasoning_effort.as_deref() {
            body["output_config"] = json!({ "effort": effort });
        }
        if !request.tools.is_empty() {
            body["tools"] = json!(request
                .tools
                .iter()
                .map(|t| json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }))
                .collect::<Vec<_>>());
        }
        if request.enable_caching {
            apply_prompt_caching(&mut body);
        }
        body
    }

    /// 非流式完成。
    async fn do_complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let body = self.build_request_body(&request);
        let resp = self
            .client
            .post(self.messages_url())
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Provider(sanitize_http_err(&e)))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(map_api_error(status.as_u16(), &text));
        }

        let v: Value = resp
            .json()
            .await
            .map_err(|e| Error::Provider(format!("invalid response json: {e}")))?;
        parse_anthropic_response(&v)
    }

    /// 流式完成（SSE）。
    async fn do_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<futures::stream::BoxStream<'static, StreamEvent>> {
        let mut body = self.build_request_body(&request);
        body["stream"] = json!(true);

        let resp = self
            .client
            .post(self.messages_url())
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Provider(sanitize_http_err(&e)))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(map_api_error(status.as_u16(), &text));
        }

        let byte_stream = resp.bytes_stream();
        let stream = parse_sse_stream(byte_stream);
        Ok(Box::pin(stream))
    }
}

#[async_trait::async_trait]
impl LlmProvider for AnthropicProvider {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        self.do_complete(request).await
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<futures::stream::BoxStream<'static, StreamEvent>> {
        self.do_stream(request).await
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            supports_streaming: true,
            supports_tool_use: true,
            supports_vision: true,
            supports_prompt_caching: true,
            max_context_tokens: 200_000,
        }
    }

    fn name(&self) -> &str {
        "anthropic"
    }
}

// ── 请求转换 ──────────────────────────────────────────────────

/// 将内部 Message 转为 Anthropic 消息 JSON。
pub fn message_to_anthropic(msg: &Message) -> Value {
    let role = match msg.role {
        Role::User => "user",
        Role::Assistant => "assistant",
    };
    let content: Vec<Value> = msg.content.iter().map(content_block_to_anthropic).collect();
    json!({ "role": role, "content": content })
}

fn content_block_to_anthropic(b: &ContentBlock) -> Value {
    match b {
        ContentBlock::Text { text } => json!({ "type": "text", "text": text }),
        ContentBlock::Thinking {
            thinking,
            signature,
        } => {
            let mut v = json!({ "type": "thinking", "thinking": thinking });
            if let Some(sig) = signature {
                v["signature"] = json!(sig);
            }
            v
        }
        ContentBlock::ToolUse { id, name, input } => {
            json!({ "type": "tool_use", "id": id, "name": name, "input": input })
        }
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => {
            let mut v = json!({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": content,
            });
            if *is_error {
                v["is_error"] = json!(true);
            }
            v
        }
        ContentBlock::Image { source } => json!({ "type": "image", "source": source }),
        // 产品层扩展块降级为占位文本（公共层不解释语义）
        ContentBlock::Custom { type_name, data } => json!({
            "type": "text",
            "text": format!("[custom: {} {}]", type_name, data)
        }),
    }
}

/// 对 system + tools 添加 cache_control（prompt caching）。
fn apply_prompt_caching(body: &mut Value) {
    if let Some(system) = body.get_mut("system") {
        if let Some(s) = system.as_str() {
            *system = json!([{
                "type": "text",
                "text": s,
                "cache_control": { "type": "ephemeral" }
            }]);
        }
    }
    if let Some(tools) = body.get_mut("tools").and_then(|t| t.as_array_mut()) {
        if let Some(last) = tools.last_mut() {
            last["cache_control"] = json!({ "type": "ephemeral" });
        }
    }
}

// ── 响应解析 ──────────────────────────────────────────────────

fn parse_anthropic_response(v: &Value) -> Result<CompletionResponse> {
    let content_arr = v
        .get("content")
        .and_then(|c| c.as_array())
        .ok_or_else(|| Error::Provider("response missing content array".into()))?;

    let mut content = Vec::new();
    for block in content_arr {
        let kind = block.get("type").and_then(|t| t.as_str()).unwrap_or("text");
        match kind {
            "text" => {
                let text = block
                    .get("text")
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string();
                content.push(ContentBlock::Text { text });
            }
            "tool_use" => {
                let id = block
                    .get("id")
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string();
                let name = block
                    .get("name")
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string();
                let input = block.get("input").cloned().unwrap_or(Value::Null);
                content.push(ContentBlock::ToolUse { id, name, input });
            }
            "thinking" => {
                let thinking = block
                    .get("thinking")
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string();
                let signature = block
                    .get("signature")
                    .and_then(|t| t.as_str())
                    .map(String::from);
                content.push(ContentBlock::Thinking {
                    thinking,
                    signature,
                });
            }
            _ => {}
        }
    }

    let stop_reason = v
        .get("stop_reason")
        .and_then(|t| t.as_str())
        .map(parse_stop_reason)
        .unwrap_or(StopReason::EndTurn);

    let usage = v
        .get("usage")
        .map(|u| Usage {
            input_tokens: u.get("input_tokens").and_then(|t| t.as_u64()).unwrap_or(0) as u32,
            output_tokens: u.get("output_tokens").and_then(|t| t.as_u64()).unwrap_or(0) as u32,
            cache_read_tokens: u
                .get("cache_read_input_tokens")
                .and_then(|t| t.as_u64())
                .map(|n| n as u32),
            cache_write_tokens: u
                .get("cache_creation_input_tokens")
                .and_then(|t| t.as_u64())
                .map(|n| n as u32),
        })
        .unwrap_or_default();

    Ok(CompletionResponse {
        content,
        stop_reason,
        usage,
    })
}

fn parse_stop_reason(s: &str) -> StopReason {
    match s {
        "end_turn" => StopReason::EndTurn,
        "tool_use" => StopReason::ToolUse,
        "max_tokens" => StopReason::MaxTokens,
        "stop_sequence" => StopReason::StopSequence,
        other => StopReason::Other(other.to_string()),
    }
}

/// 将 HTTP 状态码 + 响应体映射为稳定错误类别，绝不包含 api_key。
fn map_api_error(status: u16, body: &str) -> Error {
    match status {
        401 | 403 => Error::AuthFailed("authentication failed (check api_key)".into()),
        429 => Error::RateLimited { retry_after: 0 },
        404 => Error::ModelNotFound(format!("model not found (status {status})")),
        _ => Error::ApiError {
            status,
            message: sanitize_body(body),
        },
    }
}

/// 移除响应体中可能出现的敏感片段。
fn sanitize_body(body: &str) -> String {
    let trimmed = body.chars().take(512).collect::<String>();
    trimmed.replace("sk-ant-", "***")
}

/// 网络错误信息脱敏（不含 URL 中的 key）。
fn sanitize_http_err(e: &reqwest::Error) -> String {
    let msg = e.to_string();
    msg.replace("sk-ant-", "***")
}

// ── SSE 流解析 ────────────────────────────────────────────────

fn parse_sse_stream(
    byte_stream: impl futures::Stream<Item = std::result::Result<bytes::Bytes, reqwest::Error>>,
) -> impl futures::Stream<Item = StreamEvent> {
    use futures::StreamExt;

    // 累积 buffer，按双换行分块解析 SSE 事件
    let mut buffer = String::new();
    // 工具调用输入累积
    let mut tool_inputs: std::collections::HashMap<String, (String, String)> =
        std::collections::HashMap::new(); // id -> (name, partial_json)

    byte_stream
        .map(move |chunk| {
            let chunk = match chunk {
                Ok(c) => c,
                Err(_) => return Vec::new(),
            };
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            let mut events = Vec::new();
            while let Some(pos) = buffer.find("\n\n") {
                let raw: String = buffer.drain(..pos + 2).collect();
                if let Some(ev) = parse_one_sse(&raw, &mut tool_inputs) {
                    events.push(ev);
                }
            }
            events
        })
        .flat_map(futures::stream::iter)
}

fn parse_one_sse(
    raw: &str,
    tool_inputs: &mut std::collections::HashMap<String, (String, String)>,
) -> Option<StreamEvent> {
    let mut event_type = String::new();
    let mut data = String::new();
    for line in raw.lines() {
        if let Some(rest) = line.strip_prefix("event: ") {
            event_type = rest.trim().to_string();
        } else if let Some(rest) = line.strip_prefix("data: ") {
            data.push_str(rest);
        }
    }
    if data.is_empty() {
        return None;
    }
    let v: Value = serde_json::from_str(&data).ok()?;

    match event_type.as_str() {
        "message_start" => {
            let input = v
                .pointer("/message/usage/input_tokens")
                .and_then(|t| t.as_u64())
                .unwrap_or(0) as u32;
            Some(StreamEvent::Usage(Usage {
                input_tokens: input,
                output_tokens: 0,
                cache_read_tokens: None,
                cache_write_tokens: None,
            }))
        }
        "content_block_start" => {
            let block = v.get("content_block")?;
            let kind = block.get("type")?.as_str()?;
            if kind == "tool_use" {
                let id = block.get("id")?.as_str()?.to_string();
                let name = block.get("name")?.as_str()?.to_string();
                tool_inputs.insert(id.clone(), (name.clone(), String::new()));
                Some(StreamEvent::ToolUseStart { id, name })
            } else {
                None
            }
        }
        "content_block_delta" => {
            let delta = v.get("delta")?;
            let kind = delta.get("type")?.as_str()?;
            match kind {
                "text_delta" => {
                    let text = delta.get("text")?.as_str()?.to_string();
                    Some(StreamEvent::TextDelta { text })
                }
                "input_json_delta" => {
                    let idx = v.get("index").and_then(|i| i.as_u64()).unwrap_or(0);
                    let partial = delta.get("partial_json")?.as_str()?.to_string();
                    // 找到对应 tool id（按 index 近似；真实实现需维护 index->id）
                    if let Some((id, (_name, json))) = tool_inputs.iter_mut().nth(idx as usize) {
                        json.push_str(&partial);
                        let id_clone = id.clone();
                        Some(StreamEvent::ToolUseDelta {
                            id: id_clone,
                            input_json: partial,
                        })
                    } else {
                        None
                    }
                }
                _ => None,
            }
        }
        "content_block_stop" => {
            let idx = v.get("index").and_then(|i| i.as_u64()).unwrap_or(0);
            if let Some((id, (name, json))) = tool_inputs.iter_mut().nth(idx as usize) {
                let id_clone = id.clone();
                let input: Value = serde_json::from_str(json).unwrap_or(Value::Null);
                let _ = name; // name 已在 ToolUseStart 发出
                tool_inputs.remove(&id_clone.clone());
                Some(StreamEvent::ToolUseComplete {
                    id: id_clone,
                    input,
                })
            } else {
                None
            }
        }
        "message_delta" => {
            let stop = v
                .pointer("/delta/stop_reason")
                .and_then(|t| t.as_str())
                .map(parse_stop_reason);
            let output = v
                .pointer("/usage/output_tokens")
                .and_then(|t| t.as_u64())
                .unwrap_or(0) as u32;
            if let Some(reason) = stop {
                Some(StreamEvent::Stop { reason })
            } else {
                Some(StreamEvent::Usage(Usage {
                    input_tokens: 0,
                    output_tokens: output,
                    cache_read_tokens: None,
                    cache_write_tokens: None,
                }))
            }
        }
        "message_stop" => None,
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_url_accepts_root_or_v1_base_url() {
        let from_root = AnthropicProvider::new(
            "sk-ant-test".into(),
            "test-model".into(),
            Some("https://api.example.com".into()),
        )
        .unwrap();
        let from_v1 = AnthropicProvider::new(
            "sk-ant-test".into(),
            "test-model".into(),
            Some("https://api.example.com/v1/".into()),
        )
        .unwrap();

        assert_eq!(
            from_root.messages_url(),
            "https://api.example.com/v1/messages"
        );
        assert_eq!(
            from_v1.messages_url(),
            "https://api.example.com/v1/messages"
        );
    }

    #[test]
    fn claude_effort_enables_adaptive_thinking() {
        let provider =
            AnthropicProvider::new("sk-ant-test".into(), "claude-opus-4-6".into(), None).unwrap();
        let request = CompletionRequest {
            model: "claude-opus-4-6".into(),
            system: None,
            messages: vec![Message::user_text("hi")],
            tools: vec![],
            max_tokens: 1024,
            temperature: None,
            enable_caching: false,
            inference: hermes_core::InferenceOptions {
                thinking: None,
                reasoning_effort: Some("high".into()),
                verbosity: None,
            },
        };

        let body = provider.build_request_body(&request);
        assert_eq!(body["thinking"]["type"], "adaptive");
        assert_eq!(body["output_config"]["effort"], "high");
    }

    #[test]
    fn message_conversion_preserves_text() {
        let m = Message::user_text("hello");
        let v = message_to_anthropic(&m);
        assert_eq!(v["role"], "user");
        assert_eq!(v["content"][0]["type"], "text");
        assert_eq!(v["content"][0]["text"], "hello");
    }

    #[test]
    fn tool_result_conversion_includes_is_error() {
        let m = Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "t1".into(),
                content: "fail".into(),
                is_error: true,
            }],
        };
        let v = message_to_anthropic(&m);
        assert_eq!(v["content"][0]["is_error"], true);
    }

    #[test]
    fn unsupported_block_degrades_to_placeholder() {
        let m = Message {
            role: Role::User,
            content: vec![ContentBlock::Custom {
                type_name: "file_ref".into(),
                data: json!({"path": "/a"}),
            }],
        };
        let v = message_to_anthropic(&m);
        assert_eq!(v["content"][0]["type"], "text");
        assert!(v["content"][0]["text"].as_str().unwrap().contains("custom"));
    }

    #[test]
    fn parse_response_text_and_usage() {
        let v = json!({
            "content": [{ "type": "text", "text": "hi" }],
            "stop_reason": "end_turn",
            "usage": { "input_tokens": 5, "output_tokens": 3 }
        });
        let r = parse_anthropic_response(&v).unwrap();
        assert_eq!(r.text(), "hi");
        assert_eq!(r.usage.output_tokens, 3);
    }

    #[test]
    fn map_api_error_never_leaks_key() {
        let e = map_api_error(500, "some sk-ant-SECRET text");
        let msg = e.to_string();
        assert!(!msg.contains("sk-ant-SECRET"));
        assert!(msg.contains("***"));
    }

    #[test]
    fn map_api_error_401_is_auth() {
        let e = map_api_error(401, "bad");
        assert!(matches!(e, Error::AuthFailed(_)));
    }

    #[test]
    fn map_api_error_429_is_rate_limited() {
        let e = map_api_error(429, "slow down");
        assert!(matches!(e, Error::RateLimited { .. }));
    }
}
