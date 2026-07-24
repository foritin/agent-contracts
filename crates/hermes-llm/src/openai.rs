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
        format!(
            "{}/v1/chat/completions",
            self.base_url.trim_end_matches('/')
        )
    }

    fn build_body(&self, request: &CompletionRequest, stream: bool) -> Value {
        let mut messages: Vec<Value> = Vec::new();

        // system 作为角色注入
        if let Some(system) = &request.system {
            messages.push(json!({ "role": "system", "content": system }));
        }

        for m in &request.messages {
            messages.push(message_to_openai(m));
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
            body["stream_options"] = json!({ "include_usage": true });
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
            .map_err(|e| Error::Provider(e.to_string().replace(&self.api_key, "***")))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(map_api_error(status.as_u16(), &text));
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
            .map_err(|e| Error::Provider(e.to_string().replace(&self.api_key, "***")))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(map_api_error(status.as_u16(), &text));
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

pub fn message_to_openai(msg: &Message) -> Value {
    // 若含 ToolResult 块 -> role: tool 消息（取首个）
    if let Some(ContentBlock::ToolResult {
        tool_use_id,
        content,
        ..
    }) = msg
        .content
        .iter()
        .find(|b| matches!(b, ContentBlock::ToolResult { .. }))
    {
        return json!({
            "role": "tool",
            "tool_call_id": tool_use_id,
            "content": content,
        });
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
        return json!({
            "role": "assistant",
            "content": null,
            "tool_calls": tool_calls,
        });
    }

    // 普通文本消息
    let role = match msg.role {
        Role::User => "user",
        Role::Assistant => "assistant",
    };
    json!({ "role": role, "content": msg.text_content() })
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

fn map_api_error(status: u16, body: &str) -> Error {
    let body = body.chars().take(512).collect::<String>();
    let body = body.replace("Bearer ", "").replace("sk-", "***");
    match status {
        401 | 403 => Error::AuthFailed("authentication failed (check api_key)".into()),
        429 => Error::RateLimited { retry_after: 0 },
        404 => Error::ModelNotFound(format!("model not found (status {status})")),
        _ => Error::ApiError {
            status,
            message: body,
        },
    }
}

// ── SSE 流解析 ────────────────────────────────────────────────

fn parse_openai_sse(
    byte_stream: impl futures::Stream<Item = std::result::Result<bytes::Bytes, reqwest::Error>>,
) -> impl futures::Stream<Item = StreamEvent> {
    use futures::StreamExt;
    let mut buffer = String::new();
    let mut tool_args: std::collections::HashMap<u32, (String, String)> =
        std::collections::HashMap::new();

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
                    continue;
                }
                if let Some(ev) = parse_one_openai(data, &mut tool_args) {
                    events.push(ev);
                }
            }
            events
        })
        .flat_map(futures::stream::iter)
}

fn parse_one_openai(
    data: &str,
    tool_args: &mut std::collections::HashMap<u32, (String, String)>,
) -> Option<StreamEvent> {
    let v: Value = serde_json::from_str(data).ok()?;
    let choice = v.get("choices")?.get(0)?;
    let delta = choice.get("delta")?;

    if let Some(content) = delta.get("content").and_then(|t| t.as_str()) {
        if !content.is_empty() {
            return Some(StreamEvent::TextDelta {
                text: content.to_string(),
            });
        }
    }

    if let Some(tool_calls) = delta.get("tool_calls").and_then(|t| t.as_array()) {
        for tc in tool_calls {
            let idx = tc.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as u32;
            let entry = tool_args
                .entry(idx)
                .or_insert_with(|| (String::new(), String::new()));
            if let Some(id) = tc.get("id").and_then(|t| t.as_str()) {
                entry.0 = id.to_string();
            }
            if let Some(name) = tc.pointer("/function/name").and_then(|t| t.as_str()) {
                if !name.is_empty() {
                    let id = entry.0.clone();
                    return Some(StreamEvent::ToolUseStart {
                        id,
                        name: name.to_string(),
                    });
                }
            }
            if let Some(args) = tc.pointer("/function/arguments").and_then(|t| t.as_str()) {
                entry.1.push_str(args);
                let id = entry.0.clone();
                return Some(StreamEvent::ToolUseDelta {
                    id,
                    input_json: args.to_string(),
                });
            }
        }
    }

    if let Some(reason) = choice.get("finish_reason").and_then(|t| t.as_str()) {
        // 流末：若有待完成的工具调用，先发出 ToolUseComplete；否则发出 Stop。
        if let Some((_, (id, args))) = tool_args.drain().next() {
            let input: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
            return Some(StreamEvent::ToolUseComplete { id, input });
        }
        return Some(StreamEvent::Stop {
            reason: parse_finish_reason(reason),
        });
    }

    if let Some(usage) = v.get("usage") {
        return Some(StreamEvent::Usage(Usage {
            input_tokens: usage
                .get("prompt_tokens")
                .and_then(|t| t.as_u64())
                .unwrap_or(0) as u32,
            output_tokens: usage
                .get("completion_tokens")
                .and_then(|t| t.as_u64())
                .unwrap_or(0) as u32,
            cache_read_tokens: None,
            cache_write_tokens: None,
        }));
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let e = map_api_error(401, "Bearer sk-SECRET");
        let msg = e.to_string();
        assert!(!msg.contains("sk-SECRET"));
        assert!(matches!(e, Error::AuthFailed(_)));
    }
}
