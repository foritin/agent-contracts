//! DeepSeek Provider：复用 OpenAI 兼容实现，指向 DeepSeek base_url。
//!
//! 参见 `01-llm-provider.html §4.3`。

use crate::openai::OpenAiProvider;
use agent_contract::{Capabilities, CompletionRequest, CompletionResponse, LlmProvider, StreamEvent};
use agent_error::Result;

const DEEPSEEK_BASE_URL: &str = "https://api.deepseek.com";

pub struct DeepSeekProvider {
    inner: OpenAiProvider,
    max_context_tokens: u32,
    max_output_tokens: u32,
}

impl DeepSeekProvider {
    pub fn new(api_key: String, model: String) -> Self {
        Self::new_with_base_url(api_key, model, None)
    }

    /// 允许自定义兼容网关，同时保留 DeepSeek 官方地址作为安全默认值。
    pub fn new_with_base_url(api_key: String, model: String, base_url: Option<String>) -> Self {
        let model = model.trim().to_ascii_lowercase();
        // 实测：deepseek-chat 已被服务端别名到 deepseek-v4-flash（1M 窗口）。
        let is_v4 = model.starts_with("deepseek-v4-") || model == "deepseek-chat";
        let max_context_tokens = if is_v4 { 1_000_000 } else { 64_000 };
        // DeepSeek V4 的单次输出上限是 393_216（API 报错口径）；非 V4 未声明，
        // 由运行时回退到旧的 max_tokens 启发。
        let max_output_tokens = if is_v4 { 393_216 } else { 0 };
        let base_url = base_url
            .filter(|url| !url.trim().is_empty())
            .unwrap_or_else(|| DEEPSEEK_BASE_URL.to_string());
        Self {
            // 即使用户把 DeepSeek 指向自定义网关，也必须请求流式 usage，才能
            // 观测 prompt_cache_hit/miss_tokens。兼容性回退由 OpenAiProvider 处理。
            inner: OpenAiProvider::new(api_key, model, base_url).with_stream_usage(),
            max_context_tokens,
            max_output_tokens,
        }
    }
}

#[async_trait::async_trait]
impl LlmProvider for DeepSeekProvider {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        self.inner.complete(request).await
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<futures::stream::BoxStream<'static, StreamEvent>> {
        self.inner.stream(request).await
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            supports_streaming: true,
            supports_tool_use: true,
            supports_vision: false,
            // DeepSeek 对字节稳定前缀自动缓存 KV（无需 API 开关），能力声明置 true
            // （https://github.com/foritin/r-code/blob/main/docs/archive/deepseek-prefix-cache.md §3 A8，P0-B）。
            supports_prompt_caching: true,
            max_context_tokens: self.max_context_tokens,
            max_output_tokens: self.max_output_tokens,
        }
    }

    fn name(&self) -> &str {
        "deepseek"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_and_capabilities() {
        let p = DeepSeekProvider::new("k".into(), "deepseek-chat".into());
        assert_eq!(p.name(), "deepseek");
        assert!(p.capabilities().supports_prompt_caching);
    }

    #[test]
    fn v4_advertises_one_million_token_context() {
        let p = DeepSeekProvider::new("k".into(), "deepseek-v4-pro".into());
        assert_eq!(p.capabilities().max_context_tokens, 1_000_000);
    }

    #[test]
    fn v4_and_chat_alias_declare_max_output() {
        let v4 = DeepSeekProvider::new("k".into(), "deepseek-v4-pro".into());
        assert_eq!(v4.capabilities().max_output_tokens, 393_216);

        let chat = DeepSeekProvider::new("k".into(), "deepseek-chat".into());
        assert_eq!(chat.capabilities().max_output_tokens, 393_216);
    }

    #[test]
    fn empty_key_still_constructs() {
        // DeepSeek 复用 OpenAi，构造不做 key 校验（与文档示例一致）
        let p = DeepSeekProvider::new("".into(), "m".into());
        assert_eq!(p.name(), "deepseek");
    }
}
