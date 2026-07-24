//! DeepSeek Provider：复用 OpenAI 兼容实现，指向 DeepSeek base_url。
//!
//! 参见 `01-llm-provider.html §4.3`。

use crate::openai::OpenAiProvider;
use hermes_core::{Capabilities, CompletionRequest, CompletionResponse, LlmProvider, StreamEvent};
use hermes_error::Result;

const DEEPSEEK_BASE_URL: &str = "https://api.deepseek.com";

pub struct DeepSeekProvider {
    inner: OpenAiProvider,
}

impl DeepSeekProvider {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            inner: OpenAiProvider::new(api_key, model, DEEPSEEK_BASE_URL.to_string()),
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
            supports_prompt_caching: false,
            max_context_tokens: 64_000,
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
        assert!(!p.capabilities().supports_prompt_caching);
    }

    #[test]
    fn empty_key_still_constructs() {
        // DeepSeek 复用 OpenAi，构造不做 key 校验（与文档示例一致）
        let p = DeepSeekProvider::new("".into(), "m".into());
        assert_eq!(p.name(), "deepseek");
    }
}
