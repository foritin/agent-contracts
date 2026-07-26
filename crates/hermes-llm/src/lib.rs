//! `hermes-llm` -- LLM Provider 实现。
//!
//! 参见 `01-llm-provider.html`。提供 Anthropic / OpenAI 兼容 / DeepSeek 三个真实
//! Provider，以及用于无网络测试的 `MockProvider`。通过 [`create_provider`] 工厂
//! 按 [`ProviderConfig`] 构造 `Box<dyn LlmProvider>`。

pub mod anthropic;
pub mod deepseek;
pub mod mock;
pub mod openai;

pub use anthropic::AnthropicProvider;
pub use deepseek::DeepSeekProvider;
pub use mock::{MockProvider, RecordedTurn};
pub use openai::OpenAiProvider;

use hermes_core::LlmProvider;
use hermes_error::Result;

/// Provider 配置（工厂入参）。
#[derive(Debug, Clone)]
pub enum ProviderConfig {
    Anthropic {
        api_key: String,
        model: String,
        base_url: Option<String>,
    },
    OpenAi {
        api_key: String,
        model: String,
        base_url: String,
    },
    DeepSeek {
        api_key: String,
        model: String,
        base_url: Option<String>,
    },
}

/// 按配置构造 Provider 实例。
pub fn create_provider(config: ProviderConfig) -> Result<Box<dyn LlmProvider>> {
    match config {
        ProviderConfig::Anthropic {
            api_key,
            model,
            base_url,
        } => Ok(Box::new(AnthropicProvider::new(api_key, model, base_url)?)),
        ProviderConfig::OpenAi {
            api_key,
            model,
            base_url,
        } => Ok(Box::new(OpenAiProvider::new(api_key, model, base_url))),
        ProviderConfig::DeepSeek {
            api_key,
            model,
            base_url,
        } => Ok(Box::new(DeepSeekProvider::new_with_base_url(
            api_key, model, base_url,
        ))),
    }
}
