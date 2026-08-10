//! `hermes-llm` -- LLM Provider 实现。
//!
//! 参见 `01-llm-provider.html`。提供 Anthropic Messages / OpenAI Chat Completions /
//! OpenAI Responses / DeepSeek 四个真实 Provider，以及用于无网络测试的
//! `MockProvider`。通过 [`create_provider`] 工厂按 [`ProviderConfig`] 构造
//! `Box<dyn LlmProvider>`。
//!
//! 三种线路协议不能靠厂商名字区分，同一家的不同 base_url 往往是不同协议
//! （火山方舟 `/api/coding` 是 Anthropic 口、`/api/coding/v3` 是 Chat、
//! `/api/v3` 同时有 Chat 和 Responses）。调用方需显式选择。

pub mod anthropic;
pub mod deepseek;
pub mod mock;
pub mod openai;
pub mod responses;
pub mod url;

pub use anthropic::AnthropicProvider;
pub use deepseek::DeepSeekProvider;
pub use mock::{MockProvider, RecordedTurn};
pub use openai::OpenAiProvider;
pub use responses::{ReasoningMode, ResponsesProvider};

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
    /// Kimi For Coding 的 Anthropic Messages 入口。协议形状相同，但上下文窗口不同。
    KimiCodingAnthropic {
        api_key: String,
        model: String,
        base_url: Option<String>,
    },
    OpenAi {
        api_key: String,
        model: String,
        base_url: String,
    },
    /// OpenAI Responses API。`base_url` 必须已包含版本段（如 `/v1`、`/api/v3`）。
    Responses {
        api_key: String,
        model: String,
        base_url: String,
        /// 推理内容策略。默认 [`ReasoningMode::Drop`]（兼容性最好）；
        /// 只有 OpenAI 官方与 xAI 支持 [`ReasoningMode::EncryptedReplay`]。
        reasoning: ReasoningMode,
    },
    /// DeepSeek 的 Responses 兼容口。协议仍是 Responses，但保留厂商身份以启用
    /// 自动前缀缓存 usage 语义与 1M 上下文能力声明。
    DeepSeekResponses {
        api_key: String,
        model: String,
        base_url: String,
    },
    /// DeepSeek 的 Anthropic Messages 兼容口。它使用自动前缀缓存，不需要
    /// Anthropic `cache_control` 标记。
    DeepSeekAnthropic {
        api_key: String,
        model: String,
        base_url: Option<String>,
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
        ProviderConfig::KimiCodingAnthropic {
            api_key,
            model,
            base_url,
        } => Ok(Box::new(AnthropicProvider::new_kimi_coding(
            api_key, model, base_url,
        )?)),
        ProviderConfig::OpenAi {
            api_key,
            model,
            base_url,
        } => Ok(Box::new(OpenAiProvider::new(api_key, model, base_url))),
        ProviderConfig::Responses {
            api_key,
            model,
            base_url,
            reasoning,
        } => Ok(Box::new(
            ResponsesProvider::new(api_key, model, base_url).with_reasoning(reasoning),
        )),
        ProviderConfig::DeepSeekResponses {
            api_key,
            model,
            base_url,
        } => Ok(Box::new(ResponsesProvider::new_deepseek(
            api_key, model, base_url,
        ))),
        ProviderConfig::DeepSeekAnthropic {
            api_key,
            model,
            base_url,
        } => Ok(Box::new(AnthropicProvider::new_deepseek(
            api_key, model, base_url,
        )?)),
        ProviderConfig::DeepSeek {
            api_key,
            model,
            base_url,
        } => Ok(Box::new(DeepSeekProvider::new_with_base_url(
            api_key, model, base_url,
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factory_builds_each_protocol() {
        let anthropic = create_provider(ProviderConfig::Anthropic {
            api_key: "k".into(),
            model: "claude-sonnet-5".into(),
            base_url: None,
        })
        .unwrap();
        assert_eq!(anthropic.name(), "anthropic");

        let kimi = create_provider(ProviderConfig::KimiCodingAnthropic {
            api_key: "k".into(),
            model: "k3-256k".into(),
            base_url: Some("https://api.kimi.com/coding/".into()),
        })
        .unwrap();
        assert_eq!(kimi.name(), "kimi_coding");
        assert_eq!(kimi.capabilities().max_context_tokens, 262_144);

        let chat = create_provider(ProviderConfig::OpenAi {
            api_key: "k".into(),
            model: "gpt-5.5".into(),
            base_url: "https://api.openai.com/v1".into(),
        })
        .unwrap();
        assert_eq!(chat.name(), "openai");

        let responses = create_provider(ProviderConfig::Responses {
            api_key: "k".into(),
            model: "gpt-5.6-sol".into(),
            base_url: "https://api.openai.com/v1".into(),
            reasoning: ReasoningMode::EncryptedReplay,
        })
        .unwrap();
        assert_eq!(responses.name(), "openai_responses");

        let deepseek_responses = create_provider(ProviderConfig::DeepSeekResponses {
            api_key: "k".into(),
            model: "deepseek-v4-flash".into(),
            base_url: "https://api.deepseek.com".into(),
        })
        .unwrap();
        assert_eq!(deepseek_responses.name(), "deepseek_responses");
        assert!(deepseek_responses.capabilities().supports_prompt_caching);

        let deepseek_anthropic = create_provider(ProviderConfig::DeepSeekAnthropic {
            api_key: "k".into(),
            model: "deepseek-v4-pro".into(),
            base_url: Some("https://api.deepseek.com/anthropic".into()),
        })
        .unwrap();
        assert_eq!(deepseek_anthropic.name(), "deepseek_anthropic");
        assert_eq!(
            deepseek_anthropic.capabilities().max_context_tokens,
            1_000_000
        );
    }
}
