//! 厂商/模型族的 wire 方言。
//!
//! 只有目录内已实测冻结的官方线路返回 `Some`；未命中的 kind / 自定义网关返回
//! `None`，调用方保持与改造前完全一致的通用行为，避免把未验证的参数发给网关。

/// reasoning_effort 的发送位置。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffortWire {
    /// 不发送。
    None,
    /// 嵌在 Anthropic `thinking` 对象里。
    Nested,
    /// OpenAI Chat 顶层 `reasoning_effort`。
    TopLevel,
}

/// 一条线路的参数方言与能力声明。所有字段都只影响构造时传入该方言的实例。
#[derive(Debug, Clone)]
pub struct WireDialect {
    /// 合法 thinking wire 值；空 = 永远不发送 thinking 参数。
    pub thinking_vocab: &'static [&'static str],
    /// 本地 `adaptive` 翻译成的 wire 值；None = 原样透传。
    pub adaptive_maps_to: Option<&'static str>,
    /// 合法 reasoning_effort 值；空 = 不过滤透传。
    pub effort_vocab: &'static [&'static str],
    pub effort_wire: EffortWire,
    /// 一律不发送 temperature。
    pub omit_temperature: bool,
    /// 仅 thinking 开启时不发送 temperature。
    pub omit_temperature_when_thinking: bool,
    /// 强制 `stream_options.include_usage`（OpenAI Chat）。
    pub force_stream_usage: bool,
    /// 把明文思考内容回传到下一轮（Thinking 块 / reasoning_content）。
    pub echo_reasoning: bool,
    pub max_context_tokens: u32,
    pub max_output_tokens: u32,
    pub supports_vision: bool,
    /// Anthropic 口是否注入显式 cache_control 断点。
    pub emit_cache_control: bool,
    /// 覆盖 User-Agent；None 时使用 R-Code 默认 UA。
    pub user_agent: Option<&'static str>,
}

/// 协议端口；同一厂商的不同口可能对同一语义使用不同 wire 形状。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DialectPort {
    OpenAiChat,
    AnthropicMessages,
}

/// 把本地 thinking 值翻译成该方言的 wire 值；无法翻译时返回 None（不发送参数）。
pub fn dialect_thinking_value<'a>(
    dialect: &WireDialect,
    value: Option<&'a str>,
) -> Option<&'a str> {
    let value = value?;
    if dialect.thinking_vocab.contains(&value) {
        return Some(value);
    }
    if value == "adaptive" {
        return dialect.adaptive_maps_to;
    }
    None
}

fn ark_vision(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    model.contains("doubao-seed")
}

fn kimi_context(model: &str) -> u32 {
    if model.trim().to_ascii_lowercase() == "k3" {
        1_048_576
    } else {
        262_144
    }
}

fn kimi_effort_vocab(model: &str) -> &'static [&'static str] {
    let model = model.trim().to_ascii_lowercase();
    if model == "k3" || model == "k3-256k" {
        &["low", "high", "max"]
    } else {
        &[]
    }
}

fn ark_anthropic(model: &str, max_context_tokens: u32) -> WireDialect {
    WireDialect {
        thinking_vocab: &["enabled", "disabled", "auto", "adaptive", "low"],
        adaptive_maps_to: None,
        effort_vocab: &[],
        effort_wire: EffortWire::None,
        omit_temperature: false,
        omit_temperature_when_thinking: false,
        force_stream_usage: false,
        echo_reasoning: false,
        max_context_tokens,
        max_output_tokens: 0,
        supports_vision: ark_vision(model),
        emit_cache_control: true,
        user_agent: None,
    }
}

fn ark_chat(model: &str, max_context_tokens: u32) -> WireDialect {
    WireDialect {
        thinking_vocab: &["enabled", "disabled", "auto", "adaptive", "low"],
        adaptive_maps_to: None,
        effort_vocab: &[],
        effort_wire: EffortWire::TopLevel,
        omit_temperature: false,
        omit_temperature_when_thinking: false,
        force_stream_usage: true,
        echo_reasoning: false,
        max_context_tokens,
        max_output_tokens: 0,
        supports_vision: ark_vision(model),
        emit_cache_control: false,
        user_agent: None,
    }
}

fn kimi_anthropic(model: &str) -> WireDialect {
    WireDialect {
        thinking_vocab: &["enabled"],
        adaptive_maps_to: Some("enabled"),
        effort_vocab: kimi_effort_vocab(model),
        effort_wire: EffortWire::Nested,
        omit_temperature: true,
        omit_temperature_when_thinking: false,
        force_stream_usage: false,
        echo_reasoning: true,
        max_context_tokens: kimi_context(model),
        max_output_tokens: 32_768,
        supports_vision: true,
        emit_cache_control: true,
        user_agent: Some("KimiCLI/1.5"),
    }
}

fn kimi_chat(model: &str) -> WireDialect {
    WireDialect {
        thinking_vocab: &["enabled"],
        adaptive_maps_to: Some("enabled"),
        effort_vocab: kimi_effort_vocab(model),
        effort_wire: EffortWire::TopLevel,
        omit_temperature: true,
        omit_temperature_when_thinking: false,
        force_stream_usage: false,
        echo_reasoning: true,
        max_context_tokens: kimi_context(model),
        max_output_tokens: 32_768,
        supports_vision: true,
        emit_cache_control: false,
        user_agent: Some("KimiCLI/1.5"),
    }
}

/// 按稳定厂商身份 + 模型 + 协议口解析方言；未知组合返回 None（保持通用行为）。
pub fn dialect_for(kind: &str, model: &str, port: DialectPort) -> Option<WireDialect> {
    let kind = kind.trim().to_ascii_lowercase();
    match (kind.as_str(), port) {
        ("ark_coding", DialectPort::AnthropicMessages) => Some(ark_anthropic(model, 256_000)),
        ("ark_agent", DialectPort::AnthropicMessages) => Some(ark_anthropic(model, 1_048_576)),
        ("ark_coding_openai", DialectPort::OpenAiChat) => Some(ark_chat(model, 256_000)),
        ("kimi_coding", DialectPort::AnthropicMessages) => Some(kimi_anthropic(model)),
        ("kimi_coding", DialectPort::OpenAiChat) => Some(kimi_chat(model)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ark_anthropic_passthrough_thinking_and_keeps_temperature() {
        let d = dialect_for("ark_coding", "ark-code-latest", DialectPort::AnthropicMessages)
            .unwrap();
        assert!(d.thinking_vocab.contains(&"adaptive"));
        assert!(d.adaptive_maps_to.is_none());
        assert!(!d.omit_temperature && !d.omit_temperature_when_thinking);
        assert_eq!(d.effort_wire, EffortWire::None);
        assert!(!d.echo_reasoning);
        assert_eq!(d.max_context_tokens, 256_000);
    }

    #[test]
    fn ark_agent_uses_one_million_context() {
        let d = dialect_for("ark_agent", "deepseek-v4-flash", DialectPort::AnthropicMessages)
            .unwrap();
        assert_eq!(d.max_context_tokens, 1_048_576);
        assert!(!d.supports_vision);
    }

    #[test]
    fn ark_chat_forces_stream_usage() {
        let d = dialect_for("ark_coding_openai", "ark-code-latest", DialectPort::OpenAiChat)
            .unwrap();
        assert!(d.force_stream_usage);
        assert_eq!(d.effort_wire, EffortWire::TopLevel);
    }

    #[test]
    fn kimi_omits_temperature_and_maps_adaptive_to_enabled() {
        let d = dialect_for("kimi_coding", "k3-256k", DialectPort::AnthropicMessages).unwrap();
        assert!(d.omit_temperature);
        assert_eq!(d.adaptive_maps_to, Some("enabled"));
        assert!(d.thinking_vocab.contains(&"enabled"));
        assert!(!d.thinking_vocab.contains(&"disabled"));
        assert_eq!(d.effort_wire, EffortWire::Nested);
        assert_eq!(d.effort_vocab, &["low", "high", "max"]);
        assert!(d.echo_reasoning);
        assert_eq!(d.max_context_tokens, 262_144);
        assert_eq!(d.user_agent, Some("KimiCLI/1.5"));
    }

    #[test]
    fn kimi_k3_reports_one_million_context() {
        let d = dialect_for("kimi_coding", "k3", DialectPort::OpenAiChat).unwrap();
        assert_eq!(d.max_context_tokens, 1_048_576);
        assert_eq!(d.effort_wire, EffortWire::TopLevel);
    }

    #[test]
    fn kimi_coding_models_without_effort_leave_vocab_empty() {
        let d =
            dialect_for("kimi_coding", "kimi-for-coding", DialectPort::AnthropicMessages).unwrap();
        assert!(d.effort_vocab.is_empty());
    }

    #[test]
    fn unknown_kinds_and_payg_ark_fall_back_to_none() {
        assert!(dialect_for("ark", "doubao-seed-2-1-pro", DialectPort::OpenAiChat).is_none());
        assert!(dialect_for("my_relay", "m", DialectPort::OpenAiChat).is_none());
        assert!(dialect_for("", "m", DialectPort::AnthropicMessages).is_none());
        assert!(dialect_for("deepseek", "deepseek-v4-flash", DialectPort::OpenAiChat).is_none());
    }
}
