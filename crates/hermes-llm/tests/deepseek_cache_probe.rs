//! DeepSeek 真实 API 前缀缓存探针（docs/deepseek-prefix-cache.md §6 真实 API 层级）。
//!
//! 验证 P0-A 请求结构（稳定 system + append-only 历史）在真实 DeepSeek API 上的
//! 缓存命中率曲线，并验证 P0-B 的 usage 解析链路（`stream_options.include_usage`
//! → `prompt_cache_hit_tokens` → `cache_read_tokens`）。
//!
//! 用法（key 从环境变量读取，不落盘）：
//! ```text
//! DEEPSEEK_API_KEY=sk-... cargo test -p hermes-llm --test deepseek_cache_probe -- --ignored --nocapture
//! ```
//!
//! 预期：第 1 轮冷启动全 miss（hit=0）；第 2 轮起前缀命中，命中率随轮次增长
//! 趋近 90%+（对照 docs/deepseek-cache-baseline.md 的 80.5% 第二轮基线）。

use futures::StreamExt;
use hermes_core::{
    CompletionRequest, ContentBlock, LlmProvider, Message, Role, StopReason, StreamEvent, Usage,
};
use hermes_llm::deepseek::DeepSeekProvider;

/// 模拟 P0-A 落地后的稳定 system（不含时间戳等动态内容，字节跨轮不变）。
/// 长度约 250 tokens，明显超过 DeepSeek 缓存块粒度（~64 tokens）。
const STABLE_SYSTEM: &str = "You are a senior software architect assistant embedded in an IDE. \
You help with code architecture, refactoring, debugging, performance optimization, and best \
practices. You always ground answers in concrete code examples. When asked about caching you \
explain byte-level prefix caching in detail, including cache block granularity, billing, and how \
clients keep prefixes stable. When asked about concurrency you discuss locks, atomics, and async \
models. When asked about databases you compare SQLite and Postgres tradeoffs. You keep answers \
structured with headings and bullet points. You never invent APIs that do not exist. You admit \
uncertainty when you are not sure. You prefer showing a minimal reproducible example over abstract \
advice. You respond in the language the user uses.";

const ROUNDS: usize = 14;

#[tokio::test]
#[ignore = "需要真实 DeepSeek API key（DEEPSEEK_API_KEY 环境变量）"]
async fn deepseek_prefix_cache_hit_curve() {
    let Ok(api_key) = std::env::var("DEEPSEEK_API_KEY") else {
        eprintln!("[probe] DEEPSEEK_API_KEY 未设置，跳过（真实 API 探针）");
        return;
    };
    let provider = DeepSeekProvider::new(api_key, "deepseek-chat".into());

    // P0-A 请求结构：稳定 system + append-only 历史（每轮追加 assistant 回复 + 新 user 消息）。
    let mut messages = vec![Message::user_text(
        "请用一段话解释什么是字节级前缀缓存，以及它如何影响 API 成本。",
    )];
    let mut curve: Vec<(u32, u32, f32)> = Vec::new();

    for round in 0..ROUNDS {
        let request = CompletionRequest {
            model: "deepseek-chat".into(),
            system: Some(STABLE_SYSTEM.to_string()),
            messages: messages.clone(),
            tools: vec![],
            hosted_tools: vec![],
            max_tokens: 256,
            temperature: None,
            enable_caching: false,
            inference: Default::default(),
        };

        let mut stream = provider
            .stream(request)
            .await
            .expect("DeepSeek 流式请求必须成功");
        let mut reply = String::new();
        let mut usage = Usage::default();
        let mut stop = StopReason::EndTurn;
        while let Some(event) = stream.next().await {
            match event {
                StreamEvent::TextDelta { text } => reply.push_str(&text),
                StreamEvent::Usage(u) => usage = u,
                StreamEvent::Stop { reason } => stop = reason,
                _ => {}
            }
        }

        let hit = usage.cache_read_tokens.unwrap_or(0);
        let miss = usage.cache_write_tokens.unwrap_or(0);
        let rate = if hit + miss > 0 {
            hit as f32 * 100.0 / (hit + miss) as f32
        } else {
            0.0
        };
        curve.push((hit, miss, rate));
        eprintln!(
            "[probe] round {round}: prompt={} hit={hit} miss={miss} rate={rate:.1}% stop={stop:?}",
            usage.input_tokens
        );

        // append-only：追加 assistant 回复 + 下一轮 user 消息（与 agent 循环一致）。
        if !reply.is_empty() {
            messages.push(Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text { text: reply }],
            });
        }
        messages.push(Message::user_text(format!(
            "继续：这是第 {} 轮追问，请用一段话简短回答。",
            round + 1
        )));
    }

    // 尾 3 轮平均命中率（对齐守卫测试 tail_avg 语义）。
    let tail: Vec<f32> = curve[curve.len().saturating_sub(3)..]
        .iter()
        .map(|(_, _, rate)| *rate)
        .collect();
    let tail_avg = tail.iter().sum::<f32>() / tail.len() as f32;
    eprintln!(
        "[probe] tail_avg(3) = {tail_avg:.1}% （对照 baseline 第二轮 80.5%、守卫阈值 90%）"
    );
    // 探针不硬断言（真实网络/限流可能影响单次运行），只输出曲线供人工对照。
    assert!(tail_avg > 0.0, "真实 API 未观测到任何缓存命中——前缀可能不稳定");
}
