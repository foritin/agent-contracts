//! 真机冒烟（一次性验证工具，不进 CI）：
//!
//! 用真实 provider 验证 A01（传输错误显式 Stop 对齐）之后，两个被改动的
//! SSE 解析器（openai / anthropic）的**正常流**未被破坏——正常完成必发
//! Stop、文本完整。凭据经环境变量注入，不落仓库：
//!
//! ```text
//! REAL_SMOKE_PROTOCOL = anthropic | openai
//! REAL_SMOKE_URL      = provider base url
//! REAL_SMOKE_KEY      = api key
//! REAL_SMOKE_MODEL    = model id
//! ```
use agent_contract::provider::{CompletionRequest, LlmProvider, StreamEvent};
use agent_contract::{ContentBlock, Message, Role};
use futures::StreamExt;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let protocol = std::env::var("REAL_SMOKE_PROTOCOL").expect("REAL_SMOKE_PROTOCOL");
    let url = std::env::var("REAL_SMOKE_URL").expect("REAL_SMOKE_URL");
    let key = std::env::var("REAL_SMOKE_KEY").expect("REAL_SMOKE_KEY");
    let model = std::env::var("REAL_SMOKE_MODEL").expect("REAL_SMOKE_MODEL");

    let provider: Arc<dyn LlmProvider> = match protocol.as_str() {
        "anthropic" => Arc::new(
            agent_llm::AnthropicProvider::new(key.clone(), model.clone(), Some(url.clone()))
                .expect("anthropic provider"),
        ),
        "openai" => Arc::new(agent_llm::OpenAiProvider::new(
            key.clone(),
            model.clone(),
            url.clone(),
        )),
        other => panic!("unknown protocol {other}"),
    };

    let request = Arc::new(CompletionRequest {
        model: model.clone(),
        system: Some("You are a terse assistant. Answer in one short sentence.".into()),
        messages: vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: "用一句话说明什么是幂等性。".into(),
            }],
        }],
        tools: vec![],
        hosted_tools: vec![],
        max_tokens: 200,
        temperature: None,
        enable_caching: false,
        inference: Default::default(),
    });

    let started = std::time::Instant::now();
    let mut events = provider.stream(request).await.expect("stream start");
    let mut text = String::new();
    let mut saw_stop = false;
    while let Some(event) = events.next().await {
        match event {
            StreamEvent::TextDelta { text: delta } => text.push_str(&delta),
            StreamEvent::Stop { reason } => {
                saw_stop = true;
                println!("STOP: {reason:?}");
            }
            StreamEvent::Usage(usage) => {
                println!(
                    "USAGE: in={} out={}",
                    usage.input_tokens, usage.output_tokens
                )
            }
            other => println!("EVENT: {other:?}"),
        }
    }
    println!("ELAPSED: {:?}", started.elapsed());
    println!("TEXT: {text}");
    assert!(saw_stop, "正常流必须以 Stop 终结（A01 契约）");
    assert!(!text.trim().is_empty(), "回答文本非空");
    println!("SMOKE OK [{protocol}/{model}]");
}
