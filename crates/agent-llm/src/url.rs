//! base_url → API 根地址的拼接规则。
//!
//! 两种协议的约定不同，历史上被写成了同一份逻辑，导致带自定义路径的网关全部 404。
//!
//! - **Anthropic Messages**：`base_url` 里不含版本段，客户端一律拼 `/v1/messages`。
//!   例：`https://api.moonshot.cn/anthropic` → `.../anthropic/v1/messages`；
//!   火山方舟套餐 `https://ark.cn-beijing.volces.com/api/coding` → `.../api/coding/v1/messages`。
//!   所以"结尾不是 /v1 就补 /v1"对 Anthropic 是**正确**的。
//!
//! - **OpenAI Chat Completions / Responses**：`base_url` 按约定**已经包含**版本段，
//!   客户端只拼 `/chat/completions` 或 `/responses`。版本段各家不一样：
//!   `/v1`（OpenAI、xAI）、`/v3`（火山方舟）、`/v4`（智谱 `/api/coding/paas/v4`）、
//!   `/compatible-mode/v1`（阿里百炼）、`/openai/v1`（Novita）。
//!   对这些 URL 再补 `/v1` 会拼出 `/v4/v1/chat/completions` → 404。
//!   只有裸域名（没有任何路径）才需要补 `/v1`。

/// OpenAI 系（Chat Completions / Responses）的 API 根地址。
///
/// 规则：URL 带路径 → 原样返回（去掉结尾斜杠）；裸域名 → 补 `/v1`。
pub fn openai_api_root(base_url: &str) -> String {
    let trimmed = base_url.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return String::new();
    }
    if has_path_segment(trimmed) {
        trimmed.to_string()
    } else {
        format!("{trimmed}/v1")
    }
}

/// Anthropic Messages 的 API 根地址：结尾不是 `/v1` 就补一个。
pub fn anthropic_api_root(base_url: &str) -> String {
    let trimmed = base_url.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return String::new();
    }
    if trimmed.ends_with("/v1") {
        trimmed.to_string()
    } else {
        format!("{trimmed}/v1")
    }
}

/// URL 是否带非空路径。解析失败时退回字符串判断，保证离线可测。
fn has_path_segment(url: &str) -> bool {
    if let Ok(parsed) = reqwest::Url::parse(url) {
        let path = parsed.path().trim_matches('/');
        return !path.is_empty();
    }
    // 没有 scheme 之类的畸形输入：找到 "//" 之后还有 "/" 就算带路径
    let after_scheme = url.split_once("//").map(|(_, rest)| rest).unwrap_or(url);
    after_scheme.contains('/')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_origin_gets_v1() {
        assert_eq!(
            openai_api_root("https://api.openai.com"),
            "https://api.openai.com/v1"
        );
        assert_eq!(
            openai_api_root("https://api.deepseek.com/"),
            "https://api.deepseek.com/v1"
        );
    }

    #[test]
    fn explicit_v1_is_left_alone() {
        assert_eq!(
            openai_api_root("https://api.x.ai/v1"),
            "https://api.x.ai/v1"
        );
        assert_eq!(
            openai_api_root("https://api.x.ai/v1/"),
            "https://api.x.ai/v1"
        );
    }

    #[test]
    fn non_v1_version_segments_survive() {
        // 这四条是历史上会被拼成 /v4/v1/... 的真实地址
        assert_eq!(
            openai_api_root("https://open.bigmodel.cn/api/coding/paas/v4"),
            "https://open.bigmodel.cn/api/coding/paas/v4"
        );
        assert_eq!(
            openai_api_root("https://ark.cn-beijing.volces.com/api/coding/v3"),
            "https://ark.cn-beijing.volces.com/api/coding/v3"
        );
        assert_eq!(
            openai_api_root("https://dashscope.aliyuncs.com/compatible-mode/v1"),
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        );
        assert_eq!(
            openai_api_root("https://api.novita.ai/openai/v1"),
            "https://api.novita.ai/openai/v1"
        );
    }

    #[test]
    fn anthropic_always_appends_v1_once() {
        assert_eq!(
            anthropic_api_root("https://api.anthropic.com"),
            "https://api.anthropic.com/v1"
        );
        // 国内 Anthropic 兼容口：路径在前，版本段由客户端补
        assert_eq!(
            anthropic_api_root("https://api.moonshot.cn/anthropic"),
            "https://api.moonshot.cn/anthropic/v1"
        );
        assert_eq!(
            anthropic_api_root("https://ark.cn-beijing.volces.com/api/coding"),
            "https://ark.cn-beijing.volces.com/api/coding/v1"
        );
        assert_eq!(
            anthropic_api_root("https://api.example.com/v1/"),
            "https://api.example.com/v1"
        );
    }

    #[test]
    fn empty_input_stays_empty() {
        assert_eq!(openai_api_root("   "), "");
        assert_eq!(anthropic_api_root(""), "");
    }
}
