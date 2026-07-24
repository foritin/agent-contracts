//! Token 使用量统计。
//!
//! 参见 `01-llm-provider.html §3`。用于会话累计与计费 / 上下文预算。

use serde::{Deserialize, Serialize};

/// 单次完成的 token 使用量。
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<u32>,
}

impl Usage {
    pub fn new(input: u32, output: u32) -> Self {
        Self {
            input_tokens: input,
            output_tokens: output,
            cache_read_tokens: None,
            cache_write_tokens: None,
        }
    }

    /// 总 token 数（输入 + 输出，不含缓存）。
    pub fn total(&self) -> u32 {
        self.input_tokens + self.output_tokens
    }
}

impl std::ops::AddAssign for Usage {
    fn add_assign(&mut self, rhs: Self) {
        self.input_tokens += rhs.input_tokens;
        self.output_tokens += rhs.output_tokens;
        self.cache_read_tokens = match (self.cache_read_tokens, rhs.cache_read_tokens) {
            (Some(a), Some(b)) => Some(a + b),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        self.cache_write_tokens = match (self.cache_write_tokens, rhs.cache_write_tokens) {
            (Some(a), Some(b)) => Some(a + b),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_assign_accumulates() {
        let mut a = Usage::new(10, 5);
        a += Usage::new(3, 2);
        assert_eq!(a.input_tokens, 13);
        assert_eq!(a.output_tokens, 7);
    }

    #[test]
    fn cache_tokens_merge() {
        let mut a = Usage {
            cache_read_tokens: Some(4),
            ..Usage::new(1, 1)
        };
        a += Usage {
            cache_read_tokens: Some(6),
            ..Usage::new(0, 0)
        };
        assert_eq!(a.cache_read_tokens, Some(10));
    }
}
