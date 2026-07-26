//! ToolHost 抽象：工具执行层。
//!
//! 参见 `03-tool-host.html`、`12-api-contracts.html §3`。
//! trait 定义在 `hermes-core`；`NullToolHost` / `CompositeToolHost` 也在本 crate
//! `NullToolHost` / `CompositeToolHost` 也在本 crate
//! 提供；`McpToolHost` 在 `hermes-mcp`。产品专属 ToolHost（如网关桥接）由产品层实现。

use crate::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// 工具调用结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallOutcome {
    pub content: String,
    #[serde(default)]
    pub is_error: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

/// 工具规格。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
    pub source: ToolSource,
    /// 默认需要确认（见 14 篇兼容性表：`requires_confirmation` 默认 true）。
    #[serde(default = "default_requires_confirmation")]
    pub requires_confirmation: bool,
}

fn default_requires_confirmation() -> bool {
    true
}

/// 工具来源。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolSource {
    /// 内置工具。
    Builtin,
    /// MCP 工具。
    Mcp { server: String },
    /// 产品层自定义工具来源（如网关代理等）。公共层不解释其语义。
    Custom { id: String },
}

/// 工具执行抽象。聚合多个工具来源，为 Agent 循环提供统一调用接口。
#[async_trait::async_trait]
pub trait ToolHost: Send + Sync {
    /// 列出所有可用工具。
    async fn list_tools(&self) -> Result<Vec<ToolSpec>>;

    /// 执行工具调用。
    async fn call(&self, name: &str, args: Value) -> Result<ToolCallOutcome>;

    /// 执行带模型工具调用 ID 的调用。
    ///
    /// 默认实现保持旧 ToolHost 兼容；需要把派生资源关联到原始调用的宿主可覆盖它。
    async fn call_with_id(
        &self,
        _call_id: &str,
        name: &str,
        args: Value,
    ) -> Result<ToolCallOutcome> {
        self.call(name, args).await
    }

    /// 批量执行（默认串行；可被覆盖为并行）。
    async fn call_batch(&self, calls: Vec<(String, Value)>) -> Vec<Result<ToolCallOutcome>> {
        let mut results = Vec::with_capacity(calls.len());
        for (name, args) in calls {
            results.push(self.call(&name, args).await);
        }
        results
    }
}

/// 空实现：无工具，调用一律拒绝。
///
/// 参见 `03-tool-host.html §3`。用于未配置工具时的安全默认。
pub struct NullToolHost;

#[async_trait::async_trait]
impl ToolHost for NullToolHost {
    async fn list_tools(&self) -> Result<Vec<ToolSpec>> {
        Ok(Vec::new())
    }

    async fn call(&self, name: &str, _: Value) -> Result<ToolCallOutcome> {
        Err(crate::Error::ToolHost(format!(
            "no tool host configured; cannot call {name:?}"
        )))
    }
}

/// 组合多个 ToolHost，按优先级查找。
///
/// 参见 `03-tool-host.html §6`。
pub struct CompositeToolHost {
    hosts: Vec<Box<dyn ToolHost>>,
}

impl CompositeToolHost {
    pub fn new() -> Self {
        Self { hosts: Vec::new() }
    }

    pub fn add_host(&mut self, host: Box<dyn ToolHost>) {
        self.hosts.push(host);
    }
}

impl Default for CompositeToolHost {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl ToolHost for CompositeToolHost {
    async fn list_tools(&self) -> Result<Vec<ToolSpec>> {
        let mut all = Vec::new();
        for host in &self.hosts {
            all.extend(host.list_tools().await?);
        }
        Ok(all)
    }

    async fn call(&self, name: &str, args: Value) -> Result<ToolCallOutcome> {
        // 按优先级尝试每个 host
        for host in &self.hosts {
            let tools = host.list_tools().await?;
            if tools.iter().any(|t| t.name == name) {
                return host.call(name, args).await;
            }
        }
        Err(crate::Error::ToolNotFound(name.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toolspec_default_requires_confirmation() {
        let json =
            r#"{"name":"x","description":"d","input_schema":{},"source":{"kind":"builtin"}}"#;
        let spec: ToolSpec = serde_json::from_str(json).unwrap();
        assert!(spec.requires_confirmation);
    }

    #[test]
    fn toolsource_mcp_tagged() {
        let s = ToolSource::Mcp {
            server: "fs".into(),
        };
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains(r#""kind":"mcp""#));
    }

    #[tokio::test]
    async fn null_tool_host_rejects() {
        let host = NullToolHost;
        let tools = host.list_tools().await.unwrap();
        assert!(tools.is_empty());
        let r = host.call("anything", serde_json::json!({})).await;
        assert!(r.is_err());
    }
}
