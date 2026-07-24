//! MCP server 封装：生命周期、工具列表与调用。
//!
//! 参见 `05-mcp-client.html §4`、`12-api-contracts.html §4`。

use crate::error::McpError;
use crate::transport::{HttpTransport, StdioTransport, Transport};
use hermes_core::{ToolCallOutcome, ToolSource, ToolSpec};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;

use crate::config::ServerSpec;

/// MCP 工具描述。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub input_schema: Value,
}

/// 单个 MCP server 连接。
pub struct McpServer {
    name: String,
    transport: Arc<dyn Transport>,
    tools: tokio::sync::RwLock<Vec<Tool>>,
    connected: tokio::sync::RwLock<bool>,
}

impl McpServer {
    /// 用已建立的传输构造（测试与生产共用）。
    pub fn new(name: impl Into<String>, transport: Arc<dyn Transport>) -> Self {
        Self {
            name: name.into(),
            transport,
            tools: tokio::sync::RwLock::new(Vec::new()),
            connected: tokio::sync::RwLock::new(false),
        }
    }

    /// 由 ServerSpec 异步构造（stdio 需 spawn 子进程）。
    pub async fn from_spec(name: impl Into<String>, spec: ServerSpec) -> Result<Self, McpError> {
        let transport: Arc<dyn Transport> = match spec {
            ServerSpec::Stdio { command, args, env } => {
                Arc::new(StdioTransport::spawn(&command, &args, &env).await?)
            }
            ServerSpec::Http { url, headers } => Arc::new(HttpTransport::new(url, headers)?),
        };
        Ok(Self::new(name, transport))
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// 是否已连接。
    pub async fn is_connected(&self) -> bool {
        *self.connected.read().await
    }

    /// 执行 initialize 握手并拉取工具列表。
    pub async fn connect(&self) -> Result<(), McpError> {
        let params = json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": { "name": "hermes", "version": env!("CARGO_PKG_VERSION") }
        });
        self.transport
            .request("initialize", Some(params))
            .await
            .map_err(|e| McpError::InitializeFailed(e.to_string()))?;
        *self.connected.write().await = true;
        self.refresh_tools().await?;
        Ok(())
    }

    /// 重新拉取工具列表。
    pub async fn refresh_tools(&self) -> Result<(), McpError> {
        let result = self.transport.request("tools/list", None).await?;
        let tools: Vec<Tool> = result
            .get("tools")
            .cloned()
            .map(|t| serde_json::from_value(t).unwrap_or_default())
            .unwrap_or_default();
        *self.tools.write().await = tools;
        Ok(())
    }

    /// 当前工具列表。
    pub async fn tools(&self) -> Vec<Tool> {
        self.tools.read().await.clone()
    }

    /// 转为 ToolSpec 列表（带 server__tool 命名空间）。
    pub async fn tool_specs(&self) -> Vec<ToolSpec> {
        let name = self.name.clone();
        self.tools
            .read()
            .await
            .iter()
            .map(|t| ToolSpec {
                name: format!("{}__{}", name, t.name),
                description: t.description.clone(),
                input_schema: t.input_schema.clone(),
                source: ToolSource::Mcp {
                    server: name.clone(),
                },
                // MCP 工具默认需要确认（03 §4）
                requires_confirmation: true,
            })
            .collect()
    }

    /// 调用工具（tool_name 不含 server 前缀）。
    pub async fn call_tool(
        &self,
        tool_name: &str,
        args: Value,
    ) -> Result<ToolCallOutcome, McpError> {
        if !*self.connected.read().await {
            return Err(McpError::NotConnected(self.name.clone()));
        }
        let params = json!({ "name": tool_name, "arguments": args });
        let result = self.transport.request("tools/call", Some(params)).await?;

        let is_error = result
            .get("isError")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let content = result
            .get("content")
            .and_then(|c| c.as_array())
            .and_then(|arr| {
                arr.iter()
                    .find_map(|c| c.get("text").and_then(|t| t.as_str()).map(String::from))
            })
            .unwrap_or_default();

        Ok(ToolCallOutcome {
            content,
            is_error,
            metadata: Some(json!({ "server": self.name })),
        })
    }

    /// 健康检查：发送 ping。
    pub async fn health_check(&self) -> Result<(), McpError> {
        self.transport.request("ping", None).await?;
        Ok(())
    }
}
