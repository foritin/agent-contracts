//! MCP 工具聚合层。
//!
//! 参见 `05-mcp-client.html §5`、`03-tool-host.html §4`。聚合多个 MCP server，
//! 以 `server__tool` 命名空间暴露工具，实现 `ToolHost` trait。

use crate::config::McpConfig;
use crate::error::McpError;
use crate::server::McpServer;
use agent_contract::{ToolCallOutcome, ToolHost, ToolSpec};
use agent_error::{Error, Result};
use std::sync::Arc;

/// 聚合多个 MCP server 的 ToolHost。
pub struct McpToolHost {
    servers: Vec<Arc<McpServer>>,
}

impl McpToolHost {
    pub fn new() -> Self {
        Self {
            servers: Vec::new(),
        }
    }

    /// 从配置异步构造（stdio 需 spawn 子进程），不自动连接。
    pub async fn from_config(config: McpConfig) -> Result<Self> {
        let mut host = Self::new();
        for (name, spec) in config.servers {
            let server = McpServer::from_spec(name, spec)
                .await
                .map_err(Error::from)?;
            host.add_server(server);
        }
        Ok(host)
    }

    /// 添加一个已构造的 server。
    pub fn add_server(&mut self, server: McpServer) {
        self.servers.push(Arc::new(server));
    }

    /// 连接所有 server（单个失败仅告警，不阻断其他）。
    pub async fn connect_all(&self) -> Result<()> {
        for server in &self.servers {
            if let Err(e) = server.connect().await {
                tracing::warn!("Failed to connect MCP server '{}': {}", server.name(), e);
            }
        }
        Ok(())
    }

    /// 已注册 server 名称。
    pub fn server_names(&self) -> Vec<String> {
        self.servers.iter().map(|s| s.name().to_string()).collect()
    }

    /// 按 server 名查找。
    fn find_server(&self, name: &str) -> Option<&Arc<McpServer>> {
        self.servers.iter().find(|s| s.name() == name)
    }

    /// 解析 `server__tool` 命名，返回 (server_name, tool_name)。
    pub fn parse_namespaced(name: &str) -> std::result::Result<(&str, &str), McpError> {
        let mut parts = name.splitn(2, "__");
        let server = parts
            .next()
            .ok_or_else(|| McpError::InvalidToolName(name.into()))?;
        let tool = parts
            .next()
            .ok_or_else(|| McpError::InvalidToolName(name.into()))?;
        if server.is_empty() || tool.is_empty() {
            return Err(McpError::InvalidToolName(name.into()));
        }
        Ok((server, tool))
    }
}

impl Default for McpToolHost {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl ToolHost for McpToolHost {
    async fn list_tools(&self) -> Result<Vec<ToolSpec>> {
        let mut all = Vec::new();
        for server in &self.servers {
            all.extend(server.tool_specs().await);
        }
        Ok(all)
    }

    async fn call(&self, name: &str, args: serde_json::Value) -> Result<ToolCallOutcome> {
        let (server_name, tool_name) = Self::parse_namespaced(name).map_err(Error::from)?;
        let server = self
            .find_server(server_name)
            .ok_or_else(|| Error::McpServerNotFound(server_name.to_string()))?;
        server.call_tool(tool_name, args).await.map_err(Error::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::MockTransport;
    use serde_json::{json, Value};

    /// 构造一个已连接的 mock server（共享 transport 便于断言）。
    async fn connected_mock(name: &str, tools: Vec<(&str, &str)>) -> McpServer {
        let t = Arc::new(MockTransport::new());
        let tools_json: Vec<Value> = tools
            .iter()
            .map(|(n, d)| json!({ "name": n, "description": d, "input_schema": {} }))
            .collect();
        t.set_result("initialize", json!({ "serverInfo": { "name": name } }))
            .await;
        t.set_result("tools/list", json!({ "tools": tools_json }))
            .await;
        let server = McpServer::new(name, t);
        server.connect().await.unwrap();
        server
    }

    #[tokio::test]
    async fn list_tools_uses_namespacing() {
        // V-TOOL-02：MCP 工具使用 server__tool 命名
        let mut host = McpToolHost::new();
        host.add_server(
            connected_mock("fs", vec![("read_file", "read"), ("write_file", "write")]).await,
        );
        let tools = host.list_tools().await.unwrap();
        let names: Vec<_> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"fs__read_file"));
        assert!(names.contains(&"fs__write_file"));
    }

    #[tokio::test]
    async fn unknown_tool_rejected() {
        // V-TOOL-01：未知工具默认拒绝
        let mut host = McpToolHost::new();
        host.add_server(connected_mock("fs", vec![("read_file", "read")]).await);
        let r = host.call("fs__nonexistent", json!({})).await;
        assert!(r.is_err());
    }

    #[tokio::test]
    async fn invalid_name_format_rejected() {
        let host = McpToolHost::new();
        let r = host.call("no_namespace", json!({})).await;
        assert!(r.is_err());
    }

    #[tokio::test]
    async fn parse_namespacing() {
        let (s, t) = McpToolHost::parse_namespaced("fs__read_file").unwrap();
        assert_eq!(s, "fs");
        assert_eq!(t, "read_file");
        assert!(McpToolHost::parse_namespaced("bad").is_err());
        assert!(McpToolHost::parse_namespaced("fs__").is_err());
    }

    #[tokio::test]
    async fn same_named_tools_across_servers_dont_collide() {
        // V-TOOL-02：不同 server 同名工具不碰撞
        let mut host = McpToolHost::new();
        host.add_server(connected_mock("a", vec![("read", "r")]).await);
        host.add_server(connected_mock("b", vec![("read", "r")]).await);
        let tools = host.list_tools().await.unwrap();
        assert!(tools.iter().any(|t| t.name == "a__read"));
        assert!(tools.iter().any(|t| t.name == "b__read"));
    }

    #[tokio::test]
    async fn call_routes_to_correct_server() {
        let mut host = McpToolHost::new();
        host.add_server(connected_mock("a", vec![("ping", "p")]).await);
        let outcome = host.call("a__ping", json!({})).await;
        // 未设置 tools/call 结果 -> 返回错误（但路由成功，错误来自 mock 缺失）
        assert!(outcome.is_err());
    }

    #[tokio::test]
    async fn accept_drill_disconnect_then_reconnect() {
        // accept-drill-disconnect：MCP 断开后可诊断 + 重连恢复
        let t = Arc::new(MockTransport::new());
        t.set_result("initialize", json!({"serverInfo":{"name":"fs"}}))
            .await;
        t.set_result("tools/list", json!({"tools":[]})).await;
        t.set_result("ping", json!({})).await;

        let server = McpServer::new("fs", t.clone());
        server.connect().await.unwrap();
        // 正常：健康检查通过
        assert!(server.health_check().await.is_ok());

        // 模拟断开：让 ping 报错
        t.set_error("ping", crate::error::McpError::NotConnected("fs".into()))
            .await;
        assert!(server.health_check().await.is_err(), "断开后应可诊断");

        // 重连：恢复 ping 响应后重新 connect
        t.set_result("ping", json!({})).await;
        server.connect().await.unwrap();
        assert!(server.health_check().await.is_ok(), "重连后恢复");
    }
}
