//! MCP 服务器配置。
//!
//! 参见 `05-mcp-client.html §2 §3`。支持 stdio 子进程与 Streamable HTTP 两种传输。

use crate::error::McpError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// 单个 MCP server 的连接规格。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ServerSpec {
    /// stdio 子进程传输。
    Stdio {
        command: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        env: HashMap<String, String>,
    },
    /// Streamable HTTP 传输。
    Http {
        url: String,
        #[serde(default)]
        headers: HashMap<String, String>,
    },
}

/// MCP 配置：多个命名 server。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct McpConfig {
    #[serde(default)]
    pub servers: HashMap<String, ServerSpec>,
}

impl McpConfig {
    /// 从 TOML 文件加载。
    pub fn load(path: &Path) -> Result<Self, McpError> {
        let content = std::fs::read_to_string(path)
            .map_err(|_| McpError::ConfigNotFound(path.to_path_buf()))?;
        let config: McpConfig =
            toml::from_str(&content).map_err(|e| McpError::Config(e.to_string()))?;
        Ok(config)
    }

    /// 从默认路径加载（`~/.hermes/config.toml` 的 `[mcp_servers]` 段）。
    pub fn load_default() -> Result<Self, McpError> {
        let path = dirs_home().join(".hermes/config.toml");
        if !path.exists() {
            return Ok(Self::default());
        }
        Self::load(&path)
    }

    /// 构造空配置（用于测试）。
    pub fn empty() -> Self {
        Self::default()
    }

    /// 添加一个 server。
    pub fn with(mut self, name: impl Into<String>, spec: ServerSpec) -> Self {
        self.servers.insert(name.into(), spec);
        self
    }
}

fn dirs_home() -> std::path::PathBuf {
    std::env::var("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stdio_spec_serde() {
        let toml_str = r#"
type = "stdio"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem"]
"#;
        let spec: ServerSpec = toml::from_str(toml_str).unwrap();
        match spec {
            ServerSpec::Stdio { command, args, .. } => {
                assert_eq!(command, "npx");
                assert_eq!(args.len(), 2);
            }
            _ => panic!("expected stdio"),
        }
    }

    #[test]
    fn http_spec_serde() {
        let toml_str = r#"
type = "http"
url = "https://api.github.com/mcp"
"#;
        let spec: ServerSpec = toml::from_str(toml_str).unwrap();
        match spec {
            ServerSpec::Http { url, .. } => assert_eq!(url, "https://api.github.com/mcp"),
            _ => panic!("expected http"),
        }
    }

    #[test]
    fn config_builder() {
        let cfg = McpConfig::empty().with(
            "fs",
            ServerSpec::Stdio {
                command: "npx".into(),
                args: vec![],
                env: HashMap::new(),
            },
        );
        assert_eq!(cfg.servers.len(), 1);
    }
}
