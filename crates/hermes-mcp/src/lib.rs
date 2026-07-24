//! `hermes-mcp` -- MCP 客户端。
//!
//! 参见 `05-mcp-client.html`。封装 MCP 协议（JSON-RPC 2.0），支持 stdio 子进程
//! 与 Streamable HTTP 两种传输，并通过 `McpToolHost` 聚合为 `ToolHost`。

pub mod config;
pub mod error;
pub mod host;
pub mod server;
pub mod transport;

pub use config::{McpConfig, ServerSpec};
pub use error::McpError;
pub use host::McpToolHost;
pub use server::{McpServer, Tool};
pub use transport::{HttpTransport, MockTransport, StdioTransport, Transport};
