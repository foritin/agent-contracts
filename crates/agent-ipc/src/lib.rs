//! `agent-ipc` -- IPC 传输层。
//!
//! 参见 `09-ipc-transport.html`。JSON-RPC 2.0 over Unix Socket（macOS/Linux）/
//! Named Pipe（Windows）。帧格式：`[4B big-endian 长度][JSON payload]`。

// clippy 1.99 对 async_trait 展开的 boxing 方法报 double_must_use——宏输出不可控，crate 级豁免。
#![allow(clippy::double_must_use)]

pub mod client;
pub mod protocol;
pub mod server;

pub use client::IpcClient;
pub use protocol::{
    read_frame, read_message, write_frame, write_message, JsonRpcError, JsonRpcRequest,
    JsonRpcResponse,
};
pub use server::{IpcHandler, IpcServer};
