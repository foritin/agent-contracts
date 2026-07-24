//! `hermes-ipc` -- IPC 传输层。
//!
//! 参见 `09-ipc-transport.html`。JSON-RPC 2.0 over Unix Socket（macOS/Linux）/
//! Named Pipe（Windows）。帧格式：`[4B big-endian 长度][JSON payload]`。

pub mod client;
pub mod protocol;
pub mod server;

pub use client::IpcClient;
pub use protocol::{
    read_frame, read_message, write_frame, write_message, JsonRpcError, JsonRpcRequest,
    JsonRpcResponse,
};
pub use server::{IpcHandler, IpcServer};
