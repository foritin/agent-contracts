//! IPC 服务端：Unix Socket（macOS/Linux）/ Named Pipe（Windows）。
//!
//! 参见 `09-ipc-transport.html §4 §6`。

use crate::protocol::{read_message, write_message, JsonRpcError, JsonRpcRequest, JsonRpcResponse};
use hermes_error::{Error, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::UnixListener;

/// 请求处理器。
#[async_trait::async_trait]
pub trait IpcHandler: Send + Sync {
    async fn handle(&self, params: Value) -> Result<Value>;
}

/// IPC 服务端。
pub struct IpcServer {
    listener: UnixListener,
    handlers: HashMap<String, Arc<dyn IpcHandler>>,
    socket_path: PathBuf,
}

impl IpcServer {
    /// 绑定 Unix Socket，清理旧 socket 文件，设置 0o600 权限。
    pub fn bind(socket_path: PathBuf) -> Result<Self> {
        if socket_path.exists() {
            std::fs::remove_file(&socket_path)?;
        }
        if let Some(parent) = socket_path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let listener = UnixListener::bind(&socket_path)
            .map_err(|e| Error::Ipc(format!("bind {} failed: {e}", socket_path.display())))?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&socket_path, std::fs::Permissions::from_mode(0o600))?;
        }

        Ok(Self {
            listener,
            handlers: HashMap::new(),
            socket_path,
        })
    }

    /// 注册方法处理器。
    pub fn register(&mut self, method: &str, handler: Arc<dyn IpcHandler>) {
        self.handlers.insert(method.to_string(), handler);
    }

    /// 服务端 socket 路径。
    pub fn socket_path(&self) -> &PathBuf {
        &self.socket_path
    }

    /// 开始接受连接（阻塞，通常 spawn 到后台）。
    pub async fn serve(&self) -> Result<()> {
        loop {
            let (stream, _addr) = self
                .listener
                .accept()
                .await
                .map_err(|e| Error::Ipc(format!("accept failed: {e}")))?;
            let handlers = self.handlers.clone();
            tokio::spawn(async move {
                if let Err(e) = Self::handle_connection(stream, handlers).await {
                    tracing::error!("IPC connection error: {e}");
                }
            });
        }
    }

    async fn handle_connection(
        mut stream: tokio::net::UnixStream,
        handlers: HashMap<String, Arc<dyn IpcHandler>>,
    ) -> Result<()> {
        use tokio::io::AsyncWriteExt;
        loop {
            let req: JsonRpcRequest = match read_message(&mut stream).await {
                Ok(r) => r,
                Err(Error::Io(ref e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    // 连接关闭
                    return Ok(());
                }
                Err(e) => return Err(e),
            };

            // 通知（无 id）不返回响应
            let is_notification = req.id.is_none();

            // 版本握手：不兼容时返回明确错误（14 §4 兼容性表）
            if !req.is_version_compatible(crate::protocol::PROTOCOL_VERSION) {
                let resp = JsonRpcResponse::error(
                    req.id,
                    JsonRpcError::INVALID_REQUEST,
                    format!(
                        "incompatible protocol version: server={}, client={:?}",
                        crate::protocol::PROTOCOL_VERSION,
                        req.version
                    ),
                );
                if !is_notification {
                    write_message(&mut stream, &resp).await?;
                    let _ = stream.flush().await;
                }
                continue;
            }

            let response = match handlers.get(&req.method) {
                Some(handler) => match handler.handle(req.params.unwrap_or(Value::Null)).await {
                    Ok(result) => {
                        if is_notification {
                            continue;
                        }
                        JsonRpcResponse::success(req.id.unwrap_or_default(), result)
                    }
                    Err(e) => {
                        JsonRpcResponse::error(req.id, JsonRpcError::INTERNAL_ERROR, e.to_string())
                    }
                },
                None => JsonRpcResponse::error(
                    req.id,
                    JsonRpcError::METHOD_NOT_FOUND,
                    format!("method not found: {}", req.method),
                ),
            };

            if is_notification {
                continue;
            }
            write_message(&mut stream, &response).await?;
            // 显式 flush（write_message 已 flush，此处确保 stream 刷新）
            let _ = stream.flush().await;
        }
    }
}

// 错误码常量见 protocol::JsonRpcError

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use serde_json::json;
    use std::sync::Arc;

    struct EchoHandler;
    #[async_trait]
    impl IpcHandler for EchoHandler {
        async fn handle(&self, params: Value) -> Result<Value> {
            Ok(params)
        }
    }

    #[tokio::test]
    async fn server_handles_request() {
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("ipc.sock");
        let mut server = IpcServer::bind(socket.clone()).unwrap();
        server.register("echo", Arc::new(EchoHandler));

        let path = server.socket_path().clone();
        tokio::spawn(async move {
            let _ = server.serve().await;
        });

        // 客户端
        use crate::client::IpcClient;
        let client = IpcClient::connect(&path).await.unwrap();
        let result = client.call("echo", json!({"msg": "hi"})).await.unwrap();
        assert_eq!(result["msg"], "hi");
    }

    #[tokio::test]
    async fn server_unknown_method_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("ipc2.sock");
        let mut server = IpcServer::bind(socket.clone()).unwrap();
        server.register("echo", Arc::new(EchoHandler));
        let path = server.socket_path().clone();
        tokio::spawn(async move {
            let _ = server.serve().await;
        });

        use crate::client::IpcClient;
        let client = IpcClient::connect(&path).await.unwrap();
        let r = client.call("nonexistent", json!({})).await;
        assert!(r.is_err());
    }

    #[tokio::test]
    async fn version_handshake_rejects_incompatible() {
        // impl-p4-handshake：版本不兼容返回明确错误
        use crate::protocol::{read_message, write_message, JsonRpcRequest, JsonRpcResponse};
        let (mut client, server) = tokio::net::UnixStream::pair().unwrap();

        let mut handlers: HashMap<String, Arc<dyn IpcHandler>> = HashMap::new();
        handlers.insert("echo".into(), Arc::new(EchoHandler));
        tokio::spawn(async move {
            let _ = IpcServer::handle_connection(server, handlers).await;
        });

        // 发送一个版本不兼容的请求
        let mut req = JsonRpcRequest::new("echo", "1", Some(json!({})));
        req.version = Some("999".into());
        write_message(&mut client, &req).await.unwrap();

        let resp: JsonRpcResponse = read_message(&mut client).await.unwrap();
        let err = resp.error.expect("expected error for incompatible version");
        assert_eq!(err.code, JsonRpcError::INVALID_REQUEST);
        assert!(err.message.contains("incompatible protocol version"));
    }

    #[tokio::test]
    async fn version_handshake_accepts_compatible() {
        use crate::protocol::{read_message, write_message, JsonRpcRequest, JsonRpcResponse};
        let (mut client, server) = tokio::net::UnixStream::pair().unwrap();

        let mut handlers: HashMap<String, Arc<dyn IpcHandler>> = HashMap::new();
        handlers.insert("echo".into(), Arc::new(EchoHandler));
        tokio::spawn(async move {
            let _ = IpcServer::handle_connection(server, handlers).await;
        });

        let req = JsonRpcRequest::new("echo", "1", Some(json!({"ok": true})));
        write_message(&mut client, &req).await.unwrap();

        let resp: JsonRpcResponse = read_message(&mut client).await.unwrap();
        assert!(resp.error.is_none());
        assert_eq!(resp.result.unwrap()["ok"], true);
    }
}
