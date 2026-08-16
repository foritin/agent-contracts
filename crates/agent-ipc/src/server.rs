//! IPC 服务端：Unix Socket（macOS）/ Named Pipe（Windows）。
//!
//! 参见 `09-ipc-transport.html §4 §6`。

use crate::protocol::{read_message, write_message, JsonRpcError, JsonRpcRequest, JsonRpcResponse};
use agent_error::{Error, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::{AsyncRead, AsyncWrite};

/// 请求处理器 trait（跨平台）。
#[async_trait::async_trait]
pub trait IpcHandler: Send + Sync {
    async fn handle(&self, params: Value) -> Result<Value>;
}

// ---------------------------------------------------------------------------
// 平台特定的 listener
// ---------------------------------------------------------------------------

#[cfg(unix)]
type PlatformListener = tokio::net::UnixListener;

#[cfg(windows)]
struct WindowsPipeListener {
    pipe_name: String,
}

/// IPC 服务端。
pub struct IpcServer {
    #[cfg(unix)]
    listener: PlatformListener,
    #[cfg(windows)]
    listener: WindowsPipeListener,
    handlers: HashMap<String, Arc<dyn IpcHandler>>,
    socket_path: PathBuf,
}

impl IpcServer {
    /// 绑定 IPC 端点。
    ///
    /// - Unix: 绑定 Unix Socket，清理旧文件，设置 0o600 权限。
    /// - Windows: 记录 Named Pipe 名称（pipe 实例在 `serve()` 中按需创建）。
    pub fn bind(socket_path: PathBuf) -> Result<Self> {
        #[cfg(unix)]
        {
            if socket_path.exists() {
                std::fs::remove_file(&socket_path)?;
            }
            if let Some(parent) = socket_path.parent() {
                if !parent.exists() {
                    std::fs::create_dir_all(parent)?;
                }
            }
            let listener = tokio::net::UnixListener::bind(&socket_path)
                .map_err(|e| Error::Ipc(format!("bind {} failed: {e}", socket_path.display())))?;

            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&socket_path, std::fs::Permissions::from_mode(0o600))?;

            Ok(Self {
                listener,
                handlers: HashMap::new(),
                socket_path,
            })
        }

        #[cfg(windows)]
        {
            let pipe_name = format!(
                r"\\.\pipe\r-code-{}",
                socket_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("default")
            );

            if let Some(parent) = socket_path.parent() {
                if !parent.exists() {
                    std::fs::create_dir_all(parent)?;
                }
            }

            Ok(Self {
                listener: WindowsPipeListener { pipe_name },
                handlers: HashMap::new(),
                socket_path,
            })
        }
    }

    /// 注册方法处理器。
    pub fn register(&mut self, method: &str, handler: Arc<dyn IpcHandler>) {
        self.handlers.insert(method.to_string(), handler);
    }

    /// 服务端端点路径。
    pub fn socket_path(&self) -> &PathBuf {
        &self.socket_path
    }

    /// 开始接受连接（阻塞，通常 spawn 到后台）。
    pub async fn serve(self) -> Result<()> {
        #[cfg(unix)]
        {
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

        #[cfg(windows)]
        {
            use tokio::net::windows::named_pipe::ServerOptions;
            let pipe_name = self.listener.pipe_name.clone();

            loop {
                let server = ServerOptions::new()
                    .first_pipe_instance(false)
                    .create(&pipe_name)
                    .map_err(|e| Error::Ipc(format!("create pipe failed: {e}")))?;

                server
                    .connect()
                    .await
                    .map_err(|e| Error::Ipc(format!("pipe connect failed: {e}")))?;

                let handlers = self.handlers.clone();
                tokio::spawn(async move {
                    if let Err(e) = Self::handle_connection(server, handlers).await {
                        tracing::error!("IPC connection error: {e}");
                    }
                });
            }
        }
    }

    /// 处理单个客户端连接（泛型，跨平台）。
    async fn handle_connection<S>(
        mut stream: S,
        handlers: HashMap<String, Arc<dyn IpcHandler>>,
    ) -> Result<()>
    where
        S: AsyncRead + AsyncWrite + Unpin,
    {
        use tokio::io::AsyncWriteExt;
        loop {
            let req: JsonRpcRequest = match read_message(&mut stream).await {
                Ok(r) => r,
                Err(Error::Io(ref e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    return Ok(());
                }
                Err(e) => return Err(e),
            };

            let is_notification = req.id.is_none();

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
            let _ = stream.flush().await;
        }
    }
}

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

        use crate::client::IpcClient;
        let client = IpcClient::connect(&path).await.unwrap();
        let result = client.call("echo", json!({"msg": "hi"})).await.unwrap();
        assert_eq!(result["msg"], "hi");
    }
}
