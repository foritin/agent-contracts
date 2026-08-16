//! IPC 客户端：Unix Socket（macOS）/ Named Pipe（Windows）。
//!
//! 参见 `09-ipc-transport.html §5`。

use crate::protocol::{read_message, write_message, JsonRpcRequest, JsonRpcResponse};
use agent_error::{Error, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::sync::{oneshot, Mutex};

/// 默认调用超时。
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Named Pipe 名称前缀（Windows）。
#[cfg(windows)]
const PIPE_PREFIX: &str = r"\\.\pipe\r-code-";

/// IPC 客户端（跨平台）。
///
/// 内部使用 boxed trait object 持有 read/write 半边，
/// 在 Unix 上是 `OwnedReadHalf`/`OwnedWriteHalf`，
/// 在 Windows 上是 `ReadHalf<NamedPipeClient>`/`WriteHalf<NamedPipeClient>`。
pub struct IpcClient {
    write: Arc<Mutex<Box<dyn AsyncWrite + Unpin + Send>>>,
    pending: Arc<Mutex<HashMap<String, oneshot::Sender<JsonRpcResponse>>>>,
    next_id: Arc<AtomicU64>,
}

impl IpcClient {
    /// 连接到 IPC 服务端。
    ///
    /// - Unix: 连接 Unix Socket 文件。
    /// - Windows: 连接 Named Pipe（路径自动转换为 pipe 名称）。
    ///
    /// Windows 下 pipe 实例在服务端 `serve()` 中才创建，客户端在服务端刚启动时
    /// 连接会遇到短暂的 NotFound/Busy —— 这里统一做有限重试（30 × 100ms）。
    pub async fn connect(path: &Path) -> Result<Self> {
        const MAX_ATTEMPTS: u32 = 30;
        let mut last_err = None;
        for attempt in 0..MAX_ATTEMPTS {
            match Self::connect_once(path).await {
                Ok(client) => return Ok(client),
                Err(e) => {
                    last_err = Some(e);
                    if attempt + 1 < MAX_ATTEMPTS {
                        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    }
                }
            }
        }
        Err(last_err.unwrap_or_else(|| Error::Ipc("connect failed".into())))
    }

    /// 单次连接尝试（平台实现）。
    async fn connect_once(path: &Path) -> Result<Self> {
        #[cfg(unix)]
        {
            let stream = tokio::net::UnixStream::connect(path)
                .await
                .map_err(|e| Error::Ipc(format!("connect failed: {e}")))?;
            let (read, write) = stream.into_split();

            Ok(Self::from_halves(Box::new(read), Box::new(write)))
        }

        #[cfg(windows)]
        {
            let pipe_name = if path.to_string_lossy().starts_with(PIPE_PREFIX) {
                path.to_string_lossy().to_string()
            } else {
                format!(
                    "{}{}",
                    PIPE_PREFIX,
                    path.file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("default")
                )
            };

            let stream = tokio::net::windows::named_pipe::ClientOptions::new()
                .open(&pipe_name)
                .map_err(|e| Error::Ipc(format!("connect pipe failed: {e}")))?;

            let (read, write) = tokio::io::split(stream);

            Ok(Self::from_halves(Box::new(read), Box::new(write)))
        }

        #[cfg(not(any(unix, windows)))]
        {
            let _ = path;
            Err(Error::Ipc("unsupported platform".into()))
        }
    }

    /// 从已拆分的 read/write 半边构造客户端。
    fn from_halves(
        read: Box<dyn AsyncRead + Unpin + Send>,
        write: Box<dyn AsyncWrite + Unpin + Send>,
    ) -> Self {
        let pending: Arc<Mutex<HashMap<String, oneshot::Sender<JsonRpcResponse>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let pending_clone = pending.clone();

        tokio::spawn(async move {
            Self::recv_loop(read, pending_clone).await;
        });

        Self {
            write: Arc::new(Mutex::new(write)),
            pending,
            next_id: Arc::new(AtomicU64::new(1)),
        }
    }

    /// 调用远程方法，等待响应。
    pub async fn call(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst).to_string();
        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(id.clone(), tx);

        let req = JsonRpcRequest::new(method, id.clone(), Some(params));
        {
            let mut w = self.write.lock().await;
            write_message(&mut *w, &req).await?;
        }

        match tokio::time::timeout(DEFAULT_TIMEOUT, rx).await {
            Ok(Ok(resp)) => {
                if let Some(err) = resp.error {
                    Err(Error::Ipc(format!("[{}] {}", err.code, err.message)))
                } else {
                    Ok(resp.result.unwrap_or(Value::Null))
                }
            }
            Ok(Err(_)) => {
                self.pending.lock().await.remove(&id);
                Err(Error::Ipc("response channel closed".into()))
            }
            Err(_) => {
                self.pending.lock().await.remove(&id);
                Err(Error::IpcTimeout)
            }
        }
    }

    async fn recv_loop(
        mut read: Box<dyn AsyncRead + Unpin + Send>,
        pending: Arc<Mutex<HashMap<String, oneshot::Sender<JsonRpcResponse>>>>,
    ) {
        loop {
            match read_message::<_, JsonRpcResponse>(&mut read).await {
                Ok(resp) => {
                    if let Some(id) = resp.id.clone() {
                        if let Some(tx) = pending.lock().await.remove(&id) {
                            let _ = tx.send(resp);
                        }
                    }
                }
                Err(Error::Io(ref e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(_) => break,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn timeout_when_no_server_response() {
        // 连接到一个不响应的服务端 -> 超时
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("silent.sock");

        #[cfg(unix)]
        {
            let lst = tokio::net::UnixListener::bind(&socket).unwrap();
            let path = socket.clone();
            tokio::spawn(async move {
                let _ = lst.accept().await;
                tokio::time::sleep(Duration::from_secs(5)).await;
            });

            let client = IpcClient::connect(&path).await.unwrap();
            let r = tokio::time::timeout(Duration::from_secs(2), client.call("noop", Value::Null))
                .await;
            assert!(
                r.is_err(),
                "expected outer timeout while waiting for 30s ipc timeout"
            );
        }

        #[cfg(windows)]
        {
            // Windows 上类似测试：创建 pipe 但不响应
            use tokio::net::windows::named_pipe::ServerOptions;
            let pipe_name = format!(r"\\.\pipe\r-code-{}", "silent.sock");
            let server = ServerOptions::new()
                .first_pipe_instance(true)
                .create(&pipe_name)
                .unwrap();

            tokio::spawn(async move {
                let _ = server.connect().await;
                tokio::time::sleep(Duration::from_secs(5)).await;
            });

            tokio::time::sleep(Duration::from_millis(100)).await;
            let client = IpcClient::connect(&socket).await.unwrap();
            let r = tokio::time::timeout(Duration::from_secs(2), client.call("noop", Value::Null))
                .await;
            assert!(
                r.is_err(),
                "expected outer timeout while waiting for 30s ipc timeout"
            );
        }
    }
}
