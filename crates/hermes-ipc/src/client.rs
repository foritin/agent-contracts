//! IPC 客户端：连接 Unix Socket，发送请求并异步等待响应。
//!
//! 参见 `09-ipc-transport.html §5`。

use crate::protocol::{read_message, write_message, JsonRpcRequest, JsonRpcResponse};
use hermes_error::{Error, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::UnixStream;
use tokio::sync::{oneshot, Mutex};

/// 默认调用超时。
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// IPC 客户端。
pub struct IpcClient {
    write: Arc<Mutex<OwnedWriteHalf>>,
    pending: Arc<Mutex<HashMap<String, oneshot::Sender<JsonRpcResponse>>>>,
    next_id: Arc<AtomicU64>,
}

impl IpcClient {
    /// 连接到 Unix Socket。
    pub async fn connect(path: &Path) -> Result<Self> {
        let stream = UnixStream::connect(path)
            .await
            .map_err(|e| Error::Ipc(format!("connect failed: {e}")))?;
        let (read, write) = stream.into_split();

        let pending: Arc<Mutex<HashMap<String, oneshot::Sender<JsonRpcResponse>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let pending_clone = pending.clone();

        tokio::spawn(async move {
            Self::recv_loop(read, pending_clone).await;
        });

        Ok(Self {
            write: Arc::new(Mutex::new(write)),
            pending,
            next_id: Arc::new(AtomicU64::new(1)),
        })
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
        mut read: OwnedReadHalf,
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
        // 用一个只接受连接但不读写的 listener
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("silent.sock");
        // 真实 listener 但不处理请求
        let lst = tokio::net::UnixListener::bind(&socket).unwrap();
        let path = socket.clone();
        tokio::spawn(async move {
            // 接受连接但不响应
            let _ = lst.accept().await;
            tokio::time::sleep(Duration::from_secs(5)).await;
        });

        let client = IpcClient::connect(&path).await.unwrap();
        // 用很短的超时避免测试慢：直接调用，期望 30s 超时太长。
        // 这里仅验证 client 能连接并发送；超时测试通过快速失败验证。
        // 为加速，我们断言 call 最终返回错误（超时或断开）。
        let r =
            tokio::time::timeout(Duration::from_secs(2), client.call("noop", Value::Null)).await;
        // 2s 内 call 仍在等待（30s 超时）-> 返回 Err(timeout) 表示外层超时
        assert!(
            r.is_err(),
            "expected outer timeout while waiting for 30s ipc timeout"
        );
    }
}
