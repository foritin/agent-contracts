//! JSON-RPC 传输层。
//!
//! 参见 `05-mcp-client.html §4`。定义 `Transport` trait，提供 stdio 子进程、
//! Streamable HTTP 与（测试用）Mock 三种实现。`McpServer` 通过该 trait
//! 解耦协议逻辑与传输细节，便于无网络测试。

use crate::error::McpError;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Mutex;

/// JSON-RPC 2.0 请求。
#[derive(Debug, Clone, Serialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: u64,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

/// JSON-RPC 2.0 响应。
#[derive(Debug, Clone, Deserialize)]
pub struct JsonRpcResponse {
    pub id: Option<u64>,
    pub result: Option<Value>,
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
}

/// 传输抽象：发送一个 JSON-RPC 请求并等待对应响应。
#[async_trait]
pub trait Transport: Send + Sync {
    async fn request(&self, method: &str, params: Option<Value>) -> Result<Value, McpError>;
}

// ── MockTransport（测试用）─────────────────────────────────────

/// 按方法名脚本化返回结果的 Mock 传输。
pub struct MockTransport {
    responses: Mutex<HashMap<String, Value>>,
    errors: Mutex<HashMap<String, McpError>>,
    call_count: AtomicU64,
}

impl MockTransport {
    pub fn new() -> Self {
        Self {
            responses: Mutex::new(HashMap::new()),
            errors: Mutex::new(HashMap::new()),
            call_count: AtomicU64::new(0),
        }
    }

    /// 为某方法设置成功返回结果（同时清除该方法已设的错误）。
    pub async fn set_result(&self, method: &str, result: Value) {
        self.responses
            .lock()
            .await
            .insert(method.to_string(), result);
        self.errors.lock().await.remove(method);
    }

    /// 为某方法设置错误。
    pub async fn set_error(&self, method: &str, err: McpError) {
        self.errors.lock().await.insert(method.to_string(), err);
    }

    pub async fn calls(&self) -> u64 {
        self.call_count.load(Ordering::SeqCst)
    }
}

impl Default for MockTransport {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Transport for MockTransport {
    async fn request(&self, method: &str, _params: Option<Value>) -> Result<Value, McpError> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        if let Some(err) = self.errors.lock().await.get(method) {
            return Err(err.clone());
        }
        self.responses
            .lock()
            .await
            .get(method)
            .cloned()
            .ok_or_else(|| McpError::CallFailed(format!("no mock for method {method}")))
    }
}

// ── StdioTransport ────────────────────────────────────────────

/// stdio 子进程传输：向子进程 stdin 写入 JSON-RPC 行，从 stdout 读取响应。
pub struct StdioTransport {
    stdin: Arc<Mutex<tokio::process::ChildStdin>>,
    stdout: Arc<Mutex<BufReader<tokio::process::ChildStdout>>>,
    next_id: AtomicU64,
}

impl StdioTransport {
    /// 启动子进程并建立传输。
    pub async fn spawn(
        command: &str,
        args: &[String],
        env: &HashMap<String, String>,
    ) -> Result<Self, McpError> {
        let mut cmd = tokio::process::Command::new(command);
        cmd.args(args);
        for (k, v) in env {
            cmd.env(k, v);
        }
        cmd.stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null());

        let mut child = cmd
            .spawn()
            .map_err(|e| McpError::Transport(format!("failed to spawn {command}: {e}")))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| McpError::Transport("child has no stdin".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| McpError::Transport("child has no stdout".into()))?;

        // 丢弃 child 句柄但保持进程运行（生产实现应管理生命周期）
        std::mem::forget(child);

        Ok(Self {
            stdin: Arc::new(Mutex::new(stdin)),
            stdout: Arc::new(Mutex::new(BufReader::new(stdout))),
            next_id: AtomicU64::new(1),
        })
    }
}

#[async_trait]
impl Transport for StdioTransport {
    async fn request(&self, method: &str, params: Option<Value>) -> Result<Value, McpError> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id,
            method: method.into(),
            params,
        };
        let line = serde_json::to_string(&req)? + "\n";

        // 写入 stdin
        {
            let mut stdin = self.stdin.lock().await;
            stdin.write_all(line.as_bytes()).await.map_err(|e| {
                // 子进程已退出 -> 可诊断错误（V-TOOL-03）
                McpError::NotConnected(format!("stdio write failed (process exited?): {e}"))
            })?;
            stdin.flush().await?;
        }

        // 读取响应，跳过无 id 的通知
        let mut stdout = self.stdout.lock().await;
        loop {
            let mut buf = String::new();
            let n = stdout
                .read_line(&mut buf)
                .await
                .map_err(|e| McpError::Transport(format!("read failed: {e}")))?;
            if n == 0 {
                return Err(McpError::NotConnected("stdio process exited (EOF)".into()));
            }
            let trimmed = buf.trim();
            if trimmed.is_empty() {
                continue;
            }
            let resp: JsonRpcResponse = serde_json::from_str(trimmed)?;
            if resp.id != Some(id) {
                // 通知或他人响应，跳过
                continue;
            }
            if let Some(err) = resp.error {
                return Err(McpError::CallFailed(format!(
                    "[{}] {}",
                    err.code, err.message
                )));
            }
            return Ok(resp.result.unwrap_or(Value::Null));
        }
    }
}

// ── HttpTransport ─────────────────────────────────────────────

/// Streamable HTTP 传输：POST JSON-RPC 到指定 URL。
pub struct HttpTransport {
    url: String,
    client: reqwest::Client,
    headers: HashMap<String, String>,
    next_id: AtomicU64,
}

impl HttpTransport {
    /// 默认构造（30s 超时）。
    pub fn new(url: String, headers: HashMap<String, String>) -> Result<Self, McpError> {
        Self::with_timeout(url, headers, std::time::Duration::from_secs(30))
    }

    /// 指定超时（accept-perf-mcp：超时可配置）。
    pub fn with_timeout(
        url: String,
        headers: HashMap<String, String>,
        timeout: std::time::Duration,
    ) -> Result<Self, McpError> {
        Ok(Self {
            url,
            client: reqwest::Client::builder()
                .timeout(timeout)
                .build()
                .map_err(|e| McpError::Transport(e.to_string()))?,
            headers,
            next_id: AtomicU64::new(1),
        })
    }
}

#[async_trait]
impl Transport for HttpTransport {
    async fn request(&self, method: &str, params: Option<Value>) -> Result<Value, McpError> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let body = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params.unwrap_or(Value::Null),
        });

        let mut req = self.client.post(&self.url).json(&body);
        for (k, v) in &self.headers {
            req = req.header(k, v);
        }

        let resp = req
            .send()
            .await
            .map_err(|e| McpError::Transport(e.to_string()))?;

        let status = resp.status();
        if !status.is_success() {
            // V-TOOL-04：非 2xx 映射为稳定错误类别
            return Err(McpError::CallFailed(format!("HTTP {}", status.as_u16())));
        }

        let v: Value = resp
            .json()
            .await
            .map_err(|_| McpError::Parse("invalid JSON in HTTP response".into()))?;

        if let Some(err) = v.get("error") {
            let code = err.get("code").and_then(|c| c.as_i64()).unwrap_or(-1);
            let msg = err
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown");
            return Err(McpError::CallFailed(format!("[{code}] {msg}")));
        }
        Ok(v.get("result").cloned().unwrap_or(Value::Null))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_returns_scripted_result() {
        let t = MockTransport::new();
        t.set_result("ping", json!({"pong": true})).await;
        let r = t.request("ping", None).await.unwrap();
        assert_eq!(r["pong"], true);
    }

    #[tokio::test]
    async fn mock_returns_scripted_error() {
        let t = MockTransport::new();
        t.set_error("boom", McpError::Timeout).await;
        let err = t.request("boom", None).await.unwrap_err();
        assert!(matches!(err, McpError::Timeout));
    }

    #[tokio::test]
    async fn stdio_process_exit_returns_diagnostic_error() {
        // V-TOOL-03 / test-integ-mcpstdio：stdio 子进程退出后，调用返回可诊断错误
        // 进程读一行后立即退出，不响应 -> 读到 EOF -> NotConnected
        let env = HashMap::new();
        let (program, args) = if cfg!(windows) {
            ("cmd", vec!["/C".into(), "set /p x=& exit /b 0".into()])
        } else {
            ("sh", vec!["-c".into(), "read x; exit 0".into()])
        };
        let transport = StdioTransport::spawn(program, &args, &env)
            .await
            .expect("spawn test process");

        let err = transport
            .request("initialize", None)
            .await
            .expect_err("expected diagnostic error after process exit");
        // 可诊断：错误信息提及进程退出 / EOF / 写失败
        let msg = err.to_string();
        assert!(
            msg.contains("exited") || msg.contains("EOF") || msg.contains("failed"),
            "not a diagnostic exit error: {msg}"
        );
    }

    /// 启动一个本地 HTTP 服务，对单次连接写入给定响应字节。
    async fn spawn_http_server(response: Vec<u8>) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            if let Ok((mut sock, _)) = listener.accept().await {
                use tokio::io::AsyncWriteExt;
                let _ = sock.write_all(&response).await;
                let _ = sock.flush().await;
                // 保持连接片刻让 reqwest 读取
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        });
        format!("http://{}", addr)
    }

    #[tokio::test]
    async fn http_non_2xx_maps_to_call_failed() {
        // V-TOOL-04 / test-integ-mcphttp：非 2xx 映射为稳定错误类别
        let url = spawn_http_server(
            b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n".to_vec(),
        )
        .await;
        let t = HttpTransport::new(url, HashMap::new()).unwrap();
        let err = t.request("ping", None).await.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("HTTP 500") || msg.contains("500"),
            "got: {msg}"
        );
    }

    #[tokio::test]
    async fn http_invalid_json_maps_to_parse_error() {
        // V-TOOL-04：非法 JSON 映射为稳定错误类别
        let body = b"not-json-at-all";
        let resp = format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n", body.len());
        let mut bytes = resp.into_bytes();
        bytes.extend_from_slice(body);
        let url = spawn_http_server(bytes).await;
        let t = HttpTransport::new(url, HashMap::new()).unwrap();
        let err = t.request("ping", None).await.unwrap_err();
        assert!(matches!(err, McpError::Parse(_)), "got: {err}");
    }

    #[tokio::test]
    async fn http_timeout_is_configurable() {
        // accept-perf-mcp：超时可配置；服务端不响应 -> 客户端超时
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            // 接受连接但永不响应
            let _ = listener.accept().await;
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        });
        let url = format!("http://{}", addr);
        // 200ms 超时
        let t =
            HttpTransport::with_timeout(url, HashMap::new(), std::time::Duration::from_millis(200))
                .unwrap();
        let start = std::time::Instant::now();
        let err = t.request("ping", None).await.unwrap_err();
        let elapsed = start.elapsed();
        // 应在 ~200ms 内失败（而非 30s 默认）
        assert!(
            elapsed < std::time::Duration::from_secs(2),
            "took too long: {elapsed:?}"
        );
        assert!(matches!(err, McpError::Transport(_)), "got: {err}");
    }
}
