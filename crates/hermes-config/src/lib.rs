//! `hermes-config` -- 配置管理。
//!
//! 参见 `08-config-management.html`。TOML 配置 + 环境变量覆盖 + 默认值 + 验证。
//! 敏感值（api_key）绝不出现在 Debug 输出中（V-CFG-02）。

use hermes_error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};

/// 顶层配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_provider")]
    pub default_provider: String,

    #[serde(default = "default_log_level")]
    pub log_level: String,

    #[serde(default)]
    pub providers: HashMap<String, ProviderConfig>,

    #[serde(default)]
    pub mcp_servers: HashMap<String, ServerSpec>,

    #[serde(default = "default_storage")]
    pub storage: StorageConfig,

    #[serde(default = "default_compaction")]
    pub compaction: CompactionConfig,

    /// 主 Agent、跨引擎委派与结果复核策略。
    #[serde(default)]
    pub orchestration: OrchestrationConfig,

    #[serde(default)]
    pub tauri: Option<TauriConfig>,
}

/// 单个 LLM Provider 配置。api_key 在 Debug 中脱敏。
#[derive(Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// 线路协议（wire protocol）：`anthropic_messages` / `openai_chat` /
    /// `openai_responses`。决定请求体形状、SSE 事件格式和鉴权头。
    ///
    /// **这是用户显式选择的值，不是推导出来的。** 同一个 base_url 常常同时支持多种
    /// 协议（火山方舟 `/api/coding` 是 Anthropic 口、`/api/coding/v3` 是 OpenAI 口），
    /// 计费和能力都不同，只能由用户决定走哪个。
    ///
    /// `None` = 尚未选择（升级前保存的旧配置）。此时由调用方决定回退策略——
    /// R-Code 的做法见 `commands.rs::build_provider_config`：按目录推断，但绝不
    /// 自动选中 Responses。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protocol: Option<String>,
}

impl fmt::Debug for ProviderConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProviderConfig")
            .field("base_url", &self.base_url)
            .field("api_key", &"***")
            .field("model", &self.model)
            .field("max_tokens", &self.max_tokens)
            .field("temperature", &self.temperature)
            .field("protocol", &self.protocol)
            .finish()
    }
}

/// 存储路径配置。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StorageConfig {
    pub base_dir: PathBuf,
    pub sessions_dir: PathBuf,
    pub skills_dir: PathBuf,
    pub memories_dir: PathBuf,
}

/// 压缩配置。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompactionConfig {
    #[serde(default = "default_compaction_strategy")]
    pub strategy: String,

    #[serde(default = "default_max_context")]
    pub max_context_tokens: u32,

    #[serde(default = "default_trigger")]
    pub trigger_threshold: f64,
}

/// Tauri 外壳配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TauriConfig {
    #[serde(default = "default_theme")]
    pub theme: String,

    #[serde(default = "default_font_size")]
    pub font_size: u32,
}

/// Agent 编排设置。所有自动决策都必须在运行事件中公开策略结论。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestrationConfig {
    /// 新会话默认由哪一个主 Agent 执行。
    #[serde(default)]
    pub default_agent_engine: MainAgentEngine,
    /// `delegate_task(agent="auto")` 的路由策略。
    #[serde(default)]
    pub delegation_router: DelegationRouterMode,
    /// 是否允许两个 Agent 引擎互相委派。关闭后只允许同引擎子智能体。
    #[serde(default = "default_true")]
    pub allow_cross_engine_delegation: bool,
    /// 主回复完成后的显式质量复核策略。
    #[serde(default)]
    pub quality_loop: QualityLoopMode,
    /// 执行质量复核的引擎。
    #[serde(default)]
    pub quality_reviewer: QualityReviewer,
    /// 最多复核/修订轮数；产品层限制为 1..=3。
    #[serde(default = "default_review_rounds")]
    pub max_review_rounds: u8,
}

impl Default for OrchestrationConfig {
    fn default() -> Self {
        Self {
            default_agent_engine: MainAgentEngine::RCode,
            delegation_router: DelegationRouterMode::Balanced,
            allow_cross_engine_delegation: true,
            quality_loop: QualityLoopMode::Auto,
            quality_reviewer: QualityReviewer::Auto,
            max_review_rounds: default_review_rounds(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MainAgentEngine {
    #[default]
    RCode,
    Codex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DelegationRouterMode {
    /// 只有模型显式选择某个执行器时才路由；auto 回退当前 R-Code 引擎。
    Manual,
    /// 简单任务使用 R-Code，复杂任务优先 Codex；Codex 不可用时安全回退。
    #[default]
    Balanced,
    /// 除非明确标注复杂，否则优先 R-Code。
    RCodeFirst,
    /// 除非明确标注简单，否则优先 Codex。
    CodexFirst,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum QualityLoopMode {
    Off,
    /// 仅工具型/工作区任务完成后复核。
    #[default]
    Auto,
    /// 每一轮主回复都复核。
    Always,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum QualityReviewer {
    /// 与主执行器交叉复核；不可用时回退 R-Code。
    #[default]
    Auto,
    RCode,
    Codex,
}

/// MCP server 规格（config 自有版本，不依赖 hermes-mcp）。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ServerSpec {
    Stdio {
        command: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        env: HashMap<String, String>,
    },
    Http {
        url: String,
        #[serde(default)]
        headers: HashMap<String, String>,
    },
}

// ── 默认值函数 ────────────────────────────────────────────────

fn default_provider() -> String {
    "anthropic".into()
}
fn default_log_level() -> String {
    "info".into()
}
fn default_compaction_strategy() -> String {
    "auto".into()
}
fn default_max_context() -> u32 {
    180_000
}
fn default_trigger() -> f64 {
    0.8
}
fn default_theme() -> String {
    "system".into()
}
fn default_font_size() -> u32 {
    13
}
fn default_true() -> bool {
    true
}
fn default_review_rounds() -> u8 {
    1
}

fn default_storage() -> StorageConfig {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    let base = home.join(".hermes");
    StorageConfig {
        base_dir: base.join("data"),
        sessions_dir: base.join("sessions"),
        skills_dir: base.join("skills"),
        memories_dir: base.join("memories"),
    }
}

fn default_compaction() -> CompactionConfig {
    CompactionConfig {
        strategy: default_compaction_strategy(),
        max_context_tokens: default_max_context(),
        trigger_threshold: default_trigger(),
    }
}

impl Default for Config {
    fn default() -> Self {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        let base = home.join(".hermes");
        Self {
            default_provider: default_provider(),
            log_level: default_log_level(),
            providers: HashMap::new(),
            mcp_servers: HashMap::new(),
            storage: StorageConfig {
                base_dir: base.join("data"),
                sessions_dir: base.join("sessions"),
                skills_dir: base.join("skills"),
                memories_dir: base.join("memories"),
            },
            compaction: CompactionConfig {
                strategy: default_compaction_strategy(),
                max_context_tokens: default_max_context(),
                trigger_threshold: default_trigger(),
            },
            orchestration: OrchestrationConfig::default(),
            tauri: None,
        }
    }
}

impl Config {
    /// 加载配置（优先级：默认值 < 文件 < 环境变量）。
    pub fn load() -> Result<Self> {
        let path = Self::config_path();
        if path.exists() {
            Self::load_from(&path)
        } else {
            let mut config = Self::default();
            Self::apply_env(&mut config);
            config.validate()?;
            Ok(config)
        }
    }

    /// 从指定路径加载。
    pub fn load_from(path: &Path) -> Result<Self> {
        let content =
            std::fs::read_to_string(path).map_err(|_| Error::ConfigNotFound(path.to_path_buf()))?;
        let mut config: Config = toml::from_str(&content).map_err(Error::Toml)?;
        Self::apply_env(&mut config);
        config.validate()?;
        Ok(config)
    }

    /// 环境变量覆盖（ANTHROPIC_API_KEY 等）。
    fn apply_env(config: &mut Config) {
        if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
            config
                .providers
                .entry("anthropic".into())
                .and_modify(|p| p.api_key = key.clone());
        }
        if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            config
                .providers
                .entry("openai".into())
                .and_modify(|p| p.api_key = key.clone());
        }
    }

    /// 校验。
    pub fn validate(&self) -> Result<()> {
        if !(1..=3).contains(&self.orchestration.max_review_rounds) {
            return Err(Error::Config(
                "orchestration.max_review_rounds must be between 1 and 3".to_string(),
            ));
        }
        if !self.providers.contains_key(&self.default_provider) {
            return Err(Error::Config(format!(
                "default provider '{}' not configured",
                self.default_provider
            )));
        }
        for (name, provider) in &self.providers {
            if provider.api_key.is_empty() {
                return Err(Error::Config(format!(
                    "provider '{}' has empty api_key",
                    name
                )));
            }
        }
        // 确保存储目录可写（不存在则创建）
        if !self.storage.base_dir.exists() {
            std::fs::create_dir_all(&self.storage.base_dir)?;
        }
        Ok(())
    }

    fn config_path() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".hermes/config.toml")
    }

    /// 获取指定 provider 配置。
    pub fn provider(&self, name: &str) -> Result<&ProviderConfig> {
        self.providers
            .get(name)
            .ok_or_else(|| Error::ProviderNotFound(name.to_string()))
    }

    /// 获取默认 provider 配置。
    pub fn default_provider_config(&self) -> Result<&ProviderConfig> {
        self.provider(&self.default_provider)
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::Mutex;

    // 环境变量是进程级全局状态，多个测试并行操作会竞态；用锁串行化。
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn write_temp_config(content: &str) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        (dir, path)
    }

    #[test]
    fn toml_roundtrip() {
        let toml_str = r#"
default_provider = "anthropic"
log_level = "debug"

[providers.anthropic]
base_url = "https://api.anthropic.com"
api_key = "sk-ant-xxx"
model = "claude-sonnet-4"

[storage]
base_dir = "/tmp/h"
sessions_dir = "/tmp/h/s"
skills_dir = "/tmp/h/k"
memories_dir = "/tmp/h/m"
"#;
        let (_d, path) = write_temp_config(toml_str);
        let config = Config::load_from(&path).unwrap();
        assert_eq!(config.log_level, "debug");
        assert_eq!(
            config.providers.get("anthropic").unwrap().model,
            "claude-sonnet-4"
        );
    }

    #[test]
    fn v_cfg_01_env_overrides_file() {
        // V-CFG-01：默认值 < 文件 < 环境变量
        let _guard = ENV_LOCK.lock().unwrap();
        let toml_str = r#"
default_provider = "anthropic"

[providers.anthropic]
base_url = "https://api.anthropic.com"
api_key = "from_file"
model = "claude-sonnet-4"

[storage]
base_dir = "/tmp/h"
sessions_dir = "/tmp/h/s"
skills_dir = "/tmp/h/k"
memories_dir = "/tmp/h/m"
"#;
        let (_d, path) = write_temp_config(toml_str);

        std::env::set_var("ANTHROPIC_API_KEY", "from_env");
        let config = Config::load_from(&path).unwrap();
        std::env::remove_var("ANTHROPIC_API_KEY");

        assert_eq!(
            config.providers.get("anthropic").unwrap().api_key,
            "from_env"
        );
    }

    #[test]
    fn v_cfg_01_file_overrides_default_when_no_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        let toml_str = r#"
default_provider = "anthropic"

[providers.anthropic]
base_url = "https://api.anthropic.com"
api_key = "file_value"
model = "claude-sonnet-4"

[storage]
base_dir = "/tmp/h"
sessions_dir = "/tmp/h/s"
skills_dir = "/tmp/h/k"
memories_dir = "/tmp/h/m"
"#;
        let (_d, path) = write_temp_config(toml_str);
        std::env::remove_var("ANTHROPIC_API_KEY");
        let config = Config::load_from(&path).unwrap();
        assert_eq!(
            config.providers.get("anthropic").unwrap().api_key,
            "file_value"
        );
    }

    #[test]
    fn v_cfg_02_debug_does_not_leak_api_key() {
        // V-CFG-02：Debug 输出不含 api_key
        let config = ProviderConfig {
            base_url: "https://api.anthropic.com".into(),
            api_key: "sk-secret-12345".into(),
            model: "claude".into(),
            max_tokens: None,
            temperature: None,
            protocol: None,
        };
        let dbg = format!("{:?}", config);
        assert!(!dbg.contains("sk-secret-12345"), "api_key leaked: {dbg}");
        assert!(dbg.contains("***"));
    }

    #[test]
    fn validate_rejects_missing_default_provider() {
        let mut config = Config::default();
        config.default_provider = "nonexistent".into();
        config.providers.insert(
            "other".into(),
            ProviderConfig {
                base_url: "x".into(),
                api_key: "k".into(),
                model: "m".into(),
                max_tokens: None,
                temperature: None,
                protocol: None,
            },
        );
        let r = config.validate();
        assert!(r.is_err());
    }

    #[test]
    fn validate_rejects_empty_api_key() {
        let mut config = Config::default();
        config.default_provider = "anthropic".into();
        config.providers.insert(
            "anthropic".into(),
            ProviderConfig {
                base_url: "x".into(),
                api_key: "".into(),
                model: "m".into(),
                max_tokens: None,
                temperature: None,
                protocol: None,
            },
        );
        let r = config.validate();
        assert!(r.is_err());
    }

    #[test]
    fn unknown_field_ignored() {
        let toml_str = r#"
default_provider = "anthropic"
future_unknown_field = 123

[providers.anthropic]
base_url = "x"
api_key = "k"
model = "m"

[storage]
base_dir = "/tmp/h"
sessions_dir = "/tmp/h/s"
skills_dir = "/tmp/h/k"
memories_dir = "/tmp/h/m"
"#;
        let (_d, path) = write_temp_config(toml_str);
        let config = Config::load_from(&path).unwrap();
        assert_eq!(config.providers.len(), 1);
    }

    #[test]
    fn default_values_applied() {
        let config = Config::default();
        assert_eq!(config.default_provider, "anthropic");
        assert_eq!(config.log_level, "info");
        assert_eq!(config.compaction.max_context_tokens, 180_000);
        assert!((config.compaction.trigger_threshold - 0.8).abs() < 1e-9);
    }
}
