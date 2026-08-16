# agent-contracts

R-Code（桌面 Agentic IDE）与 Tiny Hermes（自进化 AI Agent）共享的公共模块。
以独立 Rust workspace 提供，可被两个产品以 git submodule 形式固定版本引用。

## 模块清单

| Crate | 说明 | 依赖 |
|-------|------|------|
| `agent-error` | 统一错误类型（thiserror） | 无 |
| `agent-contract` | 核心抽象：Message/ContentBlock/Session + Provider/ToolHost/Compaction trait | agent-error |
| `agent-llm` | LLM Provider 实现：Anthropic / OpenAI / DeepSeek / Mock | agent-contract |
| `agent-mcp` | MCP 客户端：stdio + Streamable HTTP，聚合 ToolHost | agent-contract |
| `agent-store` | JSONL 会话持久化 + 崩溃恢复 + 归档 | agent-contract |
| `agent-config` | TOML 配置 + 环境变量覆盖 + 秘密脱敏 | agent-error |
| `agent-compaction` | 上下文压缩：滑动窗口 / LLM 摘要 / 智能选择 | agent-contract |
| `agent-ipc` | JSON-RPC 2.0 over Unix Socket / Named Pipe | agent-contract |
| `agent-tauri` | Tauri 应用壳状态与事件（运行时无关核心） | agent-contract/config/store/mcp |

## 快速验证

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo doc --no-deps --workspace
```

## 作为子模块接入

```bash
# 在 R-Code 或 Tiny Hermes 根目录
git submodule add <agent-contracts-repo-url> vendor/agent-contracts
git submodule update --init --recursive

# 升级时显式审查并固定
git -C vendor/agent-contracts fetch --tags
git -C vendor/agent-contracts checkout <reviewed-tag-or-commit>
git add vendor/agent-contracts .gitmodules
git commit -m "docs: pin agent-contracts contract <version>"
```

在产品根 `Cargo.toml` 中：

```toml
[workspace]
resolver = "2"
members = ["vendor/agent-contracts/crates/*", "crates/*"]

[workspace.dependencies]
agent-contract = { path = "vendor/agent-contracts/crates/agent-contract" }
# ... 其他公共 crate
```

业务 crate 通过 `{ workspace = true }` 引用，公共 crate 不得反向依赖产品私有 crate。

## 设计原则

1. **Trait 优先**：所有抽象通过 Rust trait 定义，实现可替换。
2. **零成本抽象**：泛型 + 静态分发。
3. **错误统一**：`agent_error::Error` 跨模块传播。
4. **异步优先**：Tokio 运行时。
5. **序列化友好**：serde 支持 JSON/TOML。

## 版本与兼容性

- 遵循 Semver。
- `patch`：文档措辞、示例、不改合同。
- `minor`：新增可选字段 / 方法 / 事件。
- `major`：删除/重命名字段、改默认权限、改事件顺序。
- 破坏性变更附迁移页与版本号；两端同时升级。

完整规范见 `docs/` 下 00–15 篇离线 HTML 文档，开发对照清单见 `docs/checklist.html`。

## 迁移记录

每个 `major` 版本破坏性变更在此登记迁移说明。

| 版本 | 变更类型 | 影响 | 迁移说明 |
|------|----------|------|----------|
| 0.1.0 | 初始发布 | — | 首个公共层合同：消息/Provider/ToolHost/MCP/会话/配置/错误/IPC/压缩。 |
| 0.3.0 | 破坏性改名 | crate 名与 import 路径 | `hermes-*` → `agent-*`（`hermes-core` → `agent-contract`），两端同步更新依赖名与 `use` 路径；无类型语义变化。 |

**向前兼容策略**（无需迁移）：
- `ContentBlock` / `SessionEvent` / `ServerSpec` 等使用 tagged enum + 可选字段，旧实现读取含未知字段的数据不崩溃。
- `ToolSpec.requires_confirmation` 默认 `true`，新字段不改变旧确认策略。
- 配置 TOML 缺失字段使用安全默认值。

发生破坏性变更时，新增一行并附：受影响合同编号、旧→新映射、双端升级截止时间。

## 许可

MIT。
