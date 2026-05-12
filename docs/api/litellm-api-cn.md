# LiteLLM API 接口文档（中文）

> 基于 LiteLLM API v1.82.6 的 OpenAPI 规范自动生成
>
> 原始文档：https://litellm-api.up.railway.app/#/

LiteLLM 是一个代理服务器，支持以 OpenAI 格式调用 100+ 个大语言模型。

本文档已按类型拆分为多个子文档，便于分模块查阅。完整内容位于 [`parts/`](./parts/) 目录下。

## 文档索引

| # | 分类 | 子文档 | 主要内容 |
|---|------|--------|----------|
| 01 | 核心 API | [`parts/01-core-api.md`](./parts/01-core-api.md) | 对话补全 / 文本补全 / 嵌入向量 / 内容审核 / 音频 / 图片 / 视频 / OCR / 重排序 / 响应 / 实时 / 搜索 |
| 02 | 助手与代理 | [`parts/02-assistants-agents.md`](./parts/02-assistants-agents.md) | 助手 / 代理 / 代理管理 / [beta] 代理 / [beta] A2A 代理 / [beta] MCP / MCP 模型上下文协议 / Claude Code 市场 |
| 03 | 文件与数据 | [`parts/03-files-data.md`](./parts/03-files-data.md) | 文件管理 / 批处理 / 微调 / 向量存储管理 / 向量存储文件 / RAG / 容器 |
| 04 | 模型与路由 | [`parts/04-models-routing.md`](./parts/04-models-routing.md) | 模型管理 / 路由设置 / 降级管理 / 供应商 / 缓存 / 缓存设置 |
| 05 | 密钥与认证 | [`parts/05-keys-auth.md`](./parts/05-keys-auth.md) | 密钥管理 / 凭据管理 / JWT 密钥映射 / SSO 设置 / SCIM v2（企业版） |
| 06 | 用户与团队管理 | [`parts/06-users-teams.md`](./parts/06-users-teams.md) | 内部用户 / 团队 / 组织 / 项目 / 客户 / 访问组 |
| 07 | 预算与计费 | [`parts/07-budgets-billing.md`](./parts/07-budgets-billing.md) | 预算管理 / 预算与支出跟踪 / 成本跟踪 / 标签管理 |
| 08 | 安全与合规 | [`parts/08-security-compliance.md`](./parts/08-security-compliance.md) | 安全护栏 / 策略 / 策略管理 / 合规 / 审计日志 |
| 09 | 提示词与工具 | [`parts/09-prompts-tools.md`](./parts/09-prompts-tools.md) | 提示词管理 / 提示词 / 工具管理 / 搜索工具 |
| 10 | 系统设置与监控 | [`parts/10-system-settings.md`](./parts/10-system-settings.md) | 健康检查 / 系统设置 / 配置覆盖 / UI 设置 / UI 主题 / 日志回调 / 邮件管理 |
| 11 | 直通代理 | [`parts/11-passthrough.md`](./parts/11-passthrough.md) | 各 Provider 的 Pass-Through 路由（Anthropic / Azure / Bedrock / Cohere / Cursor / Gemini / Langfuse / Mistral / OpenAI / Vertex AI / VLLM / AssemblyAI / Milvus …） |
| 12 | 评估与分析 | [`parts/12-evaluation-analytics.md`](./parts/12-evaluation-analytics.md) | OpenAI 评估 API / 用户代理分析 / CloudZero / Vantage |
| 13 | Beta 功能 | [`parts/13-beta.md`](./parts/13-beta.md) | [beta] Anthropic 事件日志 / Token 计数 / Skills API / `/v1/messages` |
| 14 | 其他 | [`parts/14-others.md`](./parts/14-others.md) | WebSocket / 交互 / 实验性功能 / 公共接口 / LLM 工具 / 工具 / Google GenAI 端点 |
| 15 | 其他接口 | [`parts/15-uncategorized.md`](./parts/15-uncategorized.md) | 未分类路由 |

## 如何使用

- 按分类查阅：直接打开上表对应的子文档。
- 全文搜索：在仓库根目录用 `rg` / 搜索工具对 `docs/api/parts/` 进行正则或关键字搜索。
- 接口定位：若只知道路径（如 `/v1/responses`），可在 `parts/` 目录执行 `grep -rn "/v1/responses" docs/api/parts/`。

## 生成来源

本目录下的内容均来自 LiteLLM 官方 OpenAPI 规范，涵盖 Proxy Server 暴露的全部 HTTP 路由；与代码侧路由一一对应的挂载点可参考 `litellm/proxy/proxy_server.py` 以及各 `management_endpoints/`、`pass_through_endpoints/` 模块。
