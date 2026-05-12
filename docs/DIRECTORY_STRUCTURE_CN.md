# LiteLLM 项目目录结构详解

> 本文档详细说明 LiteLLM 项目的目录结构和各模块功能。
> 最后更新时间：2026-04-10

## 目录总览

```
litellm/
├── litellm/                          # 🔧 核心库源码（主包）
├── litellm-proxy-extras/             # 📦 Proxy 独立部署包（含 Prisma 迁移）
├── enterprise/                       # 🏢 企业版功能（闭源，需 License）
├── ui/litellm-dashboard/             # 🖥️ Admin Dashboard（Next.js）
├── tests/                            # 🧪 测试套件
├── docs/my-website/                  # 📚 文档网站（Docusaurus）
├── cookbook/                         # 📖 使用示例与配方
├── scripts/                          # 🔨 辅助脚本
├── ci_cd/                            # ⚙️ CI/CD 配置
├── docker/                           # 🐳 Docker 部署配置
├── deploy/                           # 🚀 部署相关文件
├── db_scripts/                       # 💾 数据库脚本
├── litellm-js/                       # 📦 JS SDK（消费日志查询）
├── schema.prisma                     # 🗃️ 数据库 Schema（主副本）
├── model_prices_and_context_window.json  # 💰 模型定价与上下文窗口
├── provider_endpoints_support.json   # 📋 Provider 端点支持矩阵
├── pyproject.toml                    # Python 项目配置（Poetry）
├── Dockerfile                        # 主 Dockerfile
├── docker-compose.yml                # Docker Compose 编排
├── Makefile                          # 构建/测试命令
├── ARCHITECTURE.md                   # 英文架构文档
├── ARCHITECTURE_CN.md                # 中文架构文档
├── AGENTS.md                         # AI Agent 协作指南
├── CLAUDE.md                         # Claude Code 开发指南
├── CONTRIBUTING.md                   # 贡献指南
└── README.md                         # 项目说明
```

---

## 一、`litellm/` — 核心库

LiteLLM 的主 Python 包，包含所有 LLM 调用、路由、缓存、代理等核心逻辑。

### 1.1 核心入口

| 文件/目录 | 说明 |
|-----------|------|
| `__init__.py` | 包入口，导出公开 API（`completion()`, `acompletion()`, `embedding()`, `responses()` 等） |
| `main.py` | 核心调用入口，实现 `completion()` / `acompletion()` 等主要函数 |
| `utils.py` | 全局工具函数集合（模型信息查询、Provider 解析、参数处理等） |
| `router.py` | **Router 实现** — 负责负载均衡、故障转移、冷却期、重试逻辑 |
| `exceptions.py` | 自定义异常类（各 Provider 异常映射为 OpenAI 格式） |
| `constants.py` | 全局常量定义 |
| `_version.py` | 版本号 |
| `_logging.py` | 日志配置与 verbose 日志管理 |
| `_uuid.py` | UUID 生成工具 |
| `_redis.py` | Redis 连接管理 |
| `cost_calculator.py` | LLM 调用成本计算器 |
| `budget_manager.py` | 预算管理器（SDK 层） |
| `timeout.py` | 超时处理 |
| `scheduler.py` | 调度器（用于后台任务） |
| `setup_wizard.py` | 交互式安装向导 |

### 1.2 `litellm/llms/` — LLM Provider 实现

**100+ 个 LLM Provider 的适配层**，每个子目录对应一个或多个 Provider。

#### 基础设施

| 目录/文件 | 说明 |
|-----------|------|
| `base.py` | Provider 基类接口定义 |
| `base_llm/` | 基础 LLM HTTP 处理器（HTTP 请求、重试、流式处理等） |
| `custom_llm.py` | 自定义 LLM Provider 基类 |
| `custom_httpx/` | 自定义 HTTPX 客户端实现 |
| `openai_like/` | OpenAI 兼容 API 的通用实现 |
| `litellm_proxy/` | 调用 LiteLLM 自身 Proxy 的 Provider |
| `pass_through/` | 透传模式（直接转发原始请求） |
| `deprecated_providers/` | 已废弃的 Provider 实现 |

#### 主要 Provider 实现

每个 Provider 子目录通常包含：`chat/`（对话补全）、`completion/`（文本补全）、`embed/`（嵌入）、`image_generation/`（图像生成）等子模块，以及 `transformation.py`（请求/响应转换）和 `cost_calculator.py`（成本计算）。

| 目录 | Provider | 说明 |
|------|----------|------|
| `openai/` | OpenAI | OpenAI API 完整适配（Chat、Completion、Responses、Files、Image、Audio 等） |
| `anthropic/` | Anthropic | Claude 系列模型适配（Messages API、流式、Tool Calling 等） |
| `azure/` | Azure OpenAI | Azure OpenAI Service 适配（Chat、Completion、Embedding、Image、Audio 等） |
| `azure_ai/` | Azure AI | Azure AI Models 适配 |
| `bedrock/` | AWS Bedrock | AWS Bedrock 适配（Converse、Invoke、Guardrails 等） |
| `vertex_ai/` | Google Vertex AI | Google Cloud Vertex AI 适配（Gemini、Anthropic on Vertex 等） |
| `gemini/` | Google AI Studio | Google Gemini API 直接适配 |
| `cohere/` | Cohere | Cohere 模型适配 |
| `mistral/` | Mistral AI | Mistral API 适配 |
| `ollama/` | Ollama | 本地 Ollama 模型适配 |
| `vllm/` | vLLM | vLLM 推理引擎适配 |
| `huggingface/` | HuggingFace | HuggingFace Inference API 适配 |
| `replicate/` | Replicate | Replicate 模型适配 |
| `together_ai/` | Together AI | Together AI 平台适配 |
| `fireworks_ai/` | Fireworks AI | Fireworks AI 平台适配 |
| `deepseek/` | DeepSeek | DeepSeek 模型适配 |
| `groq/` | Groq | Groq 高速推理适配 |
| `perplexity/` | Perplexity | Perplexity AI 搜索模型适配 |
| `openrouter/` | OpenRouter | OpenRouter 聚合平台适配 |
| `databricks/` | Databricks | Databricks Foundation Model API 适配 |
| `watsonx/` | IBM WatsonX | IBM WatsonX 模型适配 |
| `sagemaker/` | AWS SageMaker | AWS SageMaker 端点适配 |
| `cloudflare/` | Cloudflare Workers AI | Cloudflare AI 推理适配 |
| `xai/` | xAI (Grok) | xAI 的 Grok 模型适配 |
| `sambanova/` | SambaNova | SambaNova 推理适配 |
| `cerebras/` | Cerebras | Cerebras 高速推理适配 |
| `nvidia_nim/` | NVIDIA NIM | NVIDIA NIM 推理微服务适配 |
| `hosted_vllm/` | Hosted vLLM | 托管 vLLM 服务适配 |
| `elevenlabs/` | ElevenLabs | 语音合成 API 适配 |
| `deepgram/` | Deepgram | 语音识别 API 适配 |
| `stability/` | Stability AI | 图像生成 API 适配 |
| `runwayml/` | RunwayML | 视频生成 API 适配 |
| `fal_ai/` | Fal AI | 图像/视频生成 API 适配 |
| `volcengine/` | 火山引擎 | 字节跳动火山引擎模型适配 |
| `dashscope/` | 百度通义千问 | 阿里云通义千问模型适配 |
| `moonshot/` | Moonshot (月之暗面) | Kimi 模型适配 |
| `minimax/` | MiniMax | MiniMax 模型适配 |
| `amazon_nova/` | Amazon Nova | AWS Amazon Nova 模型适配 |

#### 搜索类 Provider

| 目录 | 说明 |
|------|------|
| `brave/search/` | Brave Search API 适配 |
| `tavily/search/` | Tavily 搜索 API 适配 |
| `serper/search/` | Serper 搜索 API 适配 |
| `exa_ai/search/` | Exa AI 搜索 API 适配 |
| `duckduckgo/search/` | DuckDuckGo 搜索适配 |
| `google_pse/search/` | Google Programmable Search Engine 适配 |
| `searxng/` | SearXNG 元搜索引擎适配 |

### 1.3 `litellm/proxy/` — AI Gateway 代理服务器

**LiteLLM Proxy 的完整实现**，基于 FastAPI 构建，提供统一的 LLM Gateway。

#### 核心文件

| 文件 | 说明 |
|------|------|
| `proxy_server.py` | **主服务器** — FastAPI 应用、启动事件、生命周期管理 |
| `route_llm_request.py` | LLM 请求路由核心逻辑（认证后调用 Router） |
| `common_request_processing.py` | 请求公共处理（成本追踪、日志记录、响应头注入） |
| `_types.py` | Proxy 专用类型定义（请求/响应 Schema、UI 配置等） |
| `utils.py` | Proxy 通用工具函数 |
| `_logging.py` | Proxy 日志配置 |
| `schema.prisma` | 数据库 Schema（Prisma） |
| `health_check.py` | 健康检查端点实现 |
| `caching_routes.py` | 缓存管理 API 路由 |
| `proxy_cli.py` | CLI 命令行工具 |

#### `proxy/auth/` — 认证与授权

| 文件 | 说明 |
|------|------|
| `user_api_key_auth.py` | API Key 认证（创建/验证/管理 Virtual Key） |
| `auth_checks.py` | 请求级权限检查（Key 权限、模型访问控制、预算检查） |
| `auth_checks_organization.py` | 组织级权限检查 |
| `handle_jwt.py` | JWT Token 处理 |
| `oauth2_check.py` | OAuth2 认证流程 |
| `route_checks.py` | 路由级权限检查 |
| `model_checks.py` | 模型级权限检查 |
| `login_utils.py` | 登录工具函数 |
| `litellm_license.py` | License 验证 |

#### `proxy/db/` — 数据库层

| 文件 | 说明 |
|------|------|
| `prisma_client.py` | Prisma 数据库客户端管理 |
| `db_spend_update_writer.py` | **消费记录批量写入器**（Redis 缓冲 → PostgreSQL 批量写入） |
| `db_transaction_queue/` | 数据库事务队列（分布式锁、批量写入） |

#### `proxy/hooks/` — 请求钩子

| 文件 | 说明 |
|------|------|
| `proxy_track_cost_callback.py` | **成本追踪回调** — 每个 LLM 响应后计算并记录成本 |
| `dynamic_rate_limiter.py` | 动态限流器（基于 Token 使用量） |
| `max_budget_limiter.py` | 预算限制器 |
| `parallel_request_limiter.py` | 并发请求限制器 |
| `cache_control_check.py` | 缓存控制检查 |

#### `proxy/management_endpoints/` — 管理 API 端点

| 文件/目录 | 说明 |
|-----------|------|
| `key_management_endpoints.py` | API Key 管理（创建/更新/删除/查询） |
| `model_management_endpoints.py` | 模型部署管理 |
| `team_endpoints.py` | 团队管理 |
| `organization_endpoints.py` | 组织管理 |
| `budget_management_endpoints.py` | 预算管理 |
| `tag_management_endpoints.py` | 标签管理 |
| `router_settings_endpoints.py` | 路由器设置管理 |
| `fallback_management_endpoints.py` | 故障转移配置管理 |
| `usage_endpoints/` | 使用量查询 API |
| `mcp_management_endpoints.py` | MCP Server 管理 |
| `sso/` | SSO 单点登录管理 |
| `scim/` | SCIM 用户管理协议 |

#### `proxy/pass_through_endpoints/` — 透传端点

将请求直接转发到 Provider 原始 API。

| 文件 | 说明 |
|------|------|
| `pass_through_endpoints.py` | 透传端点主入口 |
| `llm_passthrough_endpoints.py` | LLM 透传端点路由 |
| `streaming_handler.py` | 透传流式响应处理 |
| `llm_provider_handlers/` | 各 Provider 透传日志处理器 |

#### `proxy/guardrails/` — 安全护栏

| 文件/目录 | 说明 |
|-----------|------|
| `guardrail_registry.py` | 护栏注册表 |
| `guardrail_endpoints.py` | 护栏管理 API |
| `guardrail_hooks/bedrock_guardrails.py` | AWS Bedrock Guardrails |
| `guardrail_hooks/azure/` | Azure AI Content Safety |
| `guardrail_hooks/openai/` | OpenAI Moderation |
| `guardrail_hooks/ibm_guardrails/` | IBM Guardrails |
| `guardrail_hooks/guardrails_ai/` | Guardrails AI |
| `guardrail_hooks/presidio.py` | Microsoft Presidio（PII 检测） |
| `guardrail_hooks/custom_code/` | 自定义代码护栏 |
| `guardrail_hooks/tool_policy/` | 工具策略护栏 |
| `guardrail_hooks/mcp_security/` | MCP 安全护栏 |
| `guardrail_hooks/content_filter/` | LiteLLM 内置内容过滤器 |

#### 其他 Proxy 子目录

| 目录 | 说明 |
|------|------|
| `client/` | Proxy 内部 HTTP 客户端 |
| `common_utils/` | Proxy 通用工具（Banner、Swagger、调试、性能等） |
| `middleware/` | 中间件（请求计数等） |
| `prompts/` | 提示词管理端点 |
| `policy_engine/` | 策略引擎实现 |
| `enterprise/` | 企业版 Proxy 功能 |
| `example_config_yaml/` | 示例配置文件集 |
| `agent_endpoints/` | Agent 管理端点 |
| `image_endpoints/` | 图像生成端点 |
| `video_endpoints/` | 视频生成端点 |
| `audio_endpoints/` | 音频处理端点 |
| `search_endpoints/` | 搜索端点 |
| `rag_endpoints/` | RAG 端点 |
| `vector_store_endpoints/` | 向量存储端点 |
| `response_api_endpoints/` | Responses API 端点 |
| `anthropic_endpoints/` | Anthropic 兼容端点 |
| `vertex_ai_endpoints/` | Vertex AI 透传端点 |
| `discovery_endpoints/` | 服务发现端点 |
| `health_endpoints/` | 健康检查端点 |

### 1.4 `litellm/router_utils/` — 路由工具

Router 的辅助工具集。

| 文件/目录 | 说明 |
|-----------|------|
| `cooldown_handlers.py` | 冷却期处理 |
| `cooldown_cache.py` | 冷却期缓存管理 |
| `handle_error.py` | 错误处理与重试决策 |
| `pattern_match_deployments.py` | 通配符模型匹配 |
| `health_state_cache.py` | 健康状态缓存 |
| `fallback_event_handlers.py` | 故障转移事件处理 |
| `pre_call_checks/` | 调用前检查（亲和性、速率限制、缓存等） |

### 1.5 `litellm/router_strategy/` — 路由策略

| 文件/目录 | 说明 |
|-----------|------|
| `base_routing_strategy.py` | 路由策略基类 |
| `least_busy.py` | 最少忙碌策略 |
| `lowest_cost.py` | 最低成本策略 |
| `lowest_latency.py` | 最低延迟策略 |
| `lowest_tpm_rpm.py` | 最低 TPM/RPM 利用率策略 |
| `simple_shuffle.py` | 简单随机策略 |
| `budget_limiter.py` | 预算限制器策略 |
| `tag_based_routing.py` | 基于标签的路由策略 |
| `auto_router/` | 自动路由器（基于嵌入向量匹配） |
| `complexity_router/` | 复杂度路由器（基于请求复杂度选择模型） |

### 1.6 `litellm/caching/` — 缓存系统

| 文件 | 说明 |
|------|------|
| `base_cache.py` | 缓存基类接口 |
| `caching.py` | 缓存核心实现（缓存键生成、精确匹配） |
| `caching_handler.py` | 缓存处理器 |
| `dual_cache.py` | 双层缓存（内存 + Redis） |
| `in_memory_cache.py` | 内存缓存 |
| `redis_cache.py` | Redis 缓存 |
| `redis_semantic_cache.py` | Redis 语义缓存 |
| `qdrant_semantic_cache.py` | Qdrant 语义缓存 |
| `disk_cache.py` | 磁盘缓存 |
| `s3_cache.py` | S3 缓存 |
| `gcs_cache.py` | GCS 缓存 |
| `azure_blob_cache.py` | Azure Blob 缓存 |

### 1.7 `litellm/types/` — 类型定义

Pydantic 模型和类型提示。

| 目录 | 说明 |
|------|------|
| `completion.py` | Chat/Completion 请求/响应类型 |
| `embedding.py` | Embedding 请求/响应类型 |
| `responses/` | Responses API 类型 |
| `router.py` | Router 配置类型 |
| `caching.py` | 缓存相关类型 |
| `guardrails.py` | 护栏类型 |
| `mcp.py` | MCP 相关类型 |
| `mcp_server/` | MCP Server 管理类型 |
| `agents.py` | Agent 类型 |
| `llms/` | 各 Provider 专用类型 |
| `proxy/` | Proxy 专用类型（管理端点、SSO、护栏等） |
| `integrations/` | 各集成插件类型 |

### 1.8 `litellm/integrations/` — 第三方集成

#### Observability / 日志平台

| 文件/目录 | 说明 |
|-----------|------|
| `datadog/` | Datadog 集成（日志、指标、LLM Observability） |
| `langfuse/` | Langfuse 集成（LLM Observability、Prompt 管理） |
| `langsmith/` | LangSmith 集成 |
| `opentelemetry.py` | OpenTelemetry 集成 |
| `opik/` | Opik 集成 |
| `arize/` | Arize AI / Arize Phoenix 集成 |
| `braintrust_logging.py` | Braintrust 集成 |
| `prometheus.py` | Prometheus 指标集成 |
| `weights_biases.py` | Weights & Biases 集成 |
| `mlflow.py` | MLflow 集成 |
| `deepeval/` | DeepEval 集成 |

#### 存储 / 数据管道

| 文件/目录 | 说明 |
|-----------|------|
| `s3_v2.py` | AWS S3 日志存储（推荐） |
| `gcs_bucket/` | Google Cloud Storage 日志存储 |
| `gcs_pubsub/` | GCS Pub/Sub 日志推送 |
| `azure_storage/` | Azure Blob Storage |
| `dynamodb.py` | AWS DynamoDB |
| `sqs.py` | AWS SQS 消息队列 |
| `supabase.py` | Supabase |

#### 成本管理

| 文件/目录 | 说明 |
|-----------|------|
| `cloudzero/` | CloudZero 成本追踪 |
| `vantage/` | Vantage 云成本管理 |
| `lago.py` | Lago 计费平台集成 |

#### 其他

| 文件/目录 | 说明 |
|-----------|------|
| `custom_logger.py` | 自定义日志基类（所有集成的父类） |
| `SlackAlerting/` | Slack 告警通知 |
| `email_alerting.py` | 邮件告警通知 |
| `custom_sso_handler.py` | 自定义 SSO 处理器 |
| `generic_api/` | 通用 API 回调 |

### 1.9 `litellm/litellm_core_utils/` — 核心工具

| 文件/目录 | 说明 |
|-----------|------|
| `litellm_logging.py` | **核心日志记录器**（Success/Failure 事件处理、成本计算触发） |
| `streaming_handler.py` | 流式响应块处理器 |
| `token_counter.py` | Token 计数器 |
| `llm_cost_calc/` | LLM 成本计算 |
| `llm_request_utils.py` | LLM 请求参数处理 |
| `llm_response_utils/` | LLM 响应处理（元数据注入、头部提取） |
| `model_param_helper.py` | 模型参数辅助（缓存键生成） |
| `get_model_cost_map.py` | 模型成本映射 |
| `get_llm_provider_logic.py` | Provider 解析逻辑 |
| `exception_mapping_utils.py` | 异常映射工具 |
| `logging_callback_manager.py` | 日志回调管理器 |
| `health_check_utils.py` | 健康检查工具 |
| `credential_accessor.py` | 凭证访问器 |
| `fallback_utils.py` | 故障转移工具 |
| `redact_messages.py` | 消息脱敏 |
| `tokenizers/` | 本地 Tokenizer 文件 |
| `audio_utils/` | 音频处理工具 |
| `prompt_templates/` | 提示词模板处理 |

### 1.10 其他核心子目录

| 目录 | 说明 |
|------|------|
| `secret_managers/` | **密钥管理器**（AWS SM、GCP SM、HashiCorp Vault、Azure AD、CyberArk） |
| `experimental_mcp_client/` | MCP (Model Context Protocol) 实验性客户端 |
| `responses/` | Responses API 转换层 |
| `interactions/` | 交互日志记录 |
| `a2a_protocol/` | A2A (Agent-to-Agent) 协议 |
| `vector_stores/` | 向量存储集成 |
| `containers/` | 沙箱容器支持 |
| `realtime_api/` | 实时 API（WebSocket） |
| `rerank_api/` | Rerank API |
| `search/` | 搜索 API |
| `images/` | 图像生成 |
| `videos/` | 视频生成 |
| `ocr/` | OCR 光学字符识别 |
| `rag/` | RAG 检索增强生成 |
| `assistants/` | OpenAI Assistants API 兼容 |
| `fine_tuning/` | 微调支持 |
| `google_genai/` | Google GenAI SDK 适配 |
| `anthropic_interface/` | Anthropic 特定接口适配 |
| `batch_completion/` | 批量补全 |
| `batches/` | 批量任务 |
| `completion_extras/` | 补全额外功能 |

---

## 二、`litellm-proxy-extras/` — Proxy 独立部署包

| 文件/目录 | 说明 |
|-----------|------|
| `litellm_proxy_extras/` | 包源码 |
| `litellm_proxy_extras/migrations/` | **Prisma 数据库迁移文件**（Proxy 启动时自动执行） |
| `litellm_proxy_extras/schema.prisma` | Proxy 数据库 Schema 副本 |
| `dist/` | 构建产物 |
| `pyproject.toml` | 包配置 |
| `migration_runbook.md` | 迁移操作手册 |

---

## 三、`enterprise/` — 企业版功能

闭源企业功能，需要 LiteLLM Enterprise License。

| 文件/目录 | 说明 |
|-----------|------|
| `litellm_enterprise/` | 企业版核心代码 |
| `litellm_enterprise/proxy/` | 企业版 Proxy 功能（SCIM、自定义认证等） |
| `litellm_enterprise/proxy/auth/` | 企业版认证（SCIM v2、SAML 等） |
| `litellm_enterprise/enterprise_callbacks/` | 企业版回调 |
| `enterprise_hooks/` | 企业版护栏钩子 |
| `enterprise_ui/` | 企业版 UI 配置 |
| `cloudformation_stack/` | AWS CloudFormation 部署模板 |
| `dist/` | 构建产物 |

---

## 四、`ui/litellm-dashboard/` — Admin Dashboard

基于 **Next.js + React + TypeScript + Tailwind CSS** 构建的管理后台。

### 4.1 配置文件

| 文件 | 说明 |
|------|------|
| `package.json` | Node.js 依赖 |
| `next.config.mjs` | Next.js 配置 |
| `tailwind.config.ts` | Tailwind CSS 配置 |
| `vitest.config.ts` | Vitest 测试配置 |

### 4.2 `src/app/` — 页面路由

| 目录 | 说明 |
|------|------|
| `(dashboard)/` | **主仪表盘布局**（需登录） |
| `(dashboard)/logs/` | 日志查看 |
| `(dashboard)/teams/` | 团队管理 |
| `(dashboard)/users/` | 用户管理 |
| `(dashboard)/models-and-endpoints/` | 模型与端点管理 |
| `(dashboard)/virtual-keys/` | Virtual Key 管理 |
| `(dashboard)/usage/` | 使用量/消费统计 |
| `(dashboard)/settings/` | 设置（Admin、Router、日志/告警、UI 主题） |
| `(dashboard)/policies/` | 策略管理 |
| `(dashboard)/organizations/` | 组织管理 |
| `(dashboard)/guardrails/` | 安全护栏管理 |
| `(dashboard)/playground/` | API Playground 调试 |
| `(dashboard)/model-hub/` | 模型市场 |
| `(dashboard)/experimental/` | 实验性功能 |
| `(dashboard)/mcpServers/` | MCP Server 管理 |
| `(dashboard)/tools/` | 工具管理（MCP、Vector Store） |
| `(dashboard)/components/` | 各页面专用组件 |
| `(dashboard)/hooks/` | 各页面自定义 Hook |
| `login/` | 登录页面 |
| `chat/` | 聊天页面 |
| `model_hub/` | 模型市场首页 |
| `onboarding/` | 新手引导 |

### 4.3 `src/components/` — 可复用组件

| 目录 | 说明 |
|------|------|
| `ui/` | 基础 UI 组件（Radix UI + Tailwind） |
| `common_components/` | 通用业务组件（Filters、Table、Button 等） |
| `Navbar/` | 导航栏（Blog 下拉、User 下拉、Worker 下拉等） |
| `view_logs/` | 日志查看组件 |
| `UsagePage/` | 使用量页面组件 |
| `guardrails/` | 护栏管理组件 |
| `playground/` | Playground 组件 |
| `add_model/` | 添加模型组件 |
| `policies/` | 策略管理组件 |
| `Settings/` | 设置组件 |
| `prompts/` | 提示词管理组件 |
| `mcp_tools/` | MCP 工具管理组件 |
| `agents/` | Agent 管理组件 |
| `SearchTools/` | 搜索工具组件 |
| `AccessGroups/` | 访问组组件 |
| `model_dashboard/` | 模型仪表盘 |
| `router_settings/` | 路由器设置 |
| `cache_settings/` | 缓存设置 |
| `tag_management/` | 标签管理 |
| `GuardrailsMonitor/` | 护栏监控 |
| `EntityUsageExport/` | 使用量导出 |

### 4.4 其他 UI 目录

| 目录 | 说明 |
|------|------|
| `src/utils/` | 工具函数（JWT、错误处理、数据转换等） |
| `src/hooks/` | 自定义 React Hook |
| `src/contexts/` | React Context（Theme、AntD、React Query） |
| `src/data/` | 静态数据（合规提示词模板等） |
| `public/assets/logos/` | Provider Logo 资源 |
| `e2e_tests/` | E2E 测试 |

---

## 五、`tests/` — 测试套件

| 目录 | 说明 |
|------|------|
| `test_litellm/` | **核心库单元测试**（88 个文件） |
| `llm_translation/` | Provider 翻译测试（88 个文件） |
| `proxy_unit_tests/` | Proxy 单元测试（74 个文件） |
| `router_unit_tests/` | Router 单元测试 |
| `local_testing/` | 本地开发测试（160 个文件） |
| `guardrails_tests/` | 安全护栏测试 |
| `audio_tests/` | 音频功能测试 |
| `image_gen_tests/` | 图像生成测试 |
| `search_tests/` | 搜索功能测试 |
| `mcp_tests/` | MCP 功能测试 |
| `vector_store_tests/` | 向量存储测试 |
| `pass_through_tests/` | 透传端点测试 |
| `batches_tests/` | 批量任务测试 |
| `load_tests/` | 负载/压力测试 |
| `otel_tests/` | OpenTelemetry 测试 |
| `agent_tests/` | Agent 功能测试 |
| `spend_tracking_tests/` | 消费追踪测试 |
| `logging_callback_tests/` | 日志回调测试 |
| `openai_endpoints_tests/` | OpenAI 兼容端点测试 |
| `code_coverage_tests/` | 代码覆盖率测试 |
| `proxy_admin_ui_tests/` | Admin UI API 测试 |
| `ui_e2e_tests/` | UI E2E 测试 |

---

## 六、`docs/my-website/` — 文档网站

基于 **Docusaurus** 构建的文档站点（https://docs.litellm.ai）。

| 目录 | 说明 |
|------|------|
| `docs/providers/` | **各 Provider 文档**（156 个文件） |
| `docs/proxy/` | Proxy 文档（部署、认证、负载均衡、缓存、护栏等） |
| `docs/tutorials/` | 教程（62 个） |
| `docs/observability/` | 可观测性文档（45 个） |
| `docs/caching/` | 缓存文档 |
| `docs/embedding/` | 嵌入文档 |
| `docs/completion/` | 补全文档 |
| `docs/search/` | 搜索文档 |
| `docs/pass_through/` | 透传文档 |
| `docs/secret_managers/` | 密钥管理器文档 |
| `docs/troubleshoot/` | 故障排除文档 |
| `docs/routing.md` | 路由/负载均衡文档 |
| `docs/response_api.md` | Responses API 文档 |
| `docs/mcp.md` | MCP 文档 |
| `src/` | 站点源码（主题、组件） |
| `blog/` | 博客文章 |
| `release_notes/` | 发布说明 |
| `img/` | 文档图片 |
| `sidebars.js` | 侧边栏配置 |

---

## 七、辅助目录

| 目录/文件 | 说明 |
|-----------|------|
| `cookbook/` | 使用示例（55 个文件，各 Provider、Router、缓存等） |
| `scripts/` | 辅助脚本（部署、发布、测试等） |
| `ci_cd/` | CI/CD 配置 |
| `docker/` | Docker 相关配置 |
| `deploy/` | 部署配置 |
| `db_scripts/` | 数据库初始化脚本 |
| `litellm-js/` | JavaScript SDK（消费日志查询面板） |
| `model_prices_and_context_window.json` | **模型定价与上下文窗口数据** |
| `provider_endpoints_support.json` | Provider 端点支持矩阵 |
| `policy_templates.json` | 策略模板 |
| `schema.prisma` | 主数据库 Schema |
| `prometheus.yml` | Prometheus 监控配置 |
| `Makefile` | 构建/测试/格式化命令 |
| `pyproject.toml` | Python 项目配置（Poetry） |
| `ruff.toml` | Ruff Linter 配置 |
| `pyrightconfig.json` | Pyright 类型检查配置 |
| `docker-compose.yml` | Docker Compose 编排 |
| `Dockerfile` | 主 Dockerfile |

