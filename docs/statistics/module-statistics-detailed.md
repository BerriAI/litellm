# LiteLLM 功能模块细粒度统计

本文档对 LiteLLM 各功能模块进行深入的功能粒度拆解统计。

---

## 总览

| 模块分类 | 文件数 | 代码行数 |
|---------|--------|----------|
| **litellm/ 总计** | 1,720 | 550,858 |
| **tests/ 总计** | 1,638 | 611,481 |

---

## 一、SDK 核心入口 (`litellm/*.py`)

**总计：20 文件，39,559 行**

| 文件 | 代码行数 | 功能说明 |
|------|----------|----------|
| `router.py` | 10,211 | 路由器主逻辑，负载均衡、故障转移 |
| `utils.py` | 9,538 | 核心工具函数，get_llm_provider() |
| `main.py` | 7,806 | SDK 主入口，completion()/acompletion() |
| `cost_calculator.py` | 2,294 | Token 费用计算 |
| `__init__.py` | 2,177 | 包初始化，导出公共 API |
| `constants.py` | 1,583 | 常量定义 |
| `_lazy_imports_registry.py` | 1,440 | 延迟导入注册 |
| `exceptions.py` | 1,030 | 异常类定义 |
| `setup_wizard.py` | 668 | 设置向导 |
| `_redis.py` | 562 | Redis 工具 |
| `_logging.py` | 491 | 日志配置 |
| 其他 | ~2,759 | 辅助工具 |

---

## 二、提供商实现 (`litellm/llms/`)

**总计：758 文件，161,649 行**

### 2.1 主要提供商统计

| 提供商 | 文件数 | 代码行数 | 说明 |
|--------|--------|----------|------|
| `bedrock/` | 58 | 20,920 | AWS Bedrock |
| `vertex_ai/` | 69 | 20,172 | Google Vertex AI |
| `custom_httpx/` | 8 | 15,159 | 核心 HTTP 处理器 |
| `openai/` | 52 | 12,628 | OpenAI |
| `anthropic/` | 36 | 11,557 | Anthropic Claude |
| `base_llm/` | 42 | 7,649 | 基础类定义 |
| `azure/` | 29 | 7,039 | Azure OpenAI |
| `azure_ai/` | 38 | 5,103 | Azure AI |
| `gemini/` | 19 | 4,254 | Google Gemini |
| `sap/` | 7 | 2,177 | SAP AI |
| `cohere/` | 11 | 2,117 | Cohere |
| `fal_ai/` | 13 | 2,015 | Fal.ai |
| `oci/` | 4 | 1,878 | Oracle Cloud |
| `sagemaker/` | 8 | 1,820 | AWS SageMaker |

### 2.2 其他提供商（代码行 < 2000）

| 提供商 | 文件数 | 代码行数 |
|--------|--------|----------|
| `runwayml/` | 8 | 1,810 |
| `litellm_proxy/` | 11 | 1,808 |
| `watsonx/` | 14 | 1,692 |
| `black_forest_labs/` | 8 | 1,649 |
| `huggingface/` | 6 | 1,591 |
| `databricks/` | 8 | 1,582 |
| `ollama/` | 4 | 1,462 |
| `openrouter/` | 8 | 1,380 |
| `gigachat/` | 8 | 1,357 |
| `mistral/` | 8 | 1,253 |
| `openai_like/` | 8 | 1,164 |
| `github_copilot/` | 5 | 1,127 |
| `a2a/` | 7 | 1,110 |
| `chatgpt/` | 5 | 1,057 |
| `fireworks_ai/` | 8 | 994 |
| `volcengine/` | 6 | 983 |
| `perplexity/` | 7 | 857 |
| `manus/` | 5 | 790 |
| `langgraph/` | 4 | 750 |
| `xai/` | 6 | 739 |
| `hosted_vllm/` | 5 | 718 |

---

## 三、代理服务器 (`litellm/proxy/`)

**总计：391 文件，193,304 行**

### 3.1 子模块统计

| 子模块 | 文件数 | 代码行数 | 功能说明 |
|--------|--------|----------|----------|
| `management_endpoints/` | 38 | 35,195 | 管理 API 端点 |
| `guardrails/` | 93 | 32,074 | 护栏/安全检查 |
| `_experimental/` | 22 | 13,036 | 实验性功能（MCP 服务器） |
| `auth/` | 14 | 10,773 | 身份验证 |
| `pass_through_endpoints/` | 17 | 10,236 | 直通端点 |
| `enterprise/` | 136 | 9,222 | 企业功能 |
| `hooks/` | 23 | 8,677 | 代理钩子 |
| `spend_tracking/` | 5 | 5,614 | 消费追踪 |
| `db/` | 17 | 5,560 | 数据库层 |
| `common_utils/` | 23 | 5,441 | 通用工具 |
| `policy_engine/` | 11 | 4,304 | 策略引擎 |
| `client/` | 24 | 4,221 | 代理客户端 CLI |

### 3.2 管理端点细分 (`management_endpoints/`)

| 文件 | 代码行数 | 功能 |
|------|----------|------|
| `key_management_endpoints.py` | 5,497 | API 密钥管理 |
| `team_endpoints.py` | 4,570 | 团队管理 |
| `ui_sso.py` | 3,832 | SSO/UI 登录 |
| `internal_user_endpoints.py` | 2,600 | 内部用户管理 |
| `mcp_management_endpoints.py` | 2,223 | MCP 管理 |
| `scim/scim_v2.py` | 1,754 | SCIM 协议 |
| `model_management_endpoints.py` | 1,476 | 模型管理 |
| `policy_endpoints/endpoints.py` | 1,317 | 策略端点 |
| `organization_endpoints.py` | 1,198 | 组织管理 |
| `project_endpoints.py` | 936 | 项目管理 |
| `customer_endpoints.py` | 925 | 客户管理 |

### 3.3 身份验证细分 (`auth/`)

| 文件 | 代码行数 | 功能 |
|------|----------|------|
| `auth_checks.py` | 3,704 | 权限检查 |
| `user_api_key_auth.py` | 2,013 | API 密钥验证 |
| `handle_jwt.py` | 1,615 | JWT 处理 |
| `auth_utils.py` | 900 | 认证工具 |
| `route_checks.py` | 673 | 路由权限检查 |
| `model_checks.py` | 381 | 模型访问检查 |
| `login_utils.py` | 356 | 登录工具 |
| `oauth2_check.py` | 224 | OAuth2 检查 |

### 3.4 代理钩子细分 (`hooks/`)

| 文件 | 代码行数 | 功能 |
|------|----------|------|
| `parallel_request_limiter_v3.py` | 1,773 | 并发请求限流 V3 |
| `litellm_skills/main.py` | 914 | Skills 注入 |
| `parallel_request_limiter.py` | 891 | 并发请求限流 |
| `dynamic_rate_limiter_v3.py` | 809 | 动态限流 V3 |
| `key_management_event_hooks.py` | 607 | 密钥事件钩子 |
| `batch_rate_limiter.py` | 456 | 批量限流 |
| `proxy_track_cost_callback.py` | 389 | 费用追踪回调 |
| `model_max_budget_limiter.py` | 318 | 模型预算限制 |
| `prompt_injection_detection.py` | 284 | 注入检测 |
| `max_budget_per_session_limiter.py` | 267 | 会话预算限制 |

### 3.5 数据库层细分 (`db/`)

| 文件 | 代码行数 | 功能 |
|------|----------|------|
| `db_spend_update_writer.py` | 2,219 | 消费数据批量写入 |
| `db_transaction_queue/redis_update_buffer.py` | 677 | Redis 更新缓冲 |
| `prisma_client.py` | 488 | Prisma 客户端 |
| `tool_registry_writer.py` | 440 | 工具注册写入 |
| `db_transaction_queue/spend_update_queue.py` | 246 | 消费更新队列 |
| `create_views.py` | 229 | 数据库视图创建 |

### 3.6 护栏系统细分 (`guardrails/`)

**根目录文件（4,322 行）：**

| 文件 | 代码行数 | 功能 |
|------|----------|------|
| `guardrail_endpoints.py` | 2,175 | 护栏 API 端点 |
| `guardrail_registry.py` | 698 | 护栏注册表 |
| `usage_endpoints.py` | 683 | 使用量端点 |

**护栏钩子实现 (`guardrail_hooks/`，27,752 行）：**

| 护栏类型 | 文件数 | 代码行数 |
|----------|--------|----------|
| `litellm_content_filter/` | 7 | 3,347 |
| `panw_prisma_airs/` | 2 | 2,050 |
| `custom_code/` | 5 | 1,488 |
| `noma/` | 3 | 1,309 |
| `pillar/` | 2 | 978 |
| `mcp_jwt_signer/` | 2 | 975 |
| `model_armor/` | 2 | 832 |
| `ibm_guardrails/` | 2 | 829 |
| `grayswan/` | 2 | 769 |
| `prompt_security/` | 2 | 766 |
| `lasso/` | 2 | 713 |
| `block_code_execution/` | 2 | 706 |
| `azure/` | 4 | 682 |
| `enkryptai/` | 2 | 595 |

### 3.7 MCP 服务器 (`_experimental/`)

| 文件 | 代码行数 | 功能 |
|------|----------|------|
| `mcp_server/mcp_server_manager.py` | 3,088 | MCP 服务器管理 |
| `mcp_server/server.py` | 2,846 | MCP 服务器核心 |
| `mcp_server/rest_endpoints.py` | 1,116 | REST 端点 |
| `mcp_server/auth/user_api_key_auth_mcp.py` | 1,098 | MCP 认证 |
| `mcp_server/discoverable_endpoints.py` | 1,017 | 可发现端点 |
| `mcp_server/db.py` | 884 | MCP 数据库 |
| `mcp_server/byok_oauth_endpoints.py` | 789 | BYOK OAuth |

---

## 四、核心工具库 (`litellm/litellm_core_utils/`)

**总计：60 文件，30,453 行**

### 4.1 子目录统计

| 子目录 | 文件数 | 代码行数 | 功能 |
|--------|--------|----------|------|
| `prompt_templates/` | 4 | 7,246 | 提示词模板 |
| `llm_cost_calc/` | 3 | 1,809 | 费用计算 |
| `llm_response_utils/` | 5 | 1,253 | 响应处理工具 |
| `audio_utils/` | 1 | 275 | 音频工具 |

### 4.2 根目录核心文件

| 文件 | 代码行数 | 功能 |
|------|----------|------|
| `litellm_logging.py` | 5,762 | 日志记录系统 |
| `exception_mapping_utils.py` | 2,523 | 异常映射 |
| `streaming_handler.py` | 2,414 | 流式响应处理 |
| `get_llm_provider_logic.py` | 958 | 提供商解析逻辑 |
| `token_counter.py` | 811 | Token 计数 |
| `streaming_chunk_builder_utils.py` | 741 | 流式块构建 |
| `realtime_streaming.py` | 629 | 实时流式处理 |
| `logging_worker.py` | 527 | 日志工作器 |
| `logging_callback_manager.py` | 513 | 日志回调管理 |
| `core_helpers.py` | 437 | 核心辅助函数 |

---

## 五、路由系统

**总计：39 文件，16,287 行**

### 5.1 路由工具 (`router_utils/`)

| 文件 | 代码行数 | 功能 |
|------|----------|------|
| `pre_call_checks/deployment_affinity_check.py` | 555 | 部署亲和性检查 |
| `cooldown_handlers.py` | 459 | 冷却期处理 |
| `pre_call_checks/model_rate_limit_check.py` | 332 | 模型限流检查 |
| `pattern_match_deployments.py` | 264 | 部署模式匹配 |
| `prompt_caching_cache.py` | 259 | 提示缓存 |
| `fallback_event_handlers.py` | 257 | 故障转移处理 |
| `search_api_router.py` | 227 | 搜索 API 路由 |
| `cooldown_cache.py` | 193 | 冷却期缓存 |

### 5.2 路由策略 (`router_strategy/`)

| 文件 | 代码行数 | 功能 |
|------|----------|------|
| `budget_limiter.py` | 898 | 预算限制策略 |
| `lowest_tpm_rpm_v2.py` | 668 | 最低 TPM/RPM V2 |
| `lowest_latency.py` | 627 | 最低延迟策略 |
| `complexity_router/complexity_router.py` | 410 | 复杂度路由 |
| `lowest_cost.py` | 330 | 最低成本策略 |
| `tag_based_routing.py` | 276 | 标签路由 |
| `base_routing_strategy.py` | 261 | 基础路由策略 |
| `least_busy.py` | 250 | 最空闲策略 |
| `lowest_tpm_rpm.py` | 249 | 最低 TPM/RPM |
| `auto_router/auto_router.py` | 136 | 自动路由 |
| `simple_shuffle.py` | 64 | 简单随机 |

---

## 六、缓存系统 (`litellm/caching/`)

**总计：16 文件，6,159 行**

| 文件 | 代码行数 | 功能 |
|------|----------|------|
| `redis_cache.py` | 1,700 | Redis 缓存 |
| `caching_handler.py` | 1,064 | 缓存处理器 |
| `caching.py` | 926 | 缓存核心 |
| `dual_cache.py` | 515 | 双层缓存 |
| `redis_semantic_cache.py` | 450 | Redis 语义缓存 |
| `qdrant_semantic_cache.py` | 446 | Qdrant 语义缓存 |
| `in_memory_cache.py` | 288 | 内存缓存 |
| `s3_cache.py` | 193 | S3 缓存 |
| `gcs_cache.py` | 113 | GCS 缓存 |
| `redis_cluster_cache.py` | 109 | Redis 集群缓存 |
| `azure_blob_cache.py` | 107 | Azure Blob 缓存 |
| `disk_cache.py` | 93 | 磁盘缓存 |
| `base_cache.py` | 64 | 基础缓存类 |

---

## 七、类型定义 (`litellm/types/`)

**总计：160 文件，24,803 行**

### 7.1 子目录统计

| 子目录 | 文件数 | 代码行数 | 功能 |
|--------|--------|----------|------|
| `llms/` | 32 | 7,373 | LLM 相关类型 |
| `proxy/` | 59 | 4,633 | 代理相关类型 |
| `integrations/` | 24 | 1,904 | 集成相关类型 |
| `interactions/` | 2 | 1,386 | 交互相关类型 |

### 7.2 根目录主要文件

| 文件 | 代码行数 | 功能 |
|------|----------|------|
| `utils.py` | 3,639 | 工具类型 |
| `guardrails.py` | 879 | 护栏类型 |
| `router.py` | 789 | 路由器类型 |
| `files.py` | 323 | 文件类型 |
| `rag.py` | 295 | RAG 类型 |
| `agents.py` | 292 | Agent 类型 |
| `vector_stores.py` | 274 | 向量存储类型 |
| `mcp.py` | 203 | MCP 类型 |
| `completion.py` | 193 | 补全类型 |

---

## 八、集成模块 (`litellm/integrations/`)

**总计：142 文件，38,683 行**

### 8.1 主要集成

| 集成 | 文件数 | 代码行数 | 功能 |
|------|--------|----------|------|
| `SlackAlerting/` | 5 | 2,466 | Slack 告警 |
| `langfuse/` | 6 | 2,334 | Langfuse 可观测性 |
| `datadog/` | 6 | 2,244 | Datadog 监控 |
| `arize/` | 6 | 1,724 | Arize AI |
| `websearch_interception/` | 4 | 1,640 | 网页搜索拦截 |
| `focus/` | 15 | 1,259 | Focus |
| `cloudzero/` | 5 | 1,185 | CloudZero 成本 |
| `gitlab/` | 3 | 1,163 | GitLab |
| `gcs_bucket/` | 3 | 1,024 | GCS Bucket |
| `opik/` | 7 | 920 | Opik |
| `bitbucket/` | 3 | 891 | Bitbucket |
| `dotprompt/` | 3 | 837 | DotPrompt |
| `email_templates/` | 5 | 755 | 邮件模板 |

---

## 九、密钥管理 (`litellm/secret_managers/`)

**总计：11 文件，2,737 行**

| 文件 | 代码行数 | 功能 |
|------|----------|------|
| `hashicorp_secret_manager.py` | 677 | HashiCorp Vault |
| `aws_secret_manager_v2.py` | 539 | AWS Secrets Manager V2 |
| `cyberark_secret_manager.py` | 350 | CyberArk |
| `main.py` | 290 | 主入口 |
| `secret_manager_handler.py` | 184 | 处理器 |
| `base_secret_manager.py` | 180 | 基础类 |
| `aws_secret_manager.py` | 143 | AWS Secrets Manager |
| `get_azure_ad_token_provider.py` | 121 | Azure AD Token |
| `google_secret_manager.py` | 117 | Google Secret Manager |
| `google_kms.py` | 43 | Google KMS |

---

## 十、扩展 API 模块

### 10.1 Responses API (`litellm/responses/`)

**总计：12 文件，11,268 行**

| 文件 | 代码行数 | 功能 |
|------|----------|------|
| `litellm_completion_transformation/transformation.py` | 2,200 | 补全转换 |
| `main.py` | 1,988 | 主入口 |
| `streaming_iterator.py` | 1,338 | 流式迭代器 |
| `mcp/litellm_proxy_mcp_handler.py` | 1,306 | MCP 处理 |
| `litellm_completion_transformation/streaming_iterator.py` | 1,195 | 转换流式迭代 |
| `mcp/mcp_streaming_iterator.py` | 798 | MCP 流式迭代 |
| `utils.py` | 726 | 工具函数 |
| `mcp/chat_completions_handler.py` | 670 | MCP 聊天补全 |

### 10.2 A2A 协议 (`litellm/a2a_protocol/`)

**总计：26 文件，4,226 行**

| 子模块 | 文件数 | 代码行数 | 功能 |
|--------|--------|----------|------|
| `providers/` | 14 | 1,770 | 提供商适配 |
| `litellm_completion_bridge/` | 3 | 604 | 补全桥接 |

**根目录主要文件：**

| 文件 | 代码行数 | 功能 |
|------|----------|------|
| `main.py` | 744 | A2A 主入口 |
| `exception_mapping_utils.py` | 203 | 异常映射 |
| `streaming_iterator.py` | 184 | 流式迭代 |
| `exceptions.py` | 150 | 异常定义 |
| `card_resolver.py` | 144 | Agent Card 解析 |

---

## 十一、测试代码统计 (`tests/`)

**总计：1,638 文件，611,481 行**

| 测试目录 | 文件数 | 代码行数 | 功能 |
|----------|--------|----------|------|
| `test_litellm/` | 928 | 370,896 | SDK 单元测试 |
| `local_testing/` | 144 | 63,530 | 本地测试 |
| `llm_translation/` | 90 | 38,300 | 提供商集成测试 |
| `proxy_unit_tests/` | 65 | 28,134 | 代理单元测试 |
| `logging_callback_tests/` | 33 | 12,987 | 日志回调测试 |
| `mcp_tests/` | 16 | 8,563 | MCP 测试 |
| `litellm_utils_tests/` | 17 | 7,691 | 工具测试 |
| `guardrails_tests/` | 18 | 7,586 | 护栏测试 |
| `enterprise/` | 12 | 7,538 | 企业功能测试 |
| `router_unit_tests/` | 18 | 6,612 | 路由器测试 |
| `pass_through_unit_tests/` | 25 | 6,545 | 直通测试 |

---

*统计时间：2026-04-17*
*基于 LiteLLM 仓库主分支代码*
