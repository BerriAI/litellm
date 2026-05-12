# LiteLLM 核心域对象与关联关系

> 本文从工程视角梳理 LiteLLM 的领域模型：**SDK 核心域**（Router / ModelResponse / Cache / Logger / 异常）与 **Proxy 域**（数据库实体 + 运行时单例 + 认证聚合），并给出它们之间的关联关系。
>
> 所有实体均标注源码位置（`文件:行号`），便于对照查阅。

---

## 0. 整体分层

```
┌──────────────────────────────────────────────────────────────────┐
│                        Proxy 层 (FastAPI)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐    │
│  │  Auth / UAKA │  │ ProxyLogging │  │ Management Endpoints │    │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘    │
│         ↓                 ↓                     ↓                │
│   ┌─────────────────────────────────────────────────────────┐    │
│   │  PrismaClient → PostgreSQL (schema.prisma, 50+ 表)      │    │
│   └─────────────────────────────────────────────────────────┘    │
│                            ↓                                     │
│   ┌─────────────────────────────────────────────────────────┐    │
│   │  llm_router: Router  (全局单例, 负载均衡 + 熔断 + 降级)  │    │
│   └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────┬───────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────┐
│                        SDK 核心层 (litellm/)                      │
│  completion / acompletion / embedding  →  Provider Adapter        │
│       │                                                           │
│       ├── ModelResponse / Usage / Choices / Message (OpenAI 兼容) │
│       ├── CustomStreamWrapper (流式统一封装)                      │
│       ├── Cache / DualCache / RedisCache                          │
│       ├── CustomLogger + Logging (回调 & StandardLoggingPayload)  │
│       └── Exceptions (APIError / RateLimit / ContextWindow ...)   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 1. SDK 核心域对象

### 1.1 Router — 负载均衡 & 可靠性编排


| 属性                                                                                        | 位置                             | 含义                                                                                                        |
| ------------------------------------------------------------------------------------------- | -------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `Router`                                                                                    | `litellm/router.py:216`          | 核心类，统一管理多 deployment 的调度                                                                        |
| `model_list`                                                                                | 构造参数，:227                   | `List[DeploymentTypedDict]`，所有可调度节点                                                                 |
| `model_names`                                                                               | :217                             | 所有`model_name` 集合（即模型组别名）                                                                       |
| `fallbacks` / `default_fallbacks` / `context_window_fallbacks` / `content_policy_fallbacks` | :267–269, :264                  | 四类降级链（参见`cooldown_handlers_analysis.md` 与 fallback 机制）                                          |
| `cache_responses` / `cache`                                                                 | :242                             | 响应缓存开关及`Cache` 实例                                                                                  |
| `cooldown_cache`                                                                            | `router_utils/cooldown_cache.py` | 熔断节点缓存（TTL 自动恢复）                                                                                |
| `failed_calls`                                                                              | DualCache                        | v1 计数策略的失败计数器                                                                                     |
| `routing_strategy`                                                                          | :295–303                        | `simple-shuffle` / `least-busy` / `usage-based-routing` / `latency-based-routing` / `cost-based-routing` 等 |
| `allowed_fails` / `allowed_fails_policy` / `num_retries` / `retry_policy` / `cooldown_time` | :282–292                        | 重试与熔断参数                                                                                              |
| `provider_budget_config`                                                                    | :304                             | 按 provider 的预算硬限                                                                                      |
| `service_logger_obj`                                                                        | ServiceLogging 实例              | OTel/Prometheus 等指标上报                                                                                  |

#### Deployment — 一个可调度的上游节点

- **类**：`Deployment`（`litellm/types/router.py:370`）
- **字段**：
  - `model_name: str` — 对外暴露的模型组别名（e.g. `gpt-4`）
  - `litellm_params: LiteLLM_Params` — 调用 provider 所需的全部参数（`:264`）
  - `model_info: ModelInfo` — 元数据（定价、上下文长度、能力标记，`:94`）
- **关联**：多个 Deployment 共享同一个 `model_name` 构成一个"模型组"；Router 在组内做负载均衡，组间做 fallback。

#### LiteLLM_Params — 调用参数聚合

层次：`CredentialLiteLLMParams`（鉴权凭证）+ `CustomPricingLiteLLMParams`（自定义定价）→ `GenericLiteLLMParams`（:168）→ `LiteLLM_Params`（:264）。覆盖 `api_key`、`api_base`、`api_version`、`timeout`、`tpm`/`rpm`、`organization`、`vertex_project` 等。

### 1.2 响应对象体系（OpenAI 兼容）

全部定义在 `litellm/types/utils.py`，继承 `OpenAIObject`，保证与 OpenAI SDK schema 1:1 对齐：


| 对象                              | 行号                     | 说明                                                                                                           |
| --------------------------------- | ------------------------ | -------------------------------------------------------------------------------------------------------------- |
| `ModelResponse`                   | :1867                    | 非流式 completion 响应根对象                                                                                   |
| `ModelResponseBase`               | :1760                    | `id` / `created` / `model` / `object` / `system_fingerprint` 基类                                              |
| `ModelResponseStream`             | :1791                    | 流式 chunk 根对象，含`Delta`                                                                                   |
| `Choices`                         | :1369                    | `message`, `finish_reason`, `index`, `logprobs`                                                                |
| `StreamingChoices`                | :1698                    | 流式版 Choices，含`delta`                                                                                      |
| `Message`                         | :1126                    | `role`, `content`, `tool_calls`, `function_call`, `audio`, `reasoning_content`                                 |
| `Delta`                           | :1256                    | 流式增量版 Message                                                                                             |
| `Usage`                           | DeploymentTypedDict:1521 | `prompt_tokens`, `completion_tokens`, `total_tokens` + `*_tokens_details`（含 cache / reasoning / audio 拆分） |
| `EmbeddingResponse` / `Embedding` | :1972, :1990             | 向量响应                                                                                                       |
| `ImageResponse` / `ImageUsage`    | :2305, :2288             | 图像响应                                                                                                       |
| `TranscriptionResponse`           | :2404                    | 转写响应                                                                                                       |
| `TextCompletionResponse`          | :2116                    | Legacy completions 响应                                                                                        |

**关联**：`ModelResponse.choices[i].message` → `Message`，`ModelResponse.usage` → `Usage`；流式通过 `CustomStreamWrapper` 把 provider 原生 chunk → `ModelResponseStream`。

### 1.3 CustomStreamWrapper — 流式统一封装

`litellm/litellm_core_utils/streaming_handler.py:98`。包装任意 provider 的流式迭代器，逐 chunk 转 `ModelResponseStream`，并在最后汇总 usage、触发 `Logging.success_handler`。

### 1.4 BaseLLM / BaseConfig / BaseLLMException — Provider 抽象

- `BaseLLM`（`litellm/llms/base.py:13`）— 旧式 provider 基类
- `BaseConfig`（`litellm/llms/base_llm/chat/transformation.py:81`）— 新式 transformation 基类，规定 `map_openai_params` / `transform_request` / `transform_response` 三段式
- `BaseLLMException`（同文件 :50）— provider 异常基类，落地到 `APIError` 体系
- `BaseLLMHTTPHandler`（`llms/custom_httpx/llm_http_handler.py:153`）— 统一 httpx 调用流程

### 1.5 Cache 体系

- `Cache`（`litellm/caching/caching.py:55`）— 外部暴露入口，通过 `type` 选择后端
- 后端（均继承 `BaseCache`）：
  - `InMemoryCache`（`in_memory_cache.py:27`）
  - `RedisCache`（`redis_cache.py:182`）
  - `DualCache`（`dual_cache.py:51`）— 内存 + Redis 两级
  - `S3Cache`（`s3_cache.py:23`）/ `DiskCache`（`disk_cache.py:14`）/ `QdrantSemanticCache`（`qdrant_semantic_cache.py:24`）
- **关联**：Router 内部的 `cooldown_cache`、`failed_calls`、`user_api_key_cache`（Proxy 层）均基于 `DualCache`。

### 1.6 CustomLogger & Logging — 回调与审计


| 对象                                | 位置                                                | 作用                                                                                      |
| ----------------------------------- | --------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `CustomLogger`                      | `litellm/integrations/custom_logger.py:68`          | 用户自定义回调基类，实现`log_success_event` / `log_failure_event` / `async_log_*` 等 hook |
| `Logging`                           | `litellm/litellm_core_utils/litellm_logging.py:290` | 每个请求的 logging 上下文对象，贯穿 pre/post call                                         |
| `StandardLoggingPayload`            | 同文件 :4586+ (`StandardLoggingPayloadSetup`)       | 标准化日志载荷，含`metadata`、`hidden_params`、`model_information`、`response_cost` 等    |
| `StandardLoggingUserAPIKeyMetadata` | `types/utils.py:2503`                               | 记录调用方 key/team/user 信息                                                             |

**关联**：Logging 对象在 `completion()` 入口创建，贯穿 Router/Provider 调用全过程；终态落地 `StandardLoggingPayload` 并分发给所有注册的 `CustomLogger` 子类（Prometheus / OTel / Langfuse / Slack / Datadog / S3 等）。

### 1.7 异常层级（`litellm/exceptions.py`）

均继承自 `openai.*` 官方异常，保证 SDK 调用侧异常兼容：

```
APIError (:656)
├── APIConnectionError (:697)
├── APIResponseValidationError (:736)
│     └── JSONSchemaValidationError (:773)
BadRequestError (:124)
├── ImageFetchError (:175)
├── ContextWindowExceededError (:376)   ← context_window_fallbacks 触发源
├── ContentPolicyViolationError (:460)  ← content_policy_fallbacks 触发源
├── RejectedRequestError (:418)
├── UnsupportedParamsError (:793)
└── InvalidRequestError (:859)
AuthenticationError (:33)
NotFoundError (:79)
PermissionDeniedError (:284)
UnprocessableEntityError (:199)
RateLimitError (:323)        ← cooldown 主触发源（429）
ServiceUnavailableError (:505)
  └── MidStreamFallbackError (:942)    ← 流式中途降级专用
BadGatewayError (:555)
InternalServerError (:605)
BudgetExceededError (:844)             ← Proxy 预算耗尽
BlockedPiiEntityError (:927)
GuardrailInterventionNormalStringError (:1019)
```

---

## 2. Proxy 数据库域对象（schema.prisma）

### 2.1 组织 / 团队 / 用户三层架构

```
LiteLLM_OrganizationTable (:82)
    │ 1 ─ N
    ├── LiteLLM_TeamTable (:116)
    │        │ 1 ─ N
    │        ├── LiteLLM_ProjectTable (:156)  ← 新增层，Team 与 Key 之间
    │        ├── LiteLLM_TeamMembership (:609) ← User ↔ Team 多对多
    │        └── LiteLLM_ModelTable (:104)     ← 团队级 model 别名
    │
    ├── LiteLLM_UserTable (:230)
    │        ├── LiteLLM_OrganizationMembership (:619)
    │        └── LiteLLM_InvitationLink (:640)
    │
    └── LiteLLM_EndUserTable (:512)            ← 终端用户（customer）
```

核心字段速查：


| 表                               | 主键                         | 关键字段                                                                                                                       | 关联                                                         |
| -------------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------ |
| `LiteLLM_OrganizationTable`      | `organization_id`            | `organization_alias`, `budget_id`, `models`, `spend`, `object_permission_id`                                                   | 持有多个 Team / User / Key；1 个 Budget                      |
| `LiteLLM_TeamTable`              | `team_id`                    | `team_alias`, `organization_id`, `members_with_roles`, `models`, `max_budget`, `tpm/rpm_limit`, `access_group_ids`, `model_id` | 属于 Org；含多 Key / Project；可选关联自定义 ModelTable 别名 |
| `LiteLLM_ProjectTable`           | `project_id`                 | `project_alias`, `team_id`, `budget_id`, `models`, `model_spend`                                                               | 属于 Team，是 Key 的直接父级容器                             |
| `LiteLLM_UserTable`              | `user_id`                    | `sso_user_id`, `organization_id`, `teams[]`, `user_role`, `models`, `max_budget`, `object_permission_id`                       | 属于 Org；可加入多 Team                                      |
| `LiteLLM_EndUserTable`           | `user_id`                    | `alias`, `allowed_model_region`, `default_model`, `budget_id`, `blocked`                                                       | 面向终端用户计费隔离                                         |
| `LiteLLM_TeamMembership`         | `(user_id, team_id)`         | `spend`, `budget_id`                                                                                                           | 记录 User 在某 Team 内的独立预算/花费                        |
| `LiteLLM_OrganizationMembership` | `(user_id, organization_id)` | `user_role`, `spend`, `budget_id`                                                                                              | 同上，组织级                                                 |

### 2.2 API Key (VerificationToken) 域

`LiteLLM_VerificationToken`（:360）是 Proxy 最核心的鉴权主体：

- **主键**：`token`（hash 后的 virtual key）
- **外键**：`user_id`, `team_id`, `project_id`, `agent_id`, `organization_id`, `budget_id`, `object_permission_id`
- **配额**：`max_budget`, `spend`, `tpm_limit`, `rpm_limit`, `max_parallel_requests`, `budget_duration`, `budget_reset_at`
- **权限**：`models[]`, `allowed_routes[]`, `policies[]`, `access_group_ids[]`, `permissions{}`
- **轮换**：`rotation_count`, `auto_rotate`, `rotation_interval`, `key_rotation_at`, `last_rotation_at`
- **关联表**：
  - `LiteLLM_JWTKeyMapping`（:420）— JWT claim → virtual key 的映射
  - `LiteLLM_DeprecatedVerificationToken`（:439）— 轮换期旧 key 宽限
  - `LiteLLM_DeletedVerificationToken`（:452）— 审计归档（保留 spend）

### 2.3 模型 & 凭证 域


| 表                                 | 作用                                                                                                                               |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `LiteLLM_ProxyModelTable` (:48)    | 真正存储的"模型 deployment"（对应 SDK 侧的`Deployment`），字段 `model_name`/`litellm_params`/`model_info` 与 `Deployment` 完全对应 |
| `LiteLLM_CredentialsTable` (:36)   | 可复用凭证池（`credential_name` 唯一），供多个 Model 引用                                                                          |
| `LiteLLM_ModelTable` (:104)        | 团队级 model 别名表，一个 Team 最多一张                                                                                            |
| `LiteLLM_AgentsTable` (:61)        | Agent（A2A）配置，独立于 Model                                                                                                     |
| `LiteLLM_AccessGroupTable` (:1187) | 统一访问组：同时约束 Model / MCP / Agent 的成员 + 受让 Team/Key                                                                    |

### 2.4 预算 & 权限域

- `LiteLLM_BudgetTable`（:12）— 单表预算模板，被 **Org/Team/Project/Key/EndUser/Tag/Membership** 共享外键引用
- `LiteLLM_ObjectPermissionTable`（:266）— 统一对象权限（MCP servers、vector stores、agents、models、tools），被 Team/User/Key/Org/EndUser 引用

### 2.5 MCP 域（Model Context Protocol）


| 表                                  | 作用                                                                                     |
| ----------------------------------- | ---------------------------------------------------------------------------------------- |
| `LiteLLM_MCPServerTable` (:287)     | MCP Server 注册表（URL、transport=`sse`/`http`/`stdio`、auth_type、credentials、health） |
| `LiteLLM_MCPToolsetTable` (:336)    | Toolset = 多个`{server_id, tool_name}` 的命名集合                                        |
| `LiteLLM_MCPUserCredentials` (:348) | 用户级 BYOK 凭证；OAuth 凭证也落这张表（靠 JSON 内`type` 字段区分，见 `CLAUDE.md` 指引） |

### 2.6 消费（Spend）与观测

**实时粒度**：

- `LiteLLM_SpendLogs`（:546）— 每次请求一条，字段含 `api_key`(hash)/`model`/`model_group`/`team_id`/`user`/`end_user`/`request_tags`/`mcp_namespaced_tool_name`/`session_id`/`messages`/`response`
- `LiteLLM_ErrorLogs`（:586）— 异常请求日志

**每日汇总**（6 张 `Daily*Spend`）：


| 表                                      | 主维度            |
| --------------------------------------- | ----------------- |
| `LiteLLM_DailyUserSpend` (:672)         | `user_id`         |
| `LiteLLM_DailyTeamSpend` (:794)         | `team_id`         |
| `LiteLLM_DailyOrganizationSpend` (:703) | `organization_id` |
| `LiteLLM_DailyEndUserSpend` (:734)      | `end_user_id`     |
| `LiteLLM_DailyAgentSpend` (:764)        | `agent_id`        |
| `LiteLLM_DailyTagSpend` (:825)          | `tag`             |

统一副维度：`(date, api_key, model, custom_llm_provider, mcp_namespaced_tool_name, endpoint)`，记录 prompt/completion/cache token、请求数、成功/失败数。

### 2.7 Guardrails & Policy 域


| 表                                                                            | 作用                                                                          |
| ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `LiteLLM_GuardrailsTable` (:938)                                              | Guardrail 配置（provider params、审批状态）                                   |
| `LiteLLM_PolicyTable` (:1122)                                                 | 策略，支持版本 (`version_number`/`version_status`=draft/published/production) |
| `LiteLLM_PolicyAttachmentTable` (:1147)                                       | 策略附着点（scope / teams / keys / models / tags）                            |
| `LiteLLM_ToolTable` (:1162)                                                   | 工具注册表，`input_policy`/`output_policy`=trusted/untrusted/blocked          |
| `LiteLLM_DailyGuardrailMetrics` (:956) / `LiteLLM_DailyPolicyMetrics` (:974)  | 每日 pass/block/flag 计数 + 平均分 & 延迟                                     |
| `LiteLLM_SpendLogGuardrailIndex` (:992) / `LiteLLM_SpendLogToolIndex` (:1004) | 反查某 guardrail/tool 的最近 N 条请求                                         |

### 2.8 Managed Resources & 杂项

- **Managed Files / Objects / VectorStores / Index**：`LiteLLM_ManagedFileTable`、`LiteLLM_ManagedObjectTable`（batches/fine-tuning）、`LiteLLM_ManagedVectorStoreTable`、`LiteLLM_ManagedVectorStoresTable`、`LiteLLM_ManagedVectorStoreIndexTable`
- **Config**：`LiteLLM_Config`（kv）、`LiteLLM_CacheConfig`、`LiteLLM_UISettings`、`LiteLLM_ConfigOverrides`、`LiteLLM_SSOConfig`
- **运维**：`LiteLLM_CronJob`（单 pod 选主租约）、`LiteLLM_HealthCheckTable`、`LiteLLM_AuditLog`、`LiteLLM_UserNotifications`、`LiteLLM_InvitationLink`
- **Prompt / Skill / Plugin**：`LiteLLM_PromptTable`（版本化 prompt）、`LiteLLM_SkillsTable`（Anthropic Skills 托管）、`LiteLLM_ClaudeCodePluginTable`（Plugin Marketplace）、`LiteLLM_SearchToolsTable`
- **软删除归档**：`LiteLLM_DeletedTeamTable`、`LiteLLM_DeletedVerificationToken`、`LiteLLM_DeprecatedVerificationToken`
- **Tag**：`LiteLLM_TagTable`（可挂 budget）

---

## 3. Proxy 运行时域对象（全局单例）

启动时在 `litellm/proxy/proxy_server.py` 实例化，跨请求共享：


| 全局对象               | 行号  | 类型                                             | 职责                                                           |
| ---------------------- | ----- | ------------------------------------------------ | -------------------------------------------------------------- |
| `prisma_client`        | :1536 | `Optional[PrismaClient]` (`proxy/utils.py:2355`) | DB 网关，封装所有 Prisma 模型操作（禁止写 raw SQL）            |
| `llm_router`           | :1527 | `Optional[Router]`                               | 承载全部 proxy 模型的 Router 单例                              |
| `llm_model_list`       | :1528 | `Optional[list]`                                 | 规整后的 model_list（DB + YAML 合并）                          |
| `general_settings`     | :1529 | `dict`                                           | `config.yaml` → `general_settings:` 全局设置                  |
| `master_key`           | :1533 | `Optional[str]`                                  | 最高权限 root key                                              |
| `user_api_key_cache`   | :1540 | `DualCache`                                      | 缓存验证过的 UserAPIKeyAuth，减少 DB 往返                      |
| `user_custom_auth`     | :1558 | Callable / None                                  | 用户自定义鉴权钩子                                             |
| `proxy_logging_obj`    | :1591 | `ProxyLogging`（`proxy/utils.py:303`）           | 编排 pre/post call hooks、并发限流、预算限流、Slack/Email 告警 |
| `premium_user`         | :639  | `bool`                                           | 企业版 license 是否生效                                        |
| `health_check_results` | :1573 | `dict`                                           | 各模型的最近健康检查结果                                       |

### ProxyLogging 关键子组件（构造于 :303）

- `user_api_key_cache: DualCache`（注入）
- `internal_usage_cache: InternalUsageCache`（内部 1s TTL 的 DualCache）
- `max_parallel_request_limiter: _PROXY_MaxParallelRequestsHandler` — 并发 quota
- `max_budget_limiter: _PROXY_MaxBudgetLimiter` — 预算硬卡
- `cache_control_check: _PROXY_CacheControlCheck`
- `slack_alerting_instance: SlackAlerting` — 告警
- `email_logging_instance` — 邮件告警

---

## 4. 认证聚合对象：`UserAPIKeyAuth`

位置：`litellm/proxy/_types.py:2478`，继承链：

```
LiteLLM_VerificationToken (:360 schema)
  └── LiteLLM_VerificationTokenView (:2406)   ← 合并 Team 字段
        └── UserAPIKeyAuth (:2478)             ← 再叠加 User/Org/EndUser/Project/JWT
```

它是 Proxy 里"请求身份"的**唯一真相源**，在一次鉴权流程中由 `auth/` 模块从 DB / 缓存 / JWT 中加载并合并：

- **Key 字段**：`api_key`（hash）、`models`、`spend`、`max_budget`、`tpm/rpm_limit`
- **Team 字段**（来自 Join）：`team_alias`, `team_spend`, `team_tpm_limit`, `team_rpm_limit`, `team_max_budget`, `team_models`, `team_blocked`, `team_member`, `team_member_tpm_limit` ...
- **End User 字段**：`end_user_id`, `end_user_max_budget`, `end_user_rpm_limit` ...
- **Org 字段**：`organization_alias`, `organization_max_budget`, `organization_tpm_limit` ...
- **Project 字段**：`project_alias`, `project_metadata`
- **User 字段**：`user_role: LitellmUserRoles`, `user_email`, `user_spend`, `user_max_budget`, `user` (expanded)
- **权限**：`object_permission`, `end_user_object_permission`
- **JWT**：`jwt_claims: Dict` — 上游 IdP 的 claim，传递给 MCP JWTSigner 等下游 guardrail

鉴权成功后，该对象通过 FastAPI `Depends` 贯穿整个请求链路，**每一次限流/预算/权限判定都只认这个对象**。

---

## 5. 关键关联关系汇总

### 5.1 组织层级

```
Organization 1 ─ N Team 1 ─ N Project 1 ─ N VerificationToken
             1 ─ N User      1 ─ N (via TeamMembership) User
             1 ─ N VerificationToken (组织级 key)
User  N ─ N Team        (through LiteLLM_TeamMembership)
User  N ─ N Org         (through LiteLLM_OrganizationMembership)
```

### 5.2 SDK ↔ Proxy 模型对齐

```
SDK:   Deployment(model_name, litellm_params, model_info)
       │
       │ 启动时: ProxyConfig.load_router_config() 将 DB 行转成 DeploymentTypedDict
       ↓
Proxy: LiteLLM_ProxyModelTable(model_id, model_name, litellm_params JSON, model_info JSON)
       │
       │ LiteLLM_CredentialsTable (可选, credential_name 复用)
       │ LiteLLM_ModelTable       (团队级别名, @unique team_id)
       │ LiteLLM_AccessGroupTable (跨团队统一授权)
```

### 5.3 权限合并链（请求时）

```
Request
  │ Bearer sk-xxx
  ↓
LiteLLM_VerificationToken (key)
  ├── models[]                ┐
  ├── object_permission_id ───┼─→ LiteLLM_ObjectPermissionTable (MCP/VS/Agents/Models/Tools)
  ├── team_id ────────────────┼─→ LiteLLM_TeamTable.models + object_permission
  ├── project_id ─────────────┼─→ LiteLLM_ProjectTable.models
  ├── organization_id ────────┼─→ LiteLLM_OrganizationTable.models + object_permission
  └── user_id ────────────────┘   LiteLLM_UserTable.models + object_permission
        ↓
        合并为 UserAPIKeyAuth  (权限求交集；预算/配额取最严)
        ↓
   Router / Model / Guardrail 通行决策
```

### 5.4 Spend 写入链

```
Request 完成
   │ LiteLLM_Logging.success_handler
   ↓
StandardLoggingPayload  (含 spend / token 拆分 / cache_hit)
   ↓
┌─────────────────────────────────────┐
│  LiteLLM_SpendLogs (实时, 1 行/请求) │
│  LiteLLM_DailyUserSpend             │  ← UPSERT by
│  LiteLLM_DailyTeamSpend             │     (主体_id, date, api_key, model,
│  LiteLLM_DailyOrganizationSpend     │      custom_llm_provider,
│  LiteLLM_DailyEndUserSpend          │      mcp_namespaced_tool_name, endpoint)
│  LiteLLM_DailyAgentSpend            │
│  LiteLLM_DailyTagSpend              │
└─────────────────────────────────────┘
   │
   └─→ 同时自增 spend 到: VerificationToken.spend / Team.spend / User.spend / Org.spend / EndUser.spend
         若 spend > max_budget → BudgetExceededError（下次请求拒绝）
```

### 5.5 MCP 链路

```
LiteLLM_MCPServerTable (全局/租户级 MCP server 注册表)
   │
   ├─ allowed_tools / tool_name_to_display_name
   │
   ├─ LiteLLM_MCPToolsetTable (命名 toolset = N 个 {server_id, tool_name})
   │       │
   │       └─→ 挂到 LiteLLM_ObjectPermissionTable.mcp_toolsets[]
   │
   ├─ LiteLLM_ObjectPermissionTable.mcp_servers[] / mcp_access_groups[]
   │       ↑ 被 Key/Team/User/Org 引用
   │
   └─ LiteLLM_MCPUserCredentials (per-user BYOK + OAuth, 一张表两种 type)
```

### 5.6 Fallback / Cooldown 关联（参见 `router/cooldown_handlers_analysis.md`）

```
Router.fallbacks / context_window_fallbacks / content_policy_fallbacks
          ↑ 使用
Router.async_function_with_fallbacks
          ↓ 失败时
_set_cooldown_deployments  →  CooldownCache (DualCache 后端)
          ↓
下次 pick_deployment 时从 Router.model_list 剔除冷却节点
```

---

## 6. 速查：文件 → 领域对象映射


| 文件                                                              | 核心对象                                                                                                                                                                                                                   |
| ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `litellm/router.py`                                               | `Router`, `RoutingArgs`                                                                                                                                                                                                    |
| `litellm/types/router.py`                                         | `Deployment`, `LiteLLM_Params`, `ModelInfo`, `RetryPolicy`, `AllowedFailsPolicy`, `RouterGeneralSettings`, `RoutingStrategy`                                                                                               |
| `litellm/types/utils.py`                                          | `ModelResponse`, `ModelResponseStream`, `Choices`, `StreamingChoices`, `Message`, `Delta`, `Usage`, `EmbeddingResponse`, `ImageResponse`, `StandardLoggingPayload*`                                                        |
| `litellm/exceptions.py`                                           | 全部异常层级                                                                                                                                                                                                               |
| `litellm/caching/*.py`                                            | `Cache`, `DualCache`, `RedisCache`, `InMemoryCache`, `S3Cache`, `DiskCache`, `QdrantSemanticCache`                                                                                                                         |
| `litellm/integrations/custom_logger.py`                           | `CustomLogger`                                                                                                                                                                                                             |
| `litellm/litellm_core_utils/litellm_logging.py`                   | `Logging`, `StandardLoggingPayloadSetup`                                                                                                                                                                                   |
| `litellm/litellm_core_utils/streaming_handler.py`                 | `CustomStreamWrapper`                                                                                                                                                                                                      |
| `litellm/llms/base.py` + `llms/base_llm/chat/transformation.py`   | `BaseLLM`, `BaseConfig`, `BaseLLMException`                                                                                                                                                                                |
| `litellm/router_utils/cooldown_handlers.py` + `cooldown_cache.py` | 冷却判定 / 缓存                                                                                                                                                                                                            |
| `litellm/router_utils/fallback_event_handlers.py`                 | `run_async_fallback`, `get_fallback_model_group`                                                                                                                                                                           |
| `litellm/proxy/schema.prisma`                                     | 所有`LiteLLM_*` 数据库实体                                                                                                                                                                                                 |
| `litellm/proxy/_types.py`                                         | `UserAPIKeyAuth`, `LiteLLM_VerificationTokenView`, `LiteLLM_TeamTable`, `LiteLLM_UserTable`, `LiteLLM_OrganizationTable`, `LiteLLM_ProxyModelTable`, `LiteLLM_MCPServerTable`, `NewUserRequest`, `GenerateKeyRequest`, ... |
| `litellm/proxy/utils.py`                                          | `ProxyLogging`, `PrismaClient`                                                                                                                                                                                             |
| `litellm/proxy/proxy_server.py`                                   | 全局单例（`llm_router`, `prisma_client`, `user_api_key_cache`, `proxy_logging_obj` 等）                                                                                                                                    |

---

## 7. 一句话总结

- **SDK 层核心**：`Router` 编排一组 `Deployment`，每个 `Deployment` 通过 `LiteLLM_Params` 描述如何调用一个 provider；所有结果统一成 `ModelResponse`（或其流式/嵌入/图像变体），经 `CustomStreamWrapper` / `Logging` / `Cache` 穿透，异常全部走 OpenAI 兼容的层级。
- **Proxy 层核心**：数据库用三层级（Org → Team → User/Project/Key）承载多租户身份与配额，`LiteLLM_VerificationToken` 是请求入口，`UserAPIKeyAuth` 是请求期身份聚合对象；`LiteLLM_ProxyModelTable` 反向映射 SDK 的 `Deployment`；消费通过 `LiteLLM_SpendLogs` + 六张 `Daily*Spend` 表沉淀；运行时靠 `prisma_client / llm_router / proxy_logging_obj / user_api_key_cache` 四大全局单例协同。
