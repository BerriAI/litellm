# LiteLLM 安全功能全览

本文档记录 LiteLLM 项目中所有安全相关的功能模块，涵盖认证鉴权、访问控制、内容安全、数据保护、速率限制、审计日志和密钥管理。

---

## 一、认证与鉴权（Authentication & Authorization）

### 1.1 API Key 认证

**入口**: `litellm/proxy/auth/user_api_key_auth.py`

LiteLLM Proxy 的核心认证机制。每个请求必须携带有效的 API Key（Bearer Token），系统会：

1. 从请求头提取 Bearer Token
2. 对 Token 进行 SHA-256 哈希（`hash_token`，`litellm/proxy/_types.py:213`）
3. 在缓存/数据库中查找对应的 `LiteLLM_VerificationToken` 记录
4. 验证 Token 有效期（`expires` 字段）
5. 检查 Token 是否属于有效的 Team / Organization
6. 执行预算、模型权限等检查

**数据库表**: `LiteLLM_VerificationToken`（`schema.prisma:360`）

### 1.2 JWT 认证

**入口**: `litellm/proxy/auth/handle_jwt.py` — `JWTHandler` 类

支持基于 JWT 的认证，适用于与企业 IdP（Identity Provider）集成的场景。

- **支持算法**: RS256, RS384, RS512, PS256, PS384, PS512, ES256, ES384, ES512, EdDSA
- **JWKS 支持**: 从 IdP 的 JWKS 端点动态获取公钥
- **角色映射**: JWT 中的 scope/role 映射到 LiteLLM 内部角色
- **Token 中的声明**: `sub`（用户 ID）、`scope`（权限范围）、组织/团队信息

配置项：`LiteLLM_JWTAuth` 类定义了 JWT 认证参数。

### 1.3 OAuth2 认证

**入口**: `litellm/proxy/auth/oauth2_check.py` — `Oauth2Handler` 类

支持标准 OAuth2 Token 内省（RFC 7662）：

- **Token Info 端点**: GET 方式查询 Token 有效性
- **Token Introspection 端点**: POST 方式（需 client_id + client_secret）
- 支持 Basic Auth 和 Bearer Token 方式发送凭证

### 1.4 SSO 单点登录

**入口**: `litellm/proxy/management_endpoints/ui_sso.py`

支持多种 SSO 提供商用于 UI Dashboard 登录：

| 提供商 | 环境变量 |
|--------|----------|
| Google | `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` |
| Microsoft | `MICROSOFT_CLIENT_ID` / `MICROSOFT_CLIENT_SECRET` |
| Generic OIDC | `GENERIC_CLIENT_ID` / `GENERIC_CLIENT_SECRET` / `GENERIC_AUTHORIZATION_ENDPOINT` |

流程：
1. `/sso/key/generate` — 发起 SSO 登录
2. IdP 认证后回调 `/sso/callback`
3. 生成短期 JWT，重定向到 UI Dashboard
4. 支持 `auto_redirect_ui_login_to_sso` 配置自动跳转

**自定义 SSO**: `litellm/proxy/custom_sso.py` 提供 `custom_sso_handler` 钩子。

### 1.5 Master Key

Proxy 启动时需要设置 `LITELLM_MASTER_KEY` 环境变量，作为管理员级别的超级密钥，用于：

- 创建/管理虚拟 API Key
- 访问所有管理端点
- 加解密其他密钥

---

## 二、RBAC 访问控制

### 2.1 角色体系

**定义**: `litellm/proxy/_types.py:96` — `LitellmUserRoles` 枚举

| 角色 | 枚举值 | 权限描述 |
|------|--------|----------|
| **Proxy Admin** | `proxy_admin` | 平台最高管理员，拥有所有权限 |
| **Proxy Admin (View Only)** | `proxy_admin_viewer` | 可查看所有 Key 和花费，但不能修改 |
| **Org Admin** | `org_admin` | 组织管理员，管理组织内的 Team 和 User |
| **Internal User** | `internal_user` | 内部用户，管理自己的 Key 和花费 |
| **Internal User (View Only)** | `internal_user_viewer` | 仅查看自己的 Key 和花费 |
| **Team Admin** | `admin`（Team 级） | 团队管理员 |
| **Team Member** | `user`（Team 级） | 团队普通成员 |

### 2.2 路由权限控制

**入口**: `litellm/proxy/auth/route_checks.py` — `RouteChecks` 类

- **基于角色的路由**: 不同角色可访问的 API 路由不同
- **Virtual Key 路由限制**: 每个 Key 可配置 `allowed_routes`，限制只能调用特定端点
- **路由分类**: `LiteLLMRoutes`（`_types.py:251`）定义了 OpenAI 路由、管理路由、信息路由等分类

### 2.3 模型访问控制

**入口**: `litellm/proxy/auth/model_checks.py`、`litellm/proxy/auth/auth_checks.py`

- **Key 级别模型限制**: `LiteLLM_VerificationToken.models` 字段限制 Key 可调用的模型
- **Team 级别模型限制**: `LiteLLM_TeamTable.models` 字段
- **模型访问组（Access Group）**: 将多个模型打包成一个组，Key/Team 只需授权组名
- **`can_key_call_model`**: 核心检查函数，验证 Key 是否有权调用目标模型

### 2.4 组织层级权限

**入口**: `litellm/proxy/auth/auth_checks_organization.py`

- Organization → Team → User 的层级结构
- Org Admin 可管理其下所有 Team 和 User
- 组织级别的预算和速率限制

---

## 三、Guardrails（安全防护栏）

### 3.1 架构概述

**入口**: `litellm/proxy/guardrails/`

Guardrail 在请求处理的不同阶段介入：

- **Pre-call（调用前）**: 检查输入内容
- **During-call（调用中）**: 流式处理时检查
- **Post-call（调用后）**: 检查输出内容

### 3.2 内置 Guardrail

| Guardrail | 目录/文件 | 功能 |
|-----------|-----------|------|
| **Prompt Injection 检测** | `hooks/prompt_injection_detection.py` | 基于相似度匹配检测 Prompt 注入攻击 |
| **LiteLLM 内容过滤器** | `guardrail_hooks/litellm_content_filter/` | 内置内容审核，支持自定义策略模板 |
| **代码执行阻止** | `guardrail_hooks/block_code_execution/` | 阻止 LLM 在响应中执行代码（如 tool_calls 中的代码） |
| **工具权限控制** | `guardrail_hooks/tool_permission.py` | 控制哪些 tool/function 允许被调用 |
| **工具策略** | `guardrail_hooks/tool_policy/` | 基于策略的 tool 调用控制 |
| **MCP 安全** | `guardrail_hooks/mcp_security/` | MCP Server 调用的安全防护 |
| **MCP 终端用户权限** | `guardrail_hooks/mcp_end_user_permission/` | 终端用户对 MCP 工具的权限控制 |
| **MCP JWT 签名** | `guardrail_hooks/mcp_jwt_signer/` | MCP 请求的 JWT 签名验证 |
| **PII 脱敏（Presidio）** | `guardrail_hooks/presidio.py` | 基于 Microsoft Presidio 的个人身份信息识别与脱敏 |
| **自定义代码** | `guardrail_hooks/custom_code/` | 用户自定义 Python 代码作为 Guardrail |
| **统一 Guardrail** | `guardrail_hooks/unified_guardrail/` | 统一的 Guardrail 抽象层 |
| **Semantic Guard** | `guardrail_hooks/semantic_guard/` | 基于语义的内容安全路由 |

### 3.3 第三方 Guardrail 集成

| 提供商 | 目录 | 描述 |
|--------|------|------|
| **Lakera AI** | `guardrail_hooks/lakera_ai.py`、`lakera_ai_v2.py` | Prompt 注入检测 |
| **Azure Content Safety** | `guardrail_hooks/azure/` | Azure 内容安全（Prompt Shield + Text Moderation） |
| **AWS Bedrock Guardrails** | `guardrail_hooks/bedrock_guardrails.py` | AWS Bedrock 内置安全防护 |
| **Google Model Armor** | `guardrail_hooks/model_armor/` | Google Cloud 模型安全防护 |
| **OpenAI Moderation** | `guardrail_hooks/openai/` | OpenAI 内容审核 API |
| **Aporia AI** | `guardrail_hooks/aporia_ai/` | AI 安全与合规平台 |
| **Prompt Security** | `guardrail_hooks/prompt_security/` | Prompt 安全检测 |
| **Pangea** | `guardrail_hooks/pangea/` | 安全服务平台（PII、恶意内容检测） |
| **CrowdStrike AIDR** | `guardrail_hooks/crowdstrike_aidr/` | CrowdStrike AI 检测与响应 |
| **HiddenLayer** | `guardrail_hooks/hiddenlayer/` | AI 模型安全（对抗攻击检测） |
| **Pillar** | `guardrail_hooks/pillar/` | AI 安全平台 |
| **Guardrails AI** | `guardrail_hooks/guardrails_ai/` | Guardrails AI 框架集成 |
| **Noma** | `guardrail_hooks/noma/` | AI 安全监控 |
| **DynamoAI** | `guardrail_hooks/dynamoai/` | AI 内容安全 |
| **EnkryptAI** | `guardrail_hooks/enkryptai/` | AI 安全检测 |
| **Qualifire** | `guardrail_hooks/qualifire/` | AI 输出质量和安全 |
| **GraySwan** | `guardrail_hooks/grayswan/` | AI 安全评估 |
| **IBM Guardrails** | `guardrail_hooks/ibm_guardrails/` | IBM AI 安全检测器 |
| **Lasso** | `guardrail_hooks/lasso/` | AI 安全防护 |
| **Javelin** | `guardrail_hooks/javelin/` | AI 安全网关 |
| **AIM** | `guardrail_hooks/aim/` | AI 安全监控 |
| **Akto** | `guardrail_hooks/akto/` | API 安全测试 |
| **Onyx** | `guardrail_hooks/onyx/` | AI 安全平台 |
| **Palo Alto Prisma AIRS** | `guardrail_hooks/panw_prisma_airs/` | Palo Alto AI 运行时安全 |
| **Zscaler AI Guard** | `guardrail_hooks/zscaler_ai_guard/` | Zscaler AI 安全网关 |

### 3.4 合规策略模板

**位置**: `guardrail_hooks/litellm_content_filter/policy_templates/`

| 模板 | 描述 |
|------|------|
| `eu_ai_act_article5.yaml` | EU AI Act 第五条全文 |
| `eu_ai_act_art5_biometric_profiling.yaml` | 生物识别分析禁令 |
| `eu_ai_act_art5_emotion_recognition.yaml` | 情感识别限制 |
| `eu_ai_act_art5_manipulation.yaml` | 操纵行为禁令 |
| `eu_ai_act_art5_social_scoring.yaml` | 社会评分禁令 |
| `eu_ai_act_art5_vulnerability.yaml` | 弱势群体保护 |
| `sg_mas_data_governance.yaml` | 新加坡 MAS 数据治理 |
| `sg_mas_fairness_bias.yaml` | 新加坡 MAS 公平性与偏见 |
| `sg_mas_human_oversight.yaml` | 新加坡 MAS 人工监督 |
| `sg_mas_model_security.yaml` | 新加坡 MAS 模型安全 |
| `sg_mas_transparency_explainability.yaml` | 新加坡 MAS 透明性与可解释性 |
| `prompt_injection.yaml` | Prompt 注入防护 |
| `airline_brand_protection.yaml` | 航空品牌保护 |
| `aviation_safety_topics.yaml` | 航空安全主题过滤 |

---

## 四、速率限制与资源控制

### 4.1 并发请求限制

**入口**: `litellm/proxy/hooks/parallel_request_limiter.py`、`parallel_request_limiter_v3.py`

- **`max_parallel_requests`**: 限制单个 Key 的最大并发请求数
- **`global_max_parallel_requests`**: Proxy 实例级别的全局并发限制
- 超限时直接返回 HTTP 429，响应头包含 `retry-after`

### 4.2 RPM / TPM 速率限制

| 限制维度 | 描述 |
|----------|------|
| Key RPM/TPM | 每个 API Key 的每分钟请求数/Token 数限制 |
| Model RPM/TPM | Key 对特定模型的 RPM/TPM 限制 |
| User RPM/TPM | 用户级别限制 |
| Team RPM/TPM | 团队级别限制 |
| End User RPM/TPM | 终端用户级别限制 |

### 4.3 动态速率限制

**入口**: `litellm/proxy/hooks/dynamic_rate_limiter.py`、`dynamic_rate_limiter_v3.py`

根据 Provider 返回的 `x-ratelimit-remaining-*` 响应头动态调整速率限制，避免触发上游 Provider 的限流。

### 4.4 预算限制

| Hook | 文件 | 描述 |
|------|------|------|
| **Max Budget Limiter** | `hooks/max_budget_limiter.py` | 用户级别的最大预算限制 |
| **Model Max Budget Limiter** | `hooks/model_max_budget_limiter.py` | 模型级别的最大预算限制 |
| **Max Budget Per Session** | `hooks/max_budget_per_session_limiter.py` | 单次会话的最大预算限制（适用于 Agent 场景） |
| **Max Iterations Limiter** | `hooks/max_iterations_limiter.py` | 单次会话的最大迭代次数限制（Agent 防失控） |

### 4.5 Batch 速率限制

**入口**: `litellm/proxy/hooks/batch_rate_limiter.py`

针对批量请求的速率限制。

---

## 五、数据保护

### 5.1 API Key 哈希存储

**函数**: `hash_token`（`litellm/proxy/_types.py:213`、`litellm/proxy/utils.py:4575`）

所有 API Key 在数据库中以 SHA-256 哈希值存储，永远不存储明文 Key。Key 仅在创建时返回一次明文。

### 5.2 加解密工具

**入口**: `litellm/proxy/common_utils/encrypt_decrypt_utils.py`

- `encrypt_value(value, signing_key)` — 使用 Fernet 对称加密
- `decrypt_value(value, signing_key)` — 解密
- 用于保护存储在数据库中的敏感配置（如 Provider API Key）

### 5.3 PII 脱敏

**入口**: `litellm/proxy/guardrails/guardrail_hooks/presidio.py`

基于 Microsoft Presidio 引擎：

- 识别请求/响应中的个人身份信息（姓名、邮箱、电话、身份证号等）
- 在发送给 LLM 之前自动脱敏
- 在返回给用户之前还原
- 支持自定义识别器（`example_presidio_ad_hoc_recognizer.json`）

### 5.4 消息日志控制

- **`turn_off_message_logging`**: 完全关闭请求/响应内容日志
- 启用后，`StandardLoggingPayload` 中的 `messages` 和 `response` 字段会被脱敏
- 可在 Key / Callback 级别单独配置

### 5.5 浏览器存储安全

- LiteLLM UI 仅使用 `sessionStorage`（不使用 `localStorage`）存储 Token
- `sessionStorage` 在浏览器关闭后自动清除，降低 XSS 攻击风险

---

## 六、密钥管理（Secret Management）

**入口**: `litellm/secret_managers/`

LiteLLM 集成多种密钥管理服务，用于安全存储和获取 Provider API Key 等敏感信息：

| 密钥管理器 | 文件 | 描述 |
|-----------|------|------|
| **AWS Secrets Manager** | `aws_secret_manager.py`、`aws_secret_manager_v2.py` | AWS 密钥管理服务 |
| **Google Secret Manager** | `google_secret_manager.py` | Google Cloud 密钥管理 |
| **Google KMS** | `google_kms.py` | Google Cloud KMS 加解密 |
| **HashiCorp Vault** | `hashicorp_secret_manager.py` | HashiCorp Vault 密钥管理 |
| **CyberArk** | `cyberark_secret_manager.py` | CyberArk 特权访问管理 |
| **Azure AD Token** | `get_azure_ad_token_provider.py` | Azure AD Token 获取 |
| **自定义** | `custom_secret_manager_loader.py` | 自定义密钥管理器加载 |
| **AWS RDS IAM Token** | `rds_iam_token.py`（auth 目录） | AWS RDS IAM 认证 |

---

## 七、网络安全

### 7.1 IP 白名单

**入口**: `litellm/proxy/auth/auth_utils.py`

- `allowed_ips` 配置项限制可访问 Proxy 的 IP 地址
- 在所有认证检查之前执行（`pre_db_read_auth_checks`）
- 管理端点: `GET /get/allowed_ips` 查看当前白名单

### 7.2 MCP IP 访问控制

**入口**: `litellm/proxy/auth/ip_address_utils.py` — `IPAddressUtils` 类

针对 MCP Server 的公网/内网访问控制：

- 内网 IP（10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 127.0.0.0/8）可看到所有 MCP Server
- 外网 IP 仅可看到 `available_on_public_internet=True` 的 Server
- 支持自定义内网 CIDR 范围
- 支持信任代理网络配置（用于 X-Forwarded-For 验证）

### 7.3 缓存控制检查

**入口**: `litellm/proxy/hooks/cache_control_check.py`

控制哪些 Key 可以使用缓存相关的参数，防止未授权的缓存操作。

---

## 八、Responses ID 安全

**入口**: `litellm/proxy/hooks/responses_id_security.py`

防止跨用户/跨团队的 Response ID 访问：

- 验证 Response ID 归属于当前请求的用户/团队
- 防止通过猜测 ID 获取其他用户的响应内容
- 可通过 `disable_responses_id_security` 配置关闭

---

## 九、审计日志

### 9.1 数据库审计日志

**数据库表**: `LiteLLM_AuditLog`（`schema.prisma:659`）

记录所有管理操作的变更审计：

| 字段 | 描述 |
|------|------|
| `changed_by` | 执行操作的用户 |
| `changed_by_api_key` | 使用的 API Key（哈希） |
| `action` | 操作类型: create / update / delete |
| `table_name` | 被操作的表（Team / User / Model 等） |
| `object_id` | 被操作对象的 ID |
| `before_value` | 变更前的值（JSON） |
| `updated_values` | 变更后的值（JSON） |

### 9.2 管理事件 Hook

- **Key 管理事件**: `hooks/key_management_event_hooks.py` — Key 创建/更新/删除时触发
- **User 管理事件**: `hooks/user_management_event_hooks.py` — 用户变更时触发
- 事件可推送到 Slack 等告警通道

### 9.3 Azure Sentinel 集成

**入口**: `litellm/integrations/azure_sentinel/`

将审计日志和请求日志推送到 Azure Sentinel（SIEM），用于企业级安全监控。

---

## 十、Prompt 注入防护

### 10.1 内置检测

**入口**: `litellm/proxy/hooks/prompt_injection_detection.py`

`_OPTIONAL_PromptInjectionDetection` 类实现了基于模式匹配的 Prompt 注入检测：

- **动词匹配**: Ignore, Disregard, Skip, Forget, Bypass 等
- **形容词匹配**: prior, previous, preceding, above 等
- **介词匹配**: 组合检测 "ignore the previous instructions" 等典型注入模式
- **相似度阈值**: 使用 `SequenceMatcher` 计算与已知注入模式的相似度
- **LLM 辅助检测**: 可选使用 LLM 对可疑内容做二次判断

### 10.2 第三方注入检测

- **Lakera AI**: 专业 Prompt 注入检测服务
- **Azure Prompt Shield**: Azure 原生的 Prompt 安全防护
- **Prompt Security**: 专业 Prompt 安全检测

---

## 十一、企业级安全功能

**目录**: `enterprise/`（需企业许可证）

| 功能 | 描述 |
|------|------|
| 自定义认证 | `enterprise_custom_auth` — 企业自定义认证逻辑 |
| 路由限制 | `EnterpriseRouteChecks` — 企业级路由权限控制 |
| 许可证验证 | `litellm/proxy/auth/litellm_license.py` — License 验证 |
| 高级 SSO | 企业级 SSO 配置和定制 |

---

## 十二、安全配置汇总

### YAML 配置示例

```yaml
general_settings:
  master_key: "sk-..."
  allowed_ips: ["10.0.0.0/8", "192.168.1.0/24"]
  global_max_parallel_requests: 1000

  # SSO
  auto_redirect_ui_login_to_sso: true

  # Responses 安全
  disable_responses_id_security: false

litellm_settings:
  # 速率限制
  max_parallel_requests: 100

  # Guardrails
  guardrails:
    - guardrail_name: "prompt-injection"
      litellm_params:
        guardrail: lakera
        mode: "pre_call"
    - guardrail_name: "pii-masking"
      litellm_params:
        guardrail: presidio
        mode: "pre_call"
    - guardrail_name: "content-filter"
      litellm_params:
        guardrail: litellm_content_filter
        mode: "pre_call"

  # 回调
  success_callback: ["langfuse"]
  callbacks: ["prometheus"]
```

### 环境变量

| 变量 | 描述 |
|------|------|
| `LITELLM_MASTER_KEY` | 管理员密钥 |
| `LITELLM_SECRET` | 数据库加密密钥 |
| `GOOGLE_CLIENT_ID` | Google SSO Client ID |
| `GOOGLE_CLIENT_SECRET` | Google SSO Client Secret |
| `MICROSOFT_CLIENT_ID` | Microsoft SSO Client ID |
| `GENERIC_CLIENT_ID` | 通用 OIDC Client ID |
| `JWT_PUBLIC_KEY_URL` | JWT 公钥 JWKS 端点 |

---

## 十三、安全 Hook 执行顺序

请求处理的安全检查链路：

```
请求到达
  |
  v
1. IP 白名单检查 (pre_db_read_auth_checks)
  |
  v
2. API Key / JWT / OAuth2 认证 (user_api_key_auth)
  |
  v
3. RBAC 角色与路由权限检查 (RouteChecks)
  |
  v
4. 模型访问权限检查 (can_key_call_model)
  |
  v
5. 预算检查 (max_budget_check, soft_budget_check)
  |
  v
6. 速率限制检查 (parallel_request_limiter, rpm/tpm)
  |
  v
7. Pre-call Guardrails (prompt injection, content filter, PII masking)
  |
  v
8. 调用 LLM Provider
  |
  v
9. Post-call Guardrails (output content check)
  |
  v
10. 费用记录 + 审计日志
  |
  v
返回响应
```
