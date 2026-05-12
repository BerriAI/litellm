## 系统设置与监控

### 健康检查

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/test` | 测试端点 |
| `GET` | `/health/services` | 服务健康检查 |
| `GET` | `/health` | 健康检查 |
| `GET` | `/health/history` | 健康检查历史 |
| `GET` | `/health/latest` | 最新健康检查 |
| `GET` | `/health/shared-status` | 跨 Pod 共享健康状态 |
| `GET` | `/health/license` | License 健康信息 |
| `GET` | `/active/callbacks` | 活跃回调列表 |
| `GET` | `/settings` | 活跃回调列表 |
| `GET` | `/health/readiness` | 健康就绪检查 |
| `OPTIONS` | `/health/readiness` | 健康就绪检查（OPTIONS） |
| `GET` | `/health/backlog` | 积压请求数 |
| `GET` | `/health/liveness` | 健康存活检查 |
| `OPTIONS` | `/health/liveness` | 健康存活检查（OPTIONS） |
| `GET` | `/health/liveliness` | 健康存活检查 |
| `OPTIONS` | `/health/liveliness` | 健康存活检查（OPTIONS） |
| `POST` | `/health/test_connection` | 测试模型连通性 |

#### `GET` /test

**测试端点**

[已废弃] 请改用 `/health/liveliness`。

一个用于 ping proxy 服务、检查是否健康的测试端点。

参数：
    request (Request)：进入的请求对象。

返回：
    dict：包含请求 URL 路由信息的字典。

#### `GET` /health/services

**服务健康检查**

仅限管理员使用，用于检查某个底层服务是否健康。

示例：
```
curl -L -X GET 'http://0.0.0.0:4000/health/services?service=datadog'     -H 'Authorization: Bearer sk-1234'
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `service` | 查询参数 | ✅ | 指定要检查的服务名 |

#### `GET` /health

**健康检查**

🚨 建议使用 `/health/liveliness` 来做 proxy 的健康检查 🚨

更多说明 👉 https://docs.litellm.ai/docs/proxy/health

检查 config.yaml 中所有端点的健康状态。

如需让健康检查在后台运行，在 config.yaml 中加入：
```
general_settings:
    # ... 其他配置
    background_health_checks: True
```
否则健康检查只会在调用 `/health` 时触发。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `model` | 查询参数 | ❌ | 指定模型名（可选） |
| `model_id` | 查询参数 | ❌ | 指定模型 ID（可选） |

#### `GET` /health/history

**健康检查历史**

获取模型的健康检查历史记录。

返回历史健康检查数据，支持可选过滤。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `model` | 查询参数 | ❌ | 按指定模型名过滤 |
| `status_filter` | 查询参数 | ❌ | 按状态过滤（healthy/unhealthy） |
| `limit` | 查询参数 | ❌ | 返回记录条数 |
| `offset` | 查询参数 | ❌ | 跳过的记录条数 |

#### `GET` /health/latest

**最新健康检查**

获取所有模型的最新健康检查状态。

为每个模型返回最近一次的健康检查结果。

#### `GET` /health/shared-status

**跨 Pod 共享健康状态**

获取多 Pod 之间协调共享健康检查的状态。

返回 Redis 连通性、分布式锁状态、以及缓存状态等信息。

#### `GET` /health/license

**License 健康信息**

返回当前配置的 LiteLLM License 元数据，不会暴露 License Key 本身。

#### `GET` /active/callbacks

**活跃回调列表**

返回 LiteLLM 级别的各项设置列表。

用于调试、以及确认 proxy 是否配置正确。

返回结构示例：
```
{
    "alerting": _alerting,
    "litellm.callbacks": litellm_callbacks,
    "litellm.input_callback": litellm_input_callbacks,
    "litellm.failure_callback": litellm_failure_callbacks,
    "litellm.success_callback": litellm_success_callbacks,
    "litellm._async_success_callback": litellm_async_success_callbacks,
    "litellm._async_failure_callba...
```

#### `GET` /settings

**活跃回调列表**

返回 LiteLLM 级别的各项设置列表。

用于调试、以及确认 proxy 是否配置正确。

返回结构示例：
```
{
    "alerting": _alerting,
    "litellm.callbacks": litellm_callbacks,
    "litellm.input_callback": litellm_input_callbacks,
    "litellm.failure_callback": litellm_failure_callbacks,
    "litellm.success_callback": litellm_success_callbacks,
    "litellm._async_success_callback": litellm_async_success_callbacks,
    "litellm._async_failure_callba...
```

#### `GET` /health/readiness

**健康就绪检查**

无需鉴权的端点，用于检查 worker 是否已经可以接收请求。

#### `OPTIONS` /health/readiness

**健康就绪检查（OPTIONS）**

`/health/readiness` 对应的 OPTIONS 端点。

#### `GET` /health/backlog

**积压请求数**

返回当前 uvicorn worker 上正在处理中（in-flight）的 HTTP 请求数量。

可用于衡量单个 Pod 的队列深度。数值越高说明 worker 正在并发处理大量请求——新到达的请求需要排队等待事件循环处理它们，会在 LiteLLM 开始计时之前就引入额外延迟。

#### `GET` /health/liveness

**健康存活检查**

无需鉴权的端点，用于检查 worker 是否存活。

#### `OPTIONS` /health/liveness

**健康存活检查（OPTIONS）**

`/health/liveness` 对应的 OPTIONS 端点。

#### `GET` /health/liveliness

**健康存活检查**

无需鉴权的端点，用于检查 worker 是否存活。

#### `OPTIONS` /health/liveliness

**健康存活检查（OPTIONS）**

`/health/liveliness` 对应的 OPTIONS 端点。

#### `POST` /health/test_connection

**测试模型连通性**

直接测试对指定模型的连通性。

用于验证 proxy 是否能够成功连上某个具体模型，在排查模型连通性问题时无需走完整的 proxy 路由。

示例：
```bash
# 如果模型已经在 proxy_config.yaml 中配置好，只需指定模型名即可：
curl -X POST 'http://localhost:4000/health/test_connection' \
  -H 'Authorization: Bearer sk-1234' \
  -H 'Content-Type: application/json' \
  -d '{...
```

### 系统设置

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/get/mcp_semantic_filter_settings` | 获取 MCP 语义过滤配置 |
| `PATCH` | `/update/mcp_semantic_filter_settings` | 更新 MCP 语义过滤配置 |

#### `GET` /get/mcp_semantic_filter_settings

**获取 MCP 语义过滤配置**

获取 MCP 语义过滤的当前配置，返回语义化工具过滤的相关设置。

#### `PATCH` /update/mcp_semantic_filter_settings

**更新 MCP 语义过滤配置**

将 MCP 语义过滤的新配置写入数据库。所有 Pod 会通过后台轮询在大约 10 秒内拉取到最新配置。

### 配置覆盖

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/config_overrides/hashicorp_vault` | 获取 Hashicorp Vault 配置 |
| `POST` | `/config_overrides/hashicorp_vault` | 更新 Hashicorp Vault 配置 |
| `DELETE` | `/config_overrides/hashicorp_vault` | 删除 Hashicorp Vault 配置 |
| `POST` | `/config_overrides/hashicorp_vault/test_connection` | 测试 Hashicorp Vault 连接 |

#### `GET` /config_overrides/hashicorp_vault

**获取 Hashicorp Vault 配置**

获取当前的 Hashicorp Vault 配置；返回数据库中存储的解密后值，若数据库无配置则回退到当前环境变量。

#### `POST` /config_overrides/hashicorp_vault

**更新 Hashicorp Vault 配置**

更新 Hashicorp Vault 密钥管理器的配置：设置环境变量、加密敏感字段并写入数据库，同时在当前 Pod 上重新初始化 secret manager。

#### `DELETE` /config_overrides/hashicorp_vault

**删除 Hashicorp Vault 配置**

删除 Hashicorp Vault 配置，幂等操作。

#### `POST` /config_overrides/hashicorp_vault/test_connection

**测试 Hashicorp Vault 连接**

测试当前已配置的 Hashicorp Vault 是否可连通。使用已经初始化好的 secret manager 客户端，不会修改任何状态。

### UI 设置

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/in_product_nudges` | 获取产品内提示（Nudges） |
| `GET` | `/get/ui_settings` | 获取 UI 配置 |
| `PATCH` | `/update/ui_settings` | 更新 UI 配置 |

#### `GET` /in_product_nudges

**获取产品内提示（Nudges）**

获取产品内（in-product）Nudges 提示配置。

#### `GET` /get/ui_settings

**获取 UI 配置**

获取 UI 侧的配置开关。所有已认证用户都可以获取这些配置，用于驱动客户端行为。

#### `PATCH` /update/ui_settings

**更新 UI 配置**

更新 UI 侧的配置开关，仅 proxy 管理员允许修改。

### UI 主题设置

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/get/ui_theme_settings` | 获取 UI 主题配置 |
| `PATCH` | `/update/ui_theme_settings` | 更新 UI 主题配置 |
| `POST` | `/upload/logo` | 上传 Logo |

#### `GET` /get/ui_theme_settings

**获取 UI 主题配置**

从 `litellm_settings` 中读取 UI 主题配置，返回当前用于 UI 个性化的 Logo 设置。

注意：该端点是公开的（无需鉴权），以便所有用户都能看到自定义品牌样式。只有 `/update/ui_theme_settings` 需要鉴权并仅允许管理员修改。

#### `PATCH` /update/ui_theme_settings

**更新 UI 主题配置**

更新 UI 主题配置，修改管理后台 UI 的 Logo 设置。

#### `POST` /upload/logo

**上传 Logo**

为管理后台 UI 上传自定义 Logo，接受 PNG / JPG / JPEG / SVG 图片文件并存储用于 UI 展示。

### 日志回调

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/callbacks/list` | 列出回调 |
| `GET` | `/callbacks/configs` | 获取回调配置 |

#### `GET` /callbacks/list

**列出回调**

查看当前已激活的日志回调列表。

#### `GET` /callbacks/configs

**获取回调配置**

获取所有可用回调的配置详情，包括各回调支持的参数、字段类型以及说明文案。

### 邮件管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/email/event_settings` | 获取邮件事件配置 |
| `PATCH` | `/email/event_settings` | 更新邮件事件配置 |
| `POST` | `/email/event_settings/reset` | 重置邮件事件配置 |

#### `GET` /email/event_settings

**获取邮件事件配置**

获取全部邮件事件相关配置。

#### `PATCH` /email/event_settings

**更新邮件事件配置**

更新邮件事件相关配置。

#### `POST` /email/event_settings/reset

**重置邮件事件配置**

将邮件事件配置全部重置为默认值（开启新用户邀请邮件，关闭虚拟 Key 创建通知）。

