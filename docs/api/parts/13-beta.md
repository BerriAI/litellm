## Beta 功能

### [beta] Anthropic 事件日志

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/event_logging/batch` | 事件日志批量接收 |

#### `POST` /api/event_logging/batch

**事件日志批量接收**

为 Anthropic 事件日志批量请求提供的占位（Stub）端点。

该端点会接收事件日志请求，但不做任何处理。存在的目的是为了避免 Claude Code 客户端在上报 telemetry 时返回 404 错误。

### [beta] Anthropic 消息 Token 计数

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/messages/count_tokens` | Count Tokens |

#### `POST` /v1/messages/count_tokens

**统计 Token 数**

按 Anthropic Messages API 格式统计 token 数。

该端点遵循 Anthropic Messages API 的 token 计数规范；接受与 `/v1/messages` 端点相同的入参，但只返回 token 数量，不生成实际回复。

示例：
```
curl -X POST "http://localhost:4000/v1/messages/count_tokens?beta=true"       -H "Content-Type: application/json"       -H "Authorization: Bearer your-key"       -d '{
    "model": "claude-3-sonnet-20240229",
    "messages": [{"role"...
```

### [beta] Anthropic 技能 API

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/skills` | 创建 Skill |
| `GET` | `/v1/skills` | 列表 Skills |
| `GET` | `/v1/skills/{skill_id}` | 获取 Skill |
| `DELETE` | `/v1/skills/{skill_id}` | 删除 Skill |

#### `POST` /v1/skills

**创建 Skill**

在 Anthropic 上创建一个新的 skill。

必须带上 `?beta=true` 查询参数。

基于模型的路由（用于多账号场景）：
- 通过请求头：`x-litellm-model: claude-account-1`
- 通过查询参数：`?model=claude-account-1`
- 通过表单字段：`model=claude-account-1`

示例：
```bash
# 基本用法
curl -X POST "http://localhost:4000/v1/skills?beta=true"       -H "Content-Type: multipart/form-data"       -H "Authorization: Bearer your-key"       -F "display_title=My Ski...
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `custom_llm_provider` | 查询参数 | ❌ |  |

#### `GET` /v1/skills

**列表 Skills**

列出 Anthropic 账号下的所有 skill。

必须带上 `?beta=true` 查询参数。

基于模型的路由（用于多账号场景）：
- 通过请求头：`x-litellm-model: claude-account-1`
- 通过查询参数：`?model=claude-account-1`
- 通过请求体：`{"model": "claude-account-1"}`

示例：
```bash
# 基本用法
curl "http://localhost:4000/v1/skills?beta=true&limit=10"       -H "Authorization: Bearer your-key"

# 使用基于模型的路由
curl "http://localhost:4000/v1/skills?beta=true&limi...
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `limit` | 查询参数 | ❌ |  |
| `after_id` | 查询参数 | ❌ |  |
| `before_id` | 查询参数 | ❌ |  |
| `custom_llm_provider` | 查询参数 | ❌ |  |

#### `GET` /v1/skills/{skill_id}

**获取 Skill**

通过 ID 从 Anthropic 获取指定 skill。

必须带上 `?beta=true` 查询参数。

基于模型的路由（用于多账号场景）：
- 通过请求头：`x-litellm-model: claude-account-1`
- 通过查询参数：`?model=claude-account-1`
- 通过请求体：`{"model": "claude-account-1"}`

示例：
```bash
# 基本用法
curl "http://localhost:4000/v1/skills/skill_123?beta=true"       -H "Authorization: Bearer your-key"

# 使用基于模型的路由
curl "http://localhost:4000/v1/ski...
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `skill_id` | 路径参数 | ✅ |  |
| `custom_llm_provider` | 查询参数 | ❌ |  |

#### `DELETE` /v1/skills/{skill_id}

**删除 Skill**

通过 ID 在 Anthropic 上删除指定 skill。

必须带上 `?beta=true` 查询参数。

注意：Anthropic 不允许删除已经存在版本（version）的 skill。

基于模型的路由（用于多账号场景）：
- 通过请求头：`x-litellm-model: claude-account-1`
- 通过查询参数：`?model=claude-account-1`
- 通过请求体：`{"model": "claude-account-1"}`

示例：
```bash
# 基本用法
curl -X DELETE "http://localhost:4000/v1/skills/skill_123?beta=true"       -H "Authorization: Bea...
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `skill_id` | 路径参数 | ✅ |  |
| `custom_llm_provider` | 查询参数 | ❌ |  |

### [beta] Anthropic `/v1/messages`

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/messages` | Anthropic Response |

#### `POST` /v1/messages

**Anthropic Response**

建议改用 `{PROXY_BASE_URL}/anthropic/v1/messages` —— [官方文档](https://docs.litellm.ai/docs/pass_through/anthropic_completion)。

这是一个 BETA 端点，可以用 Anthropic 格式调用 100+ 个 LLM。

