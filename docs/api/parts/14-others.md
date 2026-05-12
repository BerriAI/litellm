## 其他

### WebSocket

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/responses` | WebSocket：Responses 连接 |
| `GET` | `/v1/responses` | WebSocket：Responses 连接 |
| `GET` | `/vertex_ai/live` | WebSocket：Vertex AI Live 直通 |
| `GET` | `/realtime` | WebSocket：Realtime 连接 |
| `GET` | `/v1/realtime` | WebSocket：Realtime 连接 |

#### `GET` /responses

**WebSocket：Responses 连接**

WebSocket 连接端点。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `model` | 查询参数 | ✅ |  |

#### `GET` /v1/responses

**WebSocket：Responses 连接**

WebSocket 连接端点。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `model` | 查询参数 | ✅ |  |

#### `GET` /vertex_ai/live

**WebSocket：Vertex AI Live 直通**

WebSocket 连接端点，用于将请求通过 Vertex AI Live API 直通到上游。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `model` | 查询参数 | ❌ |  |
| `vertex_project` | 查询参数 | ❌ |  |
| `vertex_location` | 查询参数 | ❌ |  |

#### `GET` /realtime

**WebSocket：Realtime 连接**

WebSocket 连接端点，用于 OpenAI Realtime API 。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `model` | 查询参数 | ✅ |  |
| `intent` | 查询参数 | ❌ |  |
| `guardrails` | 查询参数 | ❌ |  |

#### `GET` /v1/realtime

**WebSocket：Realtime 连接**

WebSocket 连接端点，用于 OpenAI Realtime API 。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `model` | 查询参数 | ✅ |  |
| `intent` | 查询参数 | ❌ |  |
| `guardrails` | 查询参数 | ❌ |  |

### 交互

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/interactions` | 创建 Interaction |
| `POST` | `/v1beta/interactions` | 创建 Interaction |
| `GET` | `/interactions/{interaction_id}` | 获取 Interaction |
| `DELETE` | `/interactions/{interaction_id}` | 删除 Interaction |
| `GET` | `/v1beta/interactions/{interaction_id}` | 获取 Interaction |
| `DELETE` | `/v1beta/interactions/{interaction_id}` | 删除 Interaction |
| `POST` | `/interactions/{interaction_id}/cancel` | 取消 Interaction |
| `POST` | `/v1beta/interactions/{interaction_id}/cancel` | 取消 Interaction |

#### `POST` /interactions

**创建 Interaction**

使用 Google Interactions API 创建一个新的 interaction。

按 OpenAPI 规范：`POST /{api_version}/interactions`

同时支持模型类型和 agent 类型的 interaction：
- Model：传入 `model` 参数（例如 `gemini-2.5-flash`）
- Agent：传入 `agent` 参数（例如 `deep-research-pro-preview-12-2025`）

请求示例：
```bash
curl -X POST "http://localhost:4000/v1beta/interactions"         -H "Authorization: Bearer sk-1234"         -H "Content-Type: application/json"         -d '{
        "model"...
```

#### `POST` /v1beta/interactions

**创建 Interaction**

使用 Google Interactions API 创建一个新的 interaction。

按 OpenAPI 规范：`POST /{api_version}/interactions`

同时支持模型类型和 agent 类型的 interaction：
- Model：传入 `model` 参数（例如 `gemini-2.5-flash`）
- Agent：传入 `agent` 参数（例如 `deep-research-pro-preview-12-2025`）

请求示例：
```bash
curl -X POST "http://localhost:4000/v1beta/interactions"         -H "Authorization: Bearer sk-1234"         -H "Content-Type: application/json"         -d '{
        "model"...
```

#### `GET` /interactions/{interaction_id}

**获取 Interaction**

根据 ID 获取一个 interaction。

按 OpenAPI 规范：`GET /{api_version}/interactions/{interaction_id}`

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `interaction_id` | 路径参数 | ✅ |  |

#### `DELETE` /interactions/{interaction_id}

**删除 Interaction**

根据 ID 删除一个 interaction。

按 OpenAPI 规范：`DELETE /{api_version}/interactions/{interaction_id}`

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `interaction_id` | 路径参数 | ✅ |  |

#### `GET` /v1beta/interactions/{interaction_id}

**获取 Interaction**

根据 ID 获取一个 interaction。

按 OpenAPI 规范：`GET /{api_version}/interactions/{interaction_id}`

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `interaction_id` | 路径参数 | ✅ |  |

#### `DELETE` /v1beta/interactions/{interaction_id}

**删除 Interaction**

根据 ID 删除一个 interaction。

按 OpenAPI 规范：`DELETE /{api_version}/interactions/{interaction_id}`

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `interaction_id` | 路径参数 | ✅ |  |

#### `POST` /interactions/{interaction_id}/cancel

**取消 Interaction**

根据 ID 取消一个 interaction。

按 OpenAPI 规范：`POST /{api_version}/interactions/{interaction_id}:cancel`

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `interaction_id` | 路径参数 | ✅ |  |

#### `POST` /v1beta/interactions/{interaction_id}/cancel

**取消 Interaction**

根据 ID 取消一个 interaction。

按 OpenAPI 规范：`POST /{api_version}/interactions/{interaction_id}:cancel`

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `interaction_id` | 路径参数 | ✅ |  |

### 实验性功能

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/sso/readiness` | SSO 就绪检查 |

#### `GET` /sso/readiness

**SSO 就绪检查**

用于检查 SSO 是否就绪的健康探测端点。

会检查已配置的 SSO provider 所需的环境变量是否已在内存中完整设置。

### 公共接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/public/model_hub` | 公开 Model Hub |
| `GET` | `/public/agent_hub` | 获取 Agents |
| `GET` | `/public/mcp_hub` | 获取 MCP 服务器 |
| `GET` | `/public/model_hub/info` | 公开 Model Hub 信息 |
| `GET` | `/public/providers` | 获取支持的 Provider 列表 |
| `GET` | `/public/providers/fields` | 获取 Provider 字段元信息 |
| `GET` | `/public/litellm_model_cost_map` | 获取 LiteLLM 模型价格表 |
| `GET` | `/public/litellm_blog_posts` | 获取 LiteLLM 博客文章 |
| `GET` | `/public/endpoints` | 获取支持的端点列表 |
| `GET` | `/public/agents/fields` | 获取 Agent 字段元信息 |

#### `GET` /public/providers

**获取支持的 Provider 列表**

返回 LiteLLM 支持的所有 provider 的有序列表。

#### `GET` /public/providers/fields

**获取 Provider 字段元信息**

返因 dashboard 添加模型流程所需的 provider 元信息。

#### `GET` /public/litellm_model_cost_map

**获取 LiteLLM 模型价格表**

公开端点，用于获取 LiteLLM 的模型价格映射表。返回所有支持模型的计费信息。

#### `GET` /public/litellm_blog_posts

**获取 LiteLLM 博客文章**

公开端点，用于获取最新的 LiteLLM 博客文章。

默认从 GitHub 拉取，并在1 小时进程内缓存内缓。如果失败，会回落到本地内置的备份文件。

#### `GET` /public/endpoints

**获取支持的端点列表**

返回 LiteLLM proxy 的端点列表，以及每个端点被哪些 provider 支持。

从本地内置的备份文件读取，结果在服务进程的生命周期内内缓存。

#### `GET` /public/agents/fields

**获取 Agent 字段元信息**

返回 dashboard 添加 agent 流程所需的 agent 类型元信息。

如果某个 agent 设置了 `inherit_credentials_from_provider`，则所属 provider 的凭证字段会被自动追加到该 agent 的 `credential_fields` 中。

### LLM 工具

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/utils/token_counter` | 计算 token 数 |
| `GET` | `/utils/supported_openai_params` | 查询支持的 OpenAI 参数 |
| `POST` | `/utils/transform_request` | 查看请求 transform 结果 |

#### `POST` /utils/token_counter

**计算 token 数**

参数：
- `request`：`TokenCountRequest`，请求体
- `call_endpoint`：`bool`，置为 `True` 时会真正调用 provider 的 token 计数端点（例如 Anthropic、Google AI Studio 的 token counting API）。

返回 `TokenCountResponse`。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `call_endpoint` | 查询参数 | ❌ |  |

#### `GET` /utils/supported_openai_params

**查询支持的 OpenAI 参数**

返回指定 LiteLLM 模型名支持的 OpenAI 参数列表。

例如 `gpt-4` 与 `gpt-3.5-turbo` 支持的参数集合可能不同。

请求示例：
```
curl -X GET --location 'http://localhost:4000/utils/supported_openai_params?model=gpt-3.5-turbo-16k'         --header 'Authorization: Bearer sk-1234'
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `model` | 查询参数 | ✅ |  |

### 工具

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/utils/test_policies_and_guardrails` | 测试 Policies 与 Guardrails |
| `POST` | `/utils/dotprompt_json_converter` | 将 .prompt 文件转为 JSON |

#### `POST` /utils/test_policies_and_guardrails

**测试 Policies 与 Guardrails**

对输入应用 policies 和/或 guardrails（用于合规 UI 中的测试）。

- 使用 `inputs_list` 进行批量测试：每个 input 会被单独处理，以便逐个返回放行/拦截结果与错误。
- 使用 `inputs` 进行单次调用（旧模式）。

#### `POST` /utils/dotprompt_json_converter

**将 .prompt 文件转为 JSON**

将一个 `.prompt` 文件转换为 JSON 格式。

该端点接收上传的 `.prompt` 文件，并返回与之等价的 JSON 结构，便于写入数据库或程序化使用。

返回的 JSON 包含 `content` 和 `metadata` 两个字段。

### Google GenAI 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/models/{model_name}:generateContent` | Google Generate Content（内容生成） |
| `POST` | `/v1beta/models/{model_name}:generateContent` | Google Generate Content（内容生成） |
| `POST` | `/models/{model_name}:streamGenerateContent` | Google 流式内容生成 |
| `POST` | `/v1beta/models/{model_name}:streamGenerateContent` | Google 流式内容生成 |
| `POST` | `/models/{model_name}:countTokens` | Google 计算 token 数 |
| `POST` | `/v1beta/models/{model_name}:countTokens` | Google 计算 token 数 |
| `POST` | `/interactions` | 创建 Interaction |
| `POST` | `/v1beta/interactions` | 创建 Interaction |
| `GET` | `/interactions/{interaction_id}` | 获取 Interaction |
| `DELETE` | `/interactions/{interaction_id}` | 删除 Interaction |
| `GET` | `/v1beta/interactions/{interaction_id}` | 获取 Interaction |
| `DELETE` | `/v1beta/interactions/{interaction_id}` | 删除 Interaction |
| `POST` | `/interactions/{interaction_id}/cancel` | 取消 Interaction |
| `POST` | `/v1beta/interactions/{interaction_id}/cancel` | 取消 Interaction |

#### `POST` /models/{model_name}:generateContent

**Google Generate Content（内容生成）**

兼容 Google Gemini `generateContent` 的同步端点。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `model_name` | 路径参数 | ✅ |  |

#### `POST` /v1beta/models/{model_name}:generateContent

**Google Generate Content（内容生成）**

兼容 Google Gemini `generateContent` 的同步端点。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `model_name` | 路径参数 | ✅ |  |

#### `POST` /models/{model_name}:streamGenerateContent

**Google 流式内容生成**

兼容 Google Gemini `streamGenerateContent` 的流式端点。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `model_name` | 路径参数 | ✅ |  |

#### `POST` /v1beta/models/{model_name}:streamGenerateContent

**Google 流式内容生成**

兼容 Google Gemini `streamGenerateContent` 的流式端点。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `model_name` | 路径参数 | ✅ |  |

#### `POST` /models/{model_name}:countTokens

**Google 计算 token 数**

兼容 Google Gemini `countTokens` 的端点。响应示例：

```
return {
    "totalTokens": 31,
    "totalBillableCharacters": 96,
    "promptTokensDetails": [
        {
        "modality": "TEXT",
        "tokenCount": 31
        }
    ]
}
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `model_name` | 路径参数 | ✅ |  |

#### `POST` /v1beta/models/{model_name}:countTokens

**Google 计算 token 数**

兼容 Google Gemini `countTokens` 的端点。响应示例：

```
return {
    "totalTokens": 31,
    "totalBillableCharacters": 96,
    "promptTokensDetails": [
        {
        "modality": "TEXT",
        "tokenCount": 31
        }
    ]
}
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `model_name` | 路径参数 | ✅ |  |

#### `POST` /interactions

**创建 Interaction**

使用 Google Interactions API 创建一个新的 interaction。

按 OpenAPI 规范：`POST /{api_version}/interactions`

同时支持模型类型和 agent 类型的 interaction：
- Model：传入 `model` 参数（例如 `gemini-2.5-flash`）
- Agent：传入 `agent` 参数（例如 `deep-research-pro-preview-12-2025`）

请求示例：
```bash
curl -X POST "http://localhost:4000/v1beta/interactions"         -H "Authorization: Bearer sk-1234"         -H "Content-Type: application/json"         -d '{
        "model"...
```

#### `POST` /v1beta/interactions

**创建 Interaction**

使用 Google Interactions API 创建一个新的 interaction。

按 OpenAPI 规范：`POST /{api_version}/interactions`

同时支持模型类型和 agent 类型的 interaction：
- Model：传入 `model` 参数（例如 `gemini-2.5-flash`）
- Agent：传入 `agent` 参数（例如 `deep-research-pro-preview-12-2025`）

请求示例：
```bash
curl -X POST "http://localhost:4000/v1beta/interactions"         -H "Authorization: Bearer sk-1234"         -H "Content-Type: application/json"         -d '{
        "model"...
```

#### `GET` /interactions/{interaction_id}

**获取 Interaction**

根据 ID 获取一个 interaction。

按 OpenAPI 规范：`GET /{api_version}/interactions/{interaction_id}`

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `interaction_id` | 路径参数 | ✅ |  |

#### `DELETE` /interactions/{interaction_id}

**删除 Interaction**

根据 ID 删除一个 interaction。

按 OpenAPI 规范：`DELETE /{api_version}/interactions/{interaction_id}`

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `interaction_id` | 路径参数 | ✅ |  |

#### `GET` /v1beta/interactions/{interaction_id}

**获取 Interaction**

根据 ID 获取一个 interaction。

按 OpenAPI 规范：`GET /{api_version}/interactions/{interaction_id}`

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `interaction_id` | 路径参数 | ✅ |  |

#### `DELETE` /v1beta/interactions/{interaction_id}

**删除 Interaction**

根据 ID 删除一个 interaction。

按 OpenAPI 规范：`DELETE /{api_version}/interactions/{interaction_id}`

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `interaction_id` | 路径参数 | ✅ |  |

#### `POST` /interactions/{interaction_id}/cancel

**取消 Interaction**

根据 ID 取消一个 interaction。

按 OpenAPI 规范：`POST /{api_version}/interactions/{interaction_id}:cancel`

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `interaction_id` | 路径参数 | ✅ |  |

#### `POST` /v1beta/interactions/{interaction_id}/cancel

**取消 Interaction**

根据 ID 取消一个 interaction。

按 OpenAPI 规范：`POST /{api_version}/interactions/{interaction_id}:cancel`

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `interaction_id` | 路径参数 | ✅ |  |

