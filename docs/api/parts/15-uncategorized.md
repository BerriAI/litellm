## 其他接口

### 未分类

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 首页 |
| `GET` | `/routes` | 获取路由列表 |
| `POST` | `/vector_stores/{vector_store_id}/search` | 搜索向量存储 |
| `POST` | `/v1/vector_stores/{vector_store_id}/search` | 搜索向量存储 |
| `POST` | `/vector_stores` | 创建向量存储 |
| `GET` | `/vector_stores` | 列出向量存储 |
| `POST` | `/v1/vector_stores` | 创建向量存储 |
| `GET` | `/v1/vector_stores` | 列出向量存储 |
| `GET` | `/vector_stores/{vector_store_id}` | 获取向量存储 |
| `POST` | `/vector_stores/{vector_store_id}` | 更新向量存储 |
| `DELETE` | `/vector_stores/{vector_store_id}` | 删除向量存储 |
| `GET` | `/v1/vector_stores/{vector_store_id}` | 获取向量存储 |
| `POST` | `/v1/vector_stores/{vector_store_id}` | 更新向量存储 |
| `DELETE` | `/v1/vector_stores/{vector_store_id}` | 删除向量存储 |
| `POST` | `/v1/indexes` | 创建索引 |
| `GET` | `/config/pass_through_endpoint/team/{team_id}` | 获取直通端点 |
| `GET` | `/config/pass_through_endpoint` | 获取直通端点 |
| `POST` | `/config/pass_through_endpoint` | 创建直通端点 |
| `DELETE` | `/config/pass_through_endpoint` | 删除直通端点 |
| `POST` | `/config/pass_through_endpoint/{endpoint_id}` | 更新直通端点 |
| `GET` | `/team/available` | 列出可加入的 Team |
| `GET` | `/provider/budgets` | Provider 预算 |
| `POST` | `/apply_guardrail` | 应用 Guardrail |
| `POST` | `/guardrails/apply_guardrail` | 应用 Guardrail |
| `GET` | `/debug/asyncio-tasks` | 获取当前 asyncio 活跃任务统计 |
| `GET` | `/robots.txt` | 获取 robots.txt |
| `GET` | `/litellm/.well-known/litellm-ui-config` | 获取 UI 配置 |
| `GET` | `/.well-known/litellm-ui-config` | 获取 UI 配置 |
| `PATCH` | `/{mcp_server_name}/mcp` | 动态 MCP 路由 |
| `HEAD` | `/{mcp_server_name}/mcp` | 动态 MCP 路由 |
| `POST` | `/{mcp_server_name}/mcp` | 动态 MCP 路由 |
| `GET` | `/{mcp_server_name}/mcp` | 动态 MCP 路由 |
| `OPTIONS` | `/{mcp_server_name}/mcp` | 动态 MCP 路由 |
| `DELETE` | `/{mcp_server_name}/mcp` | 动态 MCP 路由 |
| `PUT` | `/{mcp_server_name}/mcp` | 动态 MCP 路由 |

#### `GET` /routes

**获取路由列表**

获取当前 FastAPI 应用中所有可用路由的列表。

#### `POST` /vector_stores/{vector_store_id}/search

**搜索向量存储**

在向量存储（vector store）中执行搜索。

OpenAI 官方规范：https://platform.openai.com/docs/api-reference/vector-stores/search

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `vector_store_id` | 路径参数 | ✅ |  |

#### `POST` /v1/vector_stores/{vector_store_id}/search

**搜索向量存储**

在向量存储（vector store）中执行搜索。

OpenAI 官方规范：https://platform.openai.com/docs/api-reference/vector-stores/search

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `vector_store_id` | 路径参数 | ✅ |  |

#### `POST` /vector_stores

**创建向量存储**

创建一个向量存储（vector store）。

OpenAI 官方规范：https://platform.openai.com/docs/api-reference/vector-stores/create

支持通过 `target_model_names` 参数在多个模型下创建向量存储：
```json
{
    "name": "my-vector-store",
    "target_model_names": "gpt-4,gemini-2.0"
}
```

#### `GET` /vector_stores

**列出向量存储**

列出向量存储。

OpenAI 官方规范：https://platform.openai.com/docs/api-reference/vector-stores/list

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `after` | 查询参数 | ❌ |  |
| `before` | 查询参数 | ❌ |  |
| `limit` | 查询参数 | ❌ |  |
| `order` | 查询参数 | ❌ |  |

#### `POST` /v1/vector_stores

**创建向量存储**

创建一个向量存储（vector store）。

OpenAI 官方规范：https://platform.openai.com/docs/api-reference/vector-stores/create

支持通过 `target_model_names` 参数在多个模型下创建向量存储：
```json
{
    "name": "my-vector-store",
    "target_model_names": "gpt-4,gemini-2.0"
}
```

#### `GET` /v1/vector_stores

**列出向量存储**

列出向量存储。

OpenAI 官方规范：https://platform.openai.com/docs/api-reference/vector-stores/list

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `after` | 查询参数 | ❌ |  |
| `before` | 查询参数 | ❌ |  |
| `limit` | 查询参数 | ❌ |  |
| `order` | 查询参数 | ❌ |  |

#### `GET` /vector_stores/{vector_store_id}

**获取向量存储**

获取指定向量存储的详情。

OpenAI 官方规范：https://platform.openai.com/docs/api-reference/vector-stores/retrieve

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `vector_store_id` | 路径参数 | ✅ |  |

#### `POST` /vector_stores/{vector_store_id}

**更新向量存储**

更新指定向量存储。

OpenAI 官方规范：https://platform.openai.com/docs/api-reference/vector-stores/modify

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `vector_store_id` | 路径参数 | ✅ |  |

#### `DELETE` /vector_stores/{vector_store_id}

**删除向量存储**

删除指定向量存储。

OpenAI 官方规范：https://platform.openai.com/docs/api-reference/vector-stores/delete

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `vector_store_id` | 路径参数 | ✅ |  |

#### `GET` /v1/vector_stores/{vector_store_id}

**获取向量存储**

获取指定向量存储的详情。

OpenAI 官方规范：https://platform.openai.com/docs/api-reference/vector-stores/retrieve

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `vector_store_id` | 路径参数 | ✅ |  |

#### `POST` /v1/vector_stores/{vector_store_id}

**更新向量存储**

更新指定向量存储。

OpenAI 官方规范：https://platform.openai.com/docs/api-reference/vector-stores/modify

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `vector_store_id` | 路径参数 | ✅ |  |

#### `DELETE` /v1/vector_stores/{vector_store_id}

**删除向量存储**

删除指定向量存储。

OpenAI 官方规范：https://platform.openai.com/docs/api-reference/vector-stores/delete

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `vector_store_id` | 路径参数 | ✅ |  |

#### `POST` /v1/indexes

**创建索引**

创建一个索引，仅将索引写入数据库（不写入底层向量库）。

```bash
curl -L -X POST 'http://0.0.0.0:4000/indexes/create'         -H 'Content-Type: application/json'         -H 'Authorization: Bearer sk-1234'         -H 'LiteLLM-Beta: indexes_beta=v1'         -d '{ 
        "index_name": "dall-e-3",
        "vector_store_index": "real-index-name",
        "vector_store_name": "azure-ai-search"
    }'
```

#### `GET` /config/pass_through_endpoint/team/{team_id}

**获取直通端点**

获取已配置的直通（pass-through）端点。

若未传入 `endpoint_id`，则返回全部已配置的直通端点。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `team_id` | 路径参数 | ✅ |  |
| `endpoint_id` | 查询参数 | ❌ |  |

#### `GET` /config/pass_through_endpoint

**获取直通端点**

获取已配置的直通（pass-through）端点。

若未传入 `endpoint_id`，则返回全部已配置的直通端点。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `endpoint_id` | 查询参数 | ❌ |  |
| `team_id` | 查询参数 | ❌ |  |

#### `POST` /config/pass_through_endpoint

**创建直通端点**

创建一个新的直通（pass-through）端点。

#### `DELETE` /config/pass_through_endpoint

**删除直通端点**

根据 ID 删除一个直通端点。

返回被删除的端点对象。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `endpoint_id` | 查询参数 | ✅ |  |

#### `POST` /config/pass_through_endpoint/{endpoint_id}

**更新直通端点**

根据 ID 更新一个直通端点。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `endpoint_id` | 路径参数 | ✅ |  |

#### `GET` /team/available

**列出可加入的 Team**

列出当前用户可以加入的 Team。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `response_model` | 查询参数 | ❌ |  |

#### `GET` /provider/budgets

**Provider 预算**

Provider 预算路由 —— 查看预算与消费详情：https://docs.litellm.ai/docs/proxy/provider_budget_routing

使用该端点可以查看某个 provider 当前的预算、消费、以及预算重置时间。

请求示例：

```bash
curl -X GET http://localhost:4000/provider/budgets     -H "Content-Type: application/json"     -H "Authorization: Bearer sk-1234"
```

响应示例：

```
{
    "providers": {
        "openai": {
            "budget_limit": 1e-12,
            "time_period": "1d",
      ...
```

#### `POST` /apply_guardrail

**应用 Guardrail**

对文本输入应用 guardrail（安全护栏），并返回处理后的结果。

该端点可以用于对自定义文本应用 guardrail 来测试其效果。

#### `POST` /guardrails/apply_guardrail

**应用 Guardrail**

对文本输入应用 guardrail（安全护栏），并返回处理后的结果。

该端点可以用于对自定义文本应用 guardrail 来测试其效果。

#### `GET` /debug/asyncio-tasks

**获取当前 asyncio 活跃任务统计**

返回当前 worker 上正在运行的 asyncio 任务统计：
- `total_active_tasks`: int，当前活跃任务的总数
- `by_name`: 按协程名统计的占比（形如 `{coroutine_name: count}`）

#### `GET` /robots.txt

**获取 robots.txt**

阻止所有网页爬虫索引 proxy 服务的端点。有助于确保 API 端点不会被搜索引擎收录。

#### `PATCH` /{mcp_server_name}/mcp

**动态 MCP 路由**

处理动态的 MCP 服务器路由，例如 `/github_mcp/mcp`。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `mcp_server_name` | 路径参数 | ✅ |  |

#### `HEAD` /{mcp_server_name}/mcp

**动态 MCP 路由**

处理动态的 MCP 服务器路由，例如 `/github_mcp/mcp`。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `mcp_server_name` | 路径参数 | ✅ |  |

#### `POST` /{mcp_server_name}/mcp

**动态 MCP 路由**

处理动态的 MCP 服务器路由，例如 `/github_mcp/mcp`。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `mcp_server_name` | 路径参数 | ✅ |  |

#### `GET` /{mcp_server_name}/mcp

**动态 MCP 路由**

处理动态的 MCP 服务器路由，例如 `/github_mcp/mcp`。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `mcp_server_name` | 路径参数 | ✅ |  |

#### `OPTIONS` /{mcp_server_name}/mcp

**动态 MCP 路由**

处理动态的 MCP 服务器路由，例如 `/github_mcp/mcp`。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `mcp_server_name` | 路径参数 | ✅ |  |

#### `DELETE` /{mcp_server_name}/mcp

**动态 MCP 路由**

处理动态的 MCP 服务器路由，例如 `/github_mcp/mcp`。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `mcp_server_name` | 路径参数 | ✅ |  |

#### `PUT` /{mcp_server_name}/mcp

**动态 MCP 路由**

处理动态的 MCP 服务器路由，例如 `/github_mcp/mcp`。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `mcp_server_name` | 路径参数 | ✅ |  |
