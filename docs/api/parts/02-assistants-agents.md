## 助手与代理

### 助手

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/assistants` | 获取助手列表 |
| `POST` | `/assistants` | 创建助手 |
| `GET` | `/v1/assistants` | 获取助手列表 |
| `POST` | `/v1/assistants` | 创建助手 |
| `DELETE` | `/assistants/{assistant_id}` | 删除助手 |
| `DELETE` | `/v1/assistants/{assistant_id}` | 删除助手 |
| `POST` | `/threads` | 创建线程 |
| `POST` | `/v1/threads` | 创建线程 |
| `GET` | `/threads/{thread_id}` | 获取线程 |
| `GET` | `/v1/threads/{thread_id}` | 获取线程 |
| `POST` | `/threads/{thread_id}/messages` | 添加消息 |
| `GET` | `/threads/{thread_id}/messages` | 获取消息 |
| `POST` | `/v1/threads/{thread_id}/messages` | 添加消息 |
| `GET` | `/v1/threads/{thread_id}/messages` | 获取消息 |
| `POST` | `/threads/{thread_id}/runs` | 运行线程 |
| `POST` | `/v1/threads/{thread_id}/runs` | 运行线程 |

#### `GET` /assistants

**获取助手列表**

返回 Assistant 列表。

OpenAPI 官方规范 docs - https://platform.openai.com/docs/api-reference/assistants/listAssistants

#### `POST` /assistants

**创建助手**

创建 Assistant。

OpenAPI 官方规范 docs - https://platform.openai.com/docs/api-reference/assistants/createAssistant

#### `GET` /v1/assistants

**获取助手列表**

返回 Assistant 列表。

OpenAPI 官方规范 docs - https://platform.openai.com/docs/api-reference/assistants/listAssistants

#### `POST` /v1/assistants

**创建助手**

创建 Assistant。

OpenAPI 官方规范 docs - https://platform.openai.com/docs/api-reference/assistants/createAssistant

#### `DELETE` /assistants/{assistant_id}

**删除助手**

删除 Assistant。

OpenAPI 官方规范 docs - https://platform.openai.com/docs/api-reference/assistants/createAssistant

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `assistant_id` | 路径参数 | ✅ |  |

#### `DELETE` /v1/assistants/{assistant_id}

**删除助手**

删除 Assistant。

OpenAPI 官方规范 docs - https://platform.openai.com/docs/api-reference/assistants/createAssistant

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `assistant_id` | 路径参数 | ✅ |  |

#### `POST` /threads

**创建线程**

创建 Thread。

OpenAPI 官方规范 - https://platform.openai.com/docs/api-reference/threads/createThread

#### `POST` /v1/threads

**创建线程**

创建 Thread。

OpenAPI 官方规范 - https://platform.openai.com/docs/api-reference/threads/createThread

#### `GET` /threads/{thread_id}

**获取线程**

获取 Thread 详情。

OpenAPI 官方规范 - https://platform.openai.com/docs/api-reference/threads/getThread

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `thread_id` | 路径参数 | ✅ |  |

#### `GET` /v1/threads/{thread_id}

**获取线程**

获取 Thread 详情。

OpenAPI 官方规范 - https://platform.openai.com/docs/api-reference/threads/getThread

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `thread_id` | 路径参数 | ✅ |  |

#### `POST` /threads/{thread_id}/messages

**添加消息**

创建 Message。

OpenAPI 官方规范 - https://platform.openai.com/docs/api-reference/messages/createMessage

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `thread_id` | 路径参数 | ✅ |  |

#### `GET` /threads/{thread_id}/messages

**获取消息**

返回指定 Thread 下的 Message 列表。

OpenAPI 官方规范 - https://platform.openai.com/docs/api-reference/messages/listMessages

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `thread_id` | 路径参数 | ✅ |  |

#### `POST` /v1/threads/{thread_id}/messages

**添加消息**

创建 Message。

OpenAPI 官方规范 - https://platform.openai.com/docs/api-reference/messages/createMessage

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `thread_id` | 路径参数 | ✅ |  |

#### `GET` /v1/threads/{thread_id}/messages

**获取消息**

返回指定 Thread 下的 Message 列表。

OpenAPI 官方规范 - https://platform.openai.com/docs/api-reference/messages/listMessages

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `thread_id` | 路径参数 | ✅ |  |

#### `POST` /threads/{thread_id}/runs

**运行线程**

创建 Run。

OpenAPI 官方规范： https://platform.openai.com/docs/api-reference/runs/createRun

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `thread_id` | 路径参数 | ✅ |  |

#### `POST` /v1/threads/{thread_id}/runs

**运行线程**

创建 Run。

OpenAPI 官方规范： https://platform.openai.com/docs/api-reference/runs/createRun

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `thread_id` | 路径参数 | ✅ |  |

### 代理

| 方法 | 路径 | 说明 |
|------|------|------|
| `DELETE` | `/v1/agents/{agent_id}` | 删除代理 |

#### `DELETE` /v1/agents/{agent_id}

**删除代理**

删除 Agent。

请求示例：
```bash
curl -X DELETE "http://localhost:4000/agents/123e4567-e89b-12d3-a456-426614174000" \
    -H "Authorization: Bearer <your_api_key>"
```

响应示例：
```json
{
    "message": "Agent 123e4567-e89b-12d3-a456-426614174000 deleted successfully"
}
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `agent_id` | 路径参数 | ✅ |  |

### 代理管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/agent/daily/activity` | 获取 Agent Daily Activity |

#### `GET` /agent/daily/activity

**获取 Agent Daily Activity**

按指定 Agent（或当前可访问的全部 Agent）获取每日活动数据。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `agent_ids` | 查询参数 | ❌ |  |
| `start_date` | 查询参数 | ❌ |  |
| `end_date` | 查询参数 | ❌ |  |
| `model` | 查询参数 | ❌ |  |
| `api_key` | 查询参数 | ❌ |  |
| `page` | 查询参数 | ❌ |  |
| `page_size` | 查询参数 | ❌ |  |
| `exclude_agent_ids` | 查询参数 | ❌ |  |

### [beta] 代理

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/public/agent_hub` | 获取 Agents |
| `GET` | `/public/agents/fields` | 获取 Agent Fields |

#### `GET` /public/agents/fields

**获取 Agent Fields**

返回管理后台创建 Agent 流程所需的 Agent 类型元数据。

如果某个 Agent 开启了 `inherit_credentials_from_provider`，则使用对应 Provider 的凭证。
字段会自动追加到该 Agent 的 `credential_fields` 里。

### [beta] A2A 代理

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/v1/agents` | 获取 Agents |
| `POST` | `/v1/agents` | 创建代理 |
| `GET` | `/v1/agents/{agent_id}` | 获取 Agent By Id |
| `PUT` | `/v1/agents/{agent_id}` | 更新代理 |
| `PATCH` | `/v1/agents/{agent_id}` | Patch Agent |
| `POST` | `/v1/agents/{agent_id}/make_public` | Make Agent Public |
| `POST` | `/v1/agents/make_public` | Make Agents Public |
| `GET` | `/a2a/{agent_id}/.well-known/agent.json` | 获取 Agent Card |
| `GET` | `/a2a/{agent_id}/.well-known/agent-card.json` | 获取 Agent Card |
| `POST` | `/v1/a2a/{agent_id}/message/send` | Invoke Agent A2A |
| `POST` | `/a2a/{agent_id}/message/send` | Invoke Agent A2A |
| `POST` | `/a2a/{agent_id}` | Invoke Agent A2A |

#### `GET` /v1/agents

**获取 Agents**

请求示例：
```
curl -X GET "http://localhost:4000/v1/agents"       -H "Content-Type: application/json"       -H "Authorization: Bearer your-key"     ```

传入 `?health_check=true` 可过滤掉 URL 不可达的 Agent：
```
curl -X GET "http://localhost:4000/v1/agents?health_check=true"       -H "Content-Type: application/json"       -H "Authorization: Bearer your-key"     ```

返回： List[AgentResponse]

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `health_check` | 查询参数 | ❌ | When true, performs a GET request to each agent's URL. Agents with reachable URLs (HTTP status < 500 |

#### `POST` /v1/agents

**创建代理**

创建新的 Agent。

请求示例：
```bash
curl -X POST "http://localhost:4000/agents" \
    -H "Authorization: Bearer <your_api_key>" \
    -H "Content-Type: application/json" \
    -d '{
        "agent": {
            "agent_name": "my-custom-agent",
            "agent_card_params": {
                "protocolVersion": "1.0",
                "name": "Hello World Agent",
                "description": "Just a hello world agent",
                "url": "http://localhost:9999/",
               ...
```

#### `GET` /v1/agents/{agent_id}

**获取 Agent By Id**

根据 ID 获取指定 Agent。

请求示例：
```bash
curl -X GET "http://localhost:4000/agents/123e4567-e89b-12d3-a456-426614174000" \
    -H "Authorization: Bearer <your_api_key>"
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `agent_id` | 路径参数 | ✅ |  |

#### `PUT` /v1/agents/{agent_id}

**更新代理**

更新已有的 Agent。

请求示例：
```bash
curl -X PUT "http://localhost:4000/agents/123e4567-e89b-12d3-a456-426614174000" \
    -H "Authorization: Bearer <your_api_key>" \
    -H "Content-Type: application/json" \
    -d '{
        "agent": {
            "agent_name": "updated-agent",
            "agent_card_params": {
                "protocolVersion": "1.0",
                "name": "Updated Agent",
                "description": "Updated description",
                "url": "http://lo...
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `agent_id` | 路径参数 | ✅ |  |

#### `PATCH` /v1/agents/{agent_id}

**局部更新 Agent**

更新已有的 Agent。

请求示例：
```bash
curl -X PUT "http://localhost:4000/agents/123e4567-e89b-12d3-a456-426614174000" \
    -H "Authorization: Bearer <your_api_key>" \
    -H "Content-Type: application/json" \
    -d '{
        "agent": {
            "agent_name": "updated-agent",
            "agent_card_params": {
                "protocolVersion": "1.0",
                "name": "Updated Agent",
                "description": "Updated description",
                "url": "http://lo...
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `agent_id` | 路径参数 | ✅ |  |

#### `POST` /v1/agents/{agent_id}/make_public

**将 Agent 设为公开**

将 Agent 设为公开可发现。

请求示例：
```bash
curl -X POST "http://localhost:4000/v1/agents/123e4567-e89b-12d3-a456-426614174000/make_public" \
    -H "Authorization: Bearer <your_api_key>" \
    -H "Content-Type: application/json"
```

响应示例：
```
{
    "agent_id": "123e4567-e89b-12d3-a456-426614174000",
    "agent_name": "my-custom-agent",
    "litellm_params": {
        "make_public": true
    },
    "agent_card_params": {...},
    "created_at": "2025-11-15T10:30...
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `agent_id` | 路径参数 | ✅ |  |

#### `POST` /v1/agents/make_public

**批量将 Agent 设为公开**

批量将多个 Agent 设为公开可发现。

请求示例：
```bash
curl -X POST "http://localhost:4000/v1/agents/make_public" \
    -H "Authorization: Bearer <your_api_key>" \
    -H "Content-Type: application/json" \
    -d '{
        "agent_ids": ["123e4567-e89b-12d3-a456-426614174000", "123e4567-e89b-12d3-a456-426614174001"]
    }'
```

响应示例：
```
{
    "agent_id": "123e4567-e89b-12d3-a456-426614174000",
    "agent_name": "my-custom-agent",
    "litellm_params": {
        "ma...
```

#### `GET` /a2a/{agent_id}/.well-known/agent.json

**获取 Agent Card**

获取 Agent 的 Agent Card（A2A 发现端点）。

同时支持以下标准路径：
- /.well-known/agent-card.json
- /.well-known/agent.json

Agent Card 中的 URL 将被改写为指向 LiteLLM 代理，这样后续全部 A2A 调用都会经过 LiteLLM，从而实现统一的日志记录和成本追踪。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `agent_id` | 路径参数 | ✅ |  |

#### `GET` /a2a/{agent_id}/.well-known/agent-card.json

**获取 Agent Card**

获取 Agent 的 Agent Card（A2A 发现端点）。

同时支持以下标准路径：
- /.well-known/agent-card.json
- /.well-known/agent.json

Agent Card 中的 URL 将被改写为指向 LiteLLM 代理，这样后续全部 A2A 调用都会经过 LiteLLM，从而实现统一的日志记录和成本追踪。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `agent_id` | 路径参数 | ✅ |  |

#### `POST` /v1/a2a/{agent_id}/message/send

**通过 A2A 协议调用 Agent**

使用 A2A 协议（JSON-RPC 2.0）调用 Agent。

支持的方法：
- message/send: Send a message and get a response
- message/stream: Send a message and stream the response

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `agent_id` | 路径参数 | ✅ |  |

#### `POST` /a2a/{agent_id}/message/send

**通过 A2A 协议调用 Agent**

使用 A2A 协议（JSON-RPC 2.0）调用 Agent。

支持的方法：
- message/send: Send a message and get a response
- message/stream: Send a message and stream the response

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `agent_id` | 路径参数 | ✅ |  |

#### `POST` /a2a/{agent_id}

**通过 A2A 协议调用 Agent**

使用 A2A 协议（JSON-RPC 2.0）调用 Agent。

支持的方法：
- message/send: Send a message and get a response
- message/stream: Send a message and stream the response

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `agent_id` | 路径参数 | ✅ |  |

### [beta] MCP 模型上下文协议

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/public/mcp_hub` | 获取 Mcp Servers |

### MCP 模型上下文协议

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/v1/mcp/tools` | 获取 Mcp Tools |
| `GET` | `/v1/mcp/access_groups` | 获取 Mcp Access Groups |
| `GET` | `/v1/mcp/network/client-ip` | 获取 Client Ip |
| `GET` | `/v1/mcp/registry.json` | 获取 Mcp Registry |
| `GET` | `/v1/mcp/server` | Fetch All Mcp Servers |
| `POST` | `/v1/mcp/server` | 添加 Mcp Server |
| `PUT` | `/v1/mcp/server` | Edit Mcp Server |
| `GET` | `/v1/mcp/server/health` | Health Check Servers |
| `POST` | `/v1/mcp/server/register` | Register Mcp Server |
| `GET` | `/v1/mcp/server/submissions` | 获取 Mcp Server Submissions |
| `PUT` | `/v1/mcp/server/{server_id}/approve` | Approve Mcp Server Submission |
| `PUT` | `/v1/mcp/server/{server_id}/reject` | Reject Mcp Server Submission |
| `GET` | `/v1/mcp/server/{server_id}` | Fetch Mcp Server |
| `DELETE` | `/v1/mcp/server/{server_id}` | 移除 Mcp Server |
| `POST` | `/v1/mcp/server/oauth/session` | 添加 Session Mcp Server |
| `POST` | `/v1/mcp/server/{server_id}/user-credential` | Store Mcp User Credential |
| `DELETE` | `/v1/mcp/server/{server_id}/user-credential` | 删除 Mcp User Credential |
| `POST` | `/v1/mcp/server/{server_id}/oauth-user-credential` | Store Mcp Oauth User Credential |
| `DELETE` | `/v1/mcp/server/{server_id}/oauth-user-credential` | 删除 Mcp Oauth User Credential |
| `GET` | `/v1/mcp/server/{server_id}/oauth-user-credential/status` | 获取 Mcp Oauth User Credential Status |
| `GET` | `/v1/mcp/user-credentials` | 列表 Mcp User Credentials |
| `POST` | `/v1/mcp/make_public` | Make Mcp Servers Public |
| `GET` | `/v1/mcp/discover` | Discover Mcp Servers |
| `GET` | `/v1/mcp/openapi-registry` | 获取 Openapi Registry |
| `GET` | `/mcp-rest/tools/list` | 列表 Tool Rest Api |
| `POST` | `/mcp-rest/tools/call` | Call Tool Rest Api |
| `POST` | `/mcp-rest/test/connection` | Test Connection |
| `POST` | `/mcp-rest/test/tools/list` | Test Tools List |
| `GET` | `/authorize` | Authorize |
| `GET` | `/{mcp_server_name}/authorize` | Authorize |
| `POST` | `/token` | Token Endpoint |
| `POST` | `/{mcp_server_name}/token` | Token Endpoint |
| `GET` | `/callback` | Callback |
| `GET` | `/.well-known/oauth-protected-resource/mcp/{mcp_server_name}` | Oauth Protected Resource Mcp Standard |
| `GET` | `/.well-known/oauth-protected-resource` | Oauth Protected Resource Mcp |
| `GET` | `/.well-known/oauth-protected-resource/{mcp_server_name}/mcp` | Oauth Protected Resource Mcp |
| `GET` | `/.well-known/oauth-authorization-server/mcp/{mcp_server_name}` | Oauth Authorization Server Mcp Standard |
| `GET` | `/.well-known/oauth-authorization-server` | Oauth Authorization Server Mcp |
| `GET` | `/.well-known/oauth-authorization-server/{mcp_server_name}` | Oauth Authorization Server Mcp |
| `GET` | `/.well-known/openid-configuration` | Openid Configuration |
| `GET` | `/.well-known/jwks.json` | Jwks Json |
| `GET` | `/.well-known/oauth-authorization-server/{mcp_server_name}/mcp` | Oauth Authorization Server Legacy |
| `POST` | `/register` | Register Client |
| `POST` | `/{mcp_server_name}/register` | Register Client |

#### `GET` /v1/mcp/tools

**获取 Mcp Tools**

获取当前 Key 可用的全部 MCP 工具（包含通过访问组可用的部分）。

#### `GET` /v1/mcp/access_groups

**获取 Mcp Access Groups**

同时从数据库与配置中获取所有可用的 MCP 访问组。

#### `GET` /v1/mcp/network/client-ip

**获取 Client Ip**

返回代理所见的调用方 IP 地址。

#### `GET` /v1/mcp/registry.json

**获取 Mcp Registry**

MCP Registry 端点。规范：https://github.com/modelcontextprotocol/registry

#### `GET` /v1/mcp/server

**获取全部 MCP 服务器**

返回 MCP 服务器列表及其关联的团队。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `team_id` | 查询参数 | ❌ | Filter MCP servers by team scope. When provided, returns only servers the team has access to plus gl |

#### `POST` /v1/mcp/server

**添加 Mcp Server**

允许创建 MCP 服务器。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `litellm-changed-by` | 请求头 | ❌ | The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of |

#### `PUT` /v1/mcp/server

**编辑 MCP 服务器**

支持删除数据库中的 MCP 服务器记录。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `litellm-changed-by` | 请求头 | ❌ | The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of |

#### `GET` /v1/mcp/server/health

**服务器健康检查**

MCP 服务器健康检查。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `server_ids` | 查询参数 | ❌ | Server IDs to check. If not provided, checks all accessible servers. |

#### `POST` /v1/mcp/server/register

**注册 MCP 服务器**

提交新的 MCP 服务器待管理员审核（面向非管理员），逻辑与 `POST /guardrails/register` 对应。

#### `GET` /v1/mcp/server/submissions

**获取 Mcp Server Submissions**

返回非管理员用户提交的全部 MCP 服务器（管理员审核队列），逻辑与 `GET /guardrails/submissions` 对应。

#### `PUT` /v1/mcp/server/{server_id}/approve

**审批 MCP 服务器提交**

审批通过待审核的 MCP 服务器（仅管理员），逻辑与 `PUT /guardrails/{id}/approve` 对应。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `server_id` | 路径参数 | ✅ |  |

#### `PUT` /v1/mcp/server/{server_id}/reject

**驳回 MCP 服务器提交**

驳回待审核的 MCP 服务器（仅管理员），逻辑与 `PUT /guardrails/{id}/reject` 对应。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `server_id` | 路径参数 | ✅ |  |

#### `GET` /v1/mcp/server/{server_id}

**获取 MCP 服务器**

返回 MCP 服务器的详细信息。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `server_id` | 路径参数 | ✅ |  |

#### `DELETE` /v1/mcp/server/{server_id}

**移除 Mcp Server**

支持删除数据库中的 MCP 服务器记录。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `server_id` | 路径参数 | ✅ |  |
| `litellm-changed-by` | 请求头 | ❌ | The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of |

#### `POST` /v1/mcp/server/oauth/session

**添加 Session Mcp Server**

将 MCP 服务器临时缓存在内存中，不写入数据库。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `litellm-changed-by` | 请求头 | ❌ | The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of |

#### `POST` /v1/mcp/server/{server_id}/user-credential

**保存 MCP 用户凭证**

为 BYOK MCP 服务器保存或更新调用方用户的 API Key。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `server_id` | 路径参数 | ✅ |  |

#### `DELETE` /v1/mcp/server/{server_id}/user-credential

**删除 Mcp User Credential**

删除调用方用户为 BYOK MCP 服务器保存的 API Key。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `server_id` | 路径参数 | ✅ |  |

#### `POST` /v1/mcp/server/{server_id}/oauth-user-credential

**保存 MCP OAuth 用户凭证**

为 OpenAPI MCP 服务器保存调用方用户的 OAuth2 token。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `server_id` | 路径参数 | ✅ |  |

#### `DELETE` /v1/mcp/server/{server_id}/oauth-user-credential

**删除 Mcp Oauth User Credential**

吊销调用方用户为某个 MCP 服务器保存的 OAuth2 token。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `server_id` | 路径参数 | ✅ |  |

#### `GET` /v1/mcp/server/{server_id}/oauth-user-credential/status

**获取 Mcp Oauth User Credential Status**

检查调用方用户是否已为该 MCP 服务器保存了 OAuth2 凭证。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `server_id` | 路径参数 | ✅ |  |

#### `GET` /v1/mcp/user-credentials

**列表 Mcp User Credentials**

列出调用方用户保存的全部 OAuth2 MCP 凭证。

#### `POST` /v1/mcp/make_public

**将 MCP 服务器设为公开**

允许将 MCP 服务器公开到 AI Hub。

#### `GET` /v1/mcp/discover

**发现 MCP 服务器**

返回用于发现 UI 的常见 MCP 服务器精选列表。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `query` | 查询参数 | ❌ | Search filter for server names and descriptions |
| `category` | 查询参数 | ❌ | Filter by category |

#### `GET` /v1/mcp/openapi-registry

**获取 Openapi Registry**

返回带有 OAuth 2.0 元数据的常见 OpenAPI 接口，供 OpenAPI MCP 选择器使用。

#### `GET` /mcp-rest/tools/list

**列表 Tool Rest Api**

列出全部可用工具，并附上所属服务器信息。

响应示例:
{
    "tools": [
        {
            "name": "create_zap",
            "description": "Create a new zap",
            "inputSchema": "tool_input_schema",
            "mcp_info": {
                "server_name": "zapier",
                "logo_url": "https://www.zapier.com/logo.png",
            }
        }
    ],
    "error": null,
    "message": "Successfully retrieved tools"
}

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `server_id` | 查询参数 | ❌ | The server id to list tools for |

#### `POST` /mcp-rest/tools/call

**通过 REST API 调用工具**

以 REST API 方式，使用给定参数调用指定的 MCP 工具。

#### `POST` /mcp-rest/test/connection

**测试连接**

在新增 MCP 服务器之前测试与其连通性。

#### `POST` /mcp-rest/test/tools/list

**测试工具列表**

在新增 MCP 服务器之前预览其提供的工具。

#### `GET` /authorize

**授权**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `redirect_uri` | 查询参数 | ✅ |  |
| `client_id` | 查询参数 | ❌ |  |
| `state` | 查询参数 | ❌ |  |
| `mcp_server_name` | 查询参数 | ❌ |  |
| `code_challenge` | 查询参数 | ❌ |  |
| `code_challenge_method` | 查询参数 | ❌ |  |
| `response_type` | 查询参数 | ❌ |  |
| `scope` | 查询参数 | ❌ |  |

#### `GET` /{mcp_server_name}/authorize

**授权**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `mcp_server_name` | 路径参数 | ✅ |  |
| `redirect_uri` | 查询参数 | ✅ |  |
| `client_id` | 查询参数 | ❌ |  |
| `state` | 查询参数 | ❌ |  |
| `code_challenge` | 查询参数 | ❌ |  |
| `code_challenge_method` | 查询参数 | ❌ |  |
| `response_type` | 查询参数 | ❌ |  |
| `scope` | 查询参数 | ❌ |  |

#### `POST` /token

**Token 端点**

接收客户端提交的 authorization code 并换取 OAuth token。
通过将 `code_verifier` 转发给上游提供商来支持 PKCE 流程。

1. Call the token endpoint with PKCE parameters
2. Store the user's token in the db - and generate a LiteLLM virtual key
3. Return the token
4. Return a virtual key in this response

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `mcp_server_name` | 查询参数 | ❌ |  |

#### `POST` /{mcp_server_name}/token

**Token 端点**

接收客户端提交的 authorization code 并换取 OAuth token。
通过将 `code_verifier` 转发给上游提供商来支持 PKCE 流程。

1. Call the token endpoint with PKCE parameters
2. Store the user's token in the db - and generate a LiteLLM virtual key
3. Return the token
4. Return a virtual key in this response

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `mcp_server_name` | 路径参数 | ✅ |  |

#### `GET` /callback

**回调**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `code` | 查询参数 | ✅ |  |
| `state` | 查询参数 | ✅ |  |

#### `GET` /.well-known/oauth-protected-resource/mcp/{mcp_server_name}

**OAuth 受保护资源（MCP 标准路径）**

使用标准 MCP URL 模式的 OAuth 受保护资源发现端点。

标准路径：`/mcp/{server_name}`
发现路径：`/.well-known/oauth-protected-resource/mcp/{server_name}`

该端点与 MCP 规范兼容，可配合标准
例如 `mcp-inspector`、VSCode Copilot 等 MCP 客户端。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `mcp_server_name` | 路径参数 | ✅ |  |

#### `GET` /.well-known/oauth-protected-resource

**OAuth 受保护资源（MCP 路径）**

使用 LiteLLM 旧版 URL 模式的 OAuth 受保护资源发现端点。

旧版路径：`/{server_name}/mcp`
发现路径：`/.well-known/oauth-protected-resource/{server_name}/mcp`

该端点保留以兼容旧版。新的集成应
请改用标准的 MCP 路径模式：`/mcp/{server_name}`。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `mcp_server_name` | 查询参数 | ❌ |  |

#### `GET` /.well-known/oauth-protected-resource/{mcp_server_name}/mcp

**OAuth 受保护资源（MCP 路径）**

使用 LiteLLM 旧版 URL 模式的 OAuth 受保护资源发现端点。

旧版路径：`/{server_name}/mcp`
发现路径：`/.well-known/oauth-protected-resource/{server_name}/mcp`

该端点保留以兼容旧版。新的集成应
请改用标准的 MCP 路径模式：`/mcp/{server_name}`。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `mcp_server_name` | 路径参数 | ✅ |  |

#### `GET` /.well-known/oauth-authorization-server/mcp/{mcp_server_name}

**OAuth 授权服务器（MCP 标准路径）**

使用标准 MCP URL 模式的 OAuth 授权服务器发现端点。

标准路径：`/mcp/{server_name}`
发现路径：`/.well-known/oauth-authorization-server/mcp/{server_name}`

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `mcp_server_name` | 路径参数 | ✅ |  |

#### `GET` /.well-known/oauth-authorization-server

**OAuth 授权服务器（MCP 路径）**

OAuth 授权服务器发现端点。

同时支持旧版路径 `/{server_name}` 和根端点。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `mcp_server_name` | 查询参数 | ❌ |  |

#### `GET` /.well-known/oauth-authorization-server/{mcp_server_name}

**OAuth 授权服务器（MCP 路径）**

OAuth 授权服务器发现端点。

同时支持旧版路径 `/{server_name}` 和根端点。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `mcp_server_name` | 路径参数 | ✅ |  |

#### `GET` /.well-known/jwks.json

**JWKS JSON**

JSON Web Key Set 端点。

返回 `MCPJWTSigner` 用于签发外发 MCP Token 的 RSA 公钥。
MCP 服务器与网关使用该端点校验 LiteLLM 下发的 JWT。

若未配置 `MCPJWTSigner`，返回空密钥集合。

#### `GET` /.well-known/oauth-authorization-server/{mcp_server_name}/mcp

**OAuth 授权服务器（旧版路径）**

面向旧版 `/{server_name}/mcp` 路径的 OAuth 授权服务器发现。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `mcp_server_name` | 路径参数 | ✅ |  |

#### `POST` /register

**注册客户端**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `mcp_server_name` | 查询参数 | ❌ |  |

#### `POST` /{mcp_server_name}/register

**注册客户端**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `mcp_server_name` | 路径参数 | ✅ |  |

### Claude Code 市场

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/claude-code/marketplace.json` | 获取 Marketplace |
| `POST` | `/claude-code/plugins` | Register Plugin |
| `GET` | `/claude-code/plugins` | 列表 Plugins |
| `GET` | `/claude-code/plugins/{plugin_name}` | 获取 Plugin |
| `DELETE` | `/claude-code/plugins/{plugin_name}` | 删除 Plugin |
| `POST` | `/claude-code/plugins/{plugin_name}/enable` | Enable Plugin |
| `POST` | `/claude-code/plugins/{plugin_name}/disable` | Disable Plugin |

#### `GET` /claude-code/marketplace.json

**获取 Marketplace**

提供 `marketplace.json`，用于 Claude Code 插件发现。

该端点由 Claude Code CLI 在用户执行如下命令时访问：
- claude plugin marketplace add <url>
- claude plugin install <name>@<marketplace>

返回：
    Marketplace catalog with list of available plugins and their git sources.

示例：
    ```bash
    claude plugin marketplace add http://localhost:4000/claude-code/marketplace.json
    claude plugin install my-plugin@litellm
    ```

#### `POST` /claude-code/plugins

**注册插件**

在 LiteLLM Marketplace 中注册插件。

LiteLLM 作为注册/发现层，插件实际托管在 GitHub / GitLab / Bitbucket；用户安装时，Claude Code 会从 git 源直接 clone。

参数：
    - name: Plugin name (kebab-case)
    - source: Git source reference (github, url, or git-subdir format)
    - version: Semantic version (optional)
    - description: Plugin description (optional)
    - author: Author information (optional)
    - homepage: Plugin homepage URL (optio...

#### `GET` /claude-code/plugins

**列表 Plugins**

列出 Marketplace 中的全部插件。

参数：
    - enabled_only: If true, only return enabled plugins

返回：
    List of plugins with their metadata.

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `enabled_only` | 查询参数 | ❌ |  |

#### `GET` /claude-code/plugins/{plugin_name}

**获取 Plugin**

获取指定插件的详细信息。

参数：
    - plugin_name: The name of the plugin

返回：
    Plugin details including source and metadata.

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `plugin_name` | 路径参数 | ✅ |  |

#### `DELETE` /claude-code/plugins/{plugin_name}

**删除 Plugin**

从 Marketplace 中删除插件。

参数：
    - plugin_name: The name of the plugin to delete

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `plugin_name` | 路径参数 | ✅ |  |

#### `POST` /claude-code/plugins/{plugin_name}/enable

**启用插件**

启用已停用的插件。

参数：
    - plugin_name: The name of the plugin to enable

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `plugin_name` | 路径参数 | ✅ |  |

#### `POST` /claude-code/plugins/{plugin_name}/disable

**停用插件**

停用插件（不删除）。

参数：
    - plugin_name: The name of the plugin to disable

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `plugin_name` | 路径参数 | ✅ |  |

