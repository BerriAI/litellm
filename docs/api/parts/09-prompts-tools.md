## 提示词与工具

### 提示词管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/prompts/list` | 提示词列表 |
| `GET` | `/prompts/{prompt_id}/versions` | 获取 Prompt Versions |
| `GET` | `/prompts/{prompt_id}/info` | 获取 Prompt Info |
| `GET` | `/prompts/{prompt_id}` | 获取 Prompt Info |
| `PUT` | `/prompts/{prompt_id}` | 更新提示词 |
| `DELETE` | `/prompts/{prompt_id}` | 删除提示词 |
| `PATCH` | `/prompts/{prompt_id}` | 部分更新提示词 |
| `POST` | `/prompts` | 创建提示词 |
| `POST` | `/prompts/test` | 测试提示词 |

#### `GET` /prompts/list

**提示词列表**

列出当前 proxy 上可用的所有提示词。

👉 [提示词文档](https://docs.litellm.ai/docs/proxy/prompt_management)

请求示例：
```bash
curl -X GET "http://localhost:4000/prompts/list" -H "Authorization: Bearer <your_api_key>"
```

响应示例：
```
{
    "prompts": [
        {
            "prompt_id": "my_prompt_id",
            "litellm_params": {
                "prompt_id": "my_prompt_id",
                "prompt_integration": "dotprompt",
                "prompt_dir...
```

#### `GET` /prompts/{prompt_id}/versions

**获取提示词的历史版本**

根据基础 prompt ID 获取指定提示词的所有版本。

👉 [提示词文档](https://docs.litellm.ai/docs/proxy/prompt_management)

请求示例：
```bash
curl -X GET "http://localhost:4000/prompts/jack_success/versions" \
    -H "Authorization: Bearer <your_api_key>"
```

响应示例：
```
{
    "prompts": [
        {
            "prompt_id": "jack_success.v1",
            "litellm_params": {...},
            "prompt_info": {"prompt_type": "db"},
            "created_at": "2023-11-09T12:3...
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `prompt_id` | 路径参数 | ✅ |  |

#### `GET` /prompts/{prompt_id}/info

**获取提示词详情**

根据 ID 获取指定提示词的详细信息（含提示词正文）。

👉 [提示词文档](https://docs.litellm.ai/docs/proxy/prompt_management)

请求示例：
```bash
curl -X GET "http://localhost:4000/prompts/my_prompt_id/info" \
    -H "Authorization: Bearer <your_api_key>"
```

响应示例：
```
{
    "prompt_id": "my_prompt_id",
    "litellm_params": {
        "prompt_id": "my_prompt_id",
        "prompt_integration": "do...
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `prompt_id` | 路径参数 | ✅ |  |

#### `GET` /prompts/{prompt_id}

**获取提示词详情**

根据 ID 获取指定提示词的详细信息（含提示词正文）。

👉 [提示词文档](https://docs.litellm.ai/docs/proxy/prompt_management)

请求示例：
```bash
curl -X GET "http://localhost:4000/prompts/my_prompt_id/info" \
    -H "Authorization: Bearer <your_api_key>"
```

响应示例：
```
{
    "prompt_id": "my_prompt_id",
    "litellm_params": {
        "prompt_id": "my_prompt_id",
        "prompt_integration": "do...
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `prompt_id` | 路径参数 | ✅ |  |

#### `PUT` /prompts/{prompt_id}

**更新提示词**

更新一个已存在的提示词。

👉 [提示词文档](https://docs.litellm.ai/docs/proxy/prompt_management)

请求示例：
```bash
curl -X PUT "http://localhost:4000/prompts/my_prompt_id" \
    -H "Authorization: Bearer <your_api_key>" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt_id": "my_prompt",
        "litellm_params": {
            "prompt_id": "my_prompt",
                "prompt_integration": "dotprompt",
                "prompt_directory": "/path/to/prompts"
            ...
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `prompt_id` | 路径参数 | ✅ |  |

#### `DELETE` /prompts/{prompt_id}

**删除提示词**

删除指定提示词。

👉 [提示词文档](https://docs.litellm.ai/docs/proxy/prompt_management)

请求示例：
```bash
curl -X DELETE "http://localhost:4000/prompts/my_prompt_id" \
    -H "Authorization: Bearer <your_api_key>"
```

响应示例：
```json
{
    "message": "Prompt my_prompt_id deleted successfully"
}
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `prompt_id` | 路径参数 | ✅ |  |

#### `PATCH` /prompts/{prompt_id}

**部分更新提示词**

部分字段地更新一个已存在的提示词，无需提交整个对象。

👉 [提示词文档](https://docs.litellm.ai/docs/proxy/prompt_management)

只能更新以下字段：
- `litellm_params`：提示词的 LiteLLM 参数
- `prompt_info`：提示词的附加信息

请求示例：
```bash
curl -X PATCH "http://localhost:4000/prompts/my_prompt_id" \
    -H "Authorization: Bearer <your_api_key>" \
    -H "Content...
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `prompt_id` | 路径参数 | ✅ |  |

#### `POST` /prompts

**创建提示词**

创建一个新的提示词。

👉 [提示词文档](https://docs.litellm.ai/docs/proxy/prompt_management)

请求示例：
```bash
curl -X POST "http://localhost:4000/prompts" \
    -H "Authorization: Bearer <your_api_key>" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt_id": "my_prompt",
        "litellm_params": {
            "prompt_id": "json_prompt",
            "prompt_integration": "dotprompt",
            ### EITHER prompt_directory OR prompt_data MUST BE PROVIDED
            "pr...
```

#### `POST` /prompts/test

**测试提示词**

使用变量渲染提示词并触发一次 LLM 调用来测试提示词。

该端点允许在将提示词保存到数据库之前先验证效果。响应始终以流式（stream）返回。

👉 [提示词文档](https://docs.litellm.ai/docs/proxy/prompt_management)

请求示例：
```bash
curl -X POST "http://localhost:4000/prompts/test" \
    -H "Authorization: Bearer <your_api_key>" \
    -H "Content-Type: application/json" \
    -d '{
        "dotprompt_content": "---\nmodel: gpt-4o\ntemperature: 0.7\n---\...
```

### 提示词

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/utils/dotprompt_json_converter` | 将 .prompt 文件转为 JSON |

#### `POST` /utils/dotprompt_json_converter

**将 .prompt 文件转为 JSON**

将一个 `.prompt` 文件转换为等价 JSON 格式。

该端点接受 `.prompt` 文件上传，返回等价的 JSON 结构，可用于数据库存储或程序化使用。

返回的 JSON 结构包含 `content` 和 `metadata` 两个字段。

### 工具管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/v1/tool/policy/options` | 获取工具策略选项 |
| `GET` | `/v1/tool/list` | 列出工具 |
| `GET` | `/v1/tool/{tool_name}/detail` | 获取工具详情 |
| `GET` | `/v1/tool/{tool_name}/logs` | 获取工具调用日志 |
| `GET` | `/v1/tool/{tool_name}` | 获取工具 |
| `POST` | `/v1/tool/policy` | 更新工具策略 |
| `DELETE` | `/v1/tool/{tool_name}/overrides` | 删除工具策略覆盖 |

#### `GET` /v1/tool/policy/options

**获取工具策略选项**

返回可用的输入策略和输出策略选项及其说明。静态数据，不会查数据库。

#### `GET` /v1/tool/list

**列出工具**

列出所有被自动发现的工具及它们的策略。

参数：
- `input_policy`：可选过滤条件，取值为 `trusted` / `untrusted` / `blocked`

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `input_policy` | 查询参数 | ❌ |  |

#### `GET` /v1/tool/{tool_name}/detail

**获取工具详情**

获取单个工具的详细信息，含其策略覆盖（供 UI 详情页使用）。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `tool_name` | 路径参数 | ✅ |  |

#### `GET` /v1/tool/{tool_name}/logs

**获取工具调用日志**

从 `SpendLogToolIndex` 中分页返回使用过该工具的请求消费日志。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `tool_name` | 路径参数 | ✅ |  |
| `page` | 查询参数 | ❌ |  |
| `page_size` | 查询参数 | ❌ |  |
| `start_date` | 查询参数 | ❌ | YYYY-MM-DD |
| `end_date` | 查询参数 | ❌ | YYYY-MM-DD |

#### `GET` /v1/tool/{tool_name}

**获取工具**

获取单个工具的详细信息。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `tool_name` | 路径参数 | ✅ |  |

#### `POST` /v1/tool/policy

**更新工具策略**

为工具设置全局的 `input_policy` 和/或 `output_policy`，或者为指定的 team / key 添加覆盖策略（override）。

参数：
- `tool_name`（str）：要更新的工具名
- `input_policy`（可选）：`trusted` / `untrusted` / `blocked`
- `output_policy`（可选）：`trusted` / `untrusted`
- `team_id`（可选）：传入时仅为该 team 创建或更新覆盖策略
- `key_hash`（可选）：传入时仅为该 key 创建或更新覆盖策略

#### `DELETE` /v1/tool/{tool_name}/overrides

**删除工具策略覆盖**

删除某个工具的策略覆盖。通过 `team_id` 或 `key_hash` 定位被删的覆盖（二者有且仅有一个需传入）。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `tool_name` | 路径参数 | ✅ |  |
| `team_id` | 查询参数 | ❌ | 待删覆盖所属的 Team ID |
| `key_hash` | 查询参数 | ❌ | 待删覆盖对应的 Key hash |

### 搜索工具

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/search_tools/list` | 搜索工具列表 |
| `POST` | `/search_tools` | 创建搜索工具 |
| `PUT` | `/search_tools/{search_tool_id}` | 更新搜索工具 |
| `DELETE` | `/search_tools/{search_tool_id}` | 删除搜索工具 |
| `GET` | `/search_tools/{search_tool_id}` | 获取搜索工具详情 |
| `POST` | `/search_tools/test_connection` | 测试搜索工具连通性 |
| `GET` | `/search_tools/ui/available_providers` | 获取可用的搜索提供商 |

#### `GET` /search_tools/list

**搜索工具列表**

列出数据库和配置文件中可用的所有搜索工具。

请求示例：
```bash
curl -X GET "http://localhost:4000/search_tools/list" -H "Authorization: Bearer <your_api_key>"
```

Example Response:
```
{
    "search_tools": [
        {
            "search_tool_id": "123e4567-e89b-12d3-a456-426614174000",
            "search_tool_name": "litellm-search",
            "litellm_params": {
                "search_provider": "perplexity",
                "api_key": "sk-***",
 ...
```

#### `POST` /search_tools

**创建搜索工具**

创建一个新的搜索工具。

请求示例：
```bash
curl -X POST "http://localhost:4000/search_tools" \
    -H "Authorization: Bearer <your_api_key>" \
    -H "Content-Type: application/json" \
    -d '{
        "search_tool": {
            "search_tool_name": "litellm-search",
            "litellm_params": {
                "search_provider": "perplexity",
                "api_key": "sk-..."
            },
            "search_tool_info": {
                "description": "Perplexity search tool"...
```

#### `PUT` /search_tools/{search_tool_id}

**更新搜索工具**

更新一个已存在的搜索工具。

请求示例：
```bash
curl -X PUT "http://localhost:4000/search_tools/123e4567-e89b-12d3-a456-426614174000" \
    -H "Authorization: Bearer <your_api_key>" \
    -H "Content-Type: application/json" \
    -d '{
        "search_tool": {
            "search_tool_name": "updated-search",
            "litellm_params": {
                "search_provider": "perplexity",
                "api_key": "sk-new-key"
            },
            "search_tool_info": {
         ...
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `search_tool_id` | 路径参数 | ✅ |  |

#### `DELETE` /search_tools/{search_tool_id}

**删除搜索工具**

删除指定搜索工具。

请求示例：
```bash
curl -X DELETE "http://localhost:4000/search_tools/123e4567-e89b-12d3-a456-426614174000" \
    -H "Authorization: Bearer <your_api_key>"
```

响应示例：
```json
{
    "message": "Search tool 123e4567-e89b-12d3-a456-426614174000 deleted successfully",
    "search_tool_name": "litellm-search"
}
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `search_tool_id` | 路径参数 | ✅ |  |

#### `GET` /search_tools/{search_tool_id}

**获取搜索工具详情**

根据 ID 获取指定搜索工具的详细信息。

请求示例：
```bash
curl -X GET "http://localhost:4000/search_tools/123e4567-e89b-12d3-a456-426614174000" \
    -H "Authorization: Bearer <your_api_key>"
```

响应示例：
```
{
    "search_tool_id": "123e4567-e89b-12d3-a456-426614174000",
    "search_tool_name": "litellm-search",
    "litellm_params": {
        "search_provider": "perplexity",
        "api_key": "sk-***"
    },
    "search_tool_info": {
        "descrip...
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `search_tool_id` | 路径参数 | ✅ |  |

#### `POST` /search_tools/test_connection

**测试搜索工具连通性**

使用传入的配置测试对某个搜索提供商的连通性。

会发起一次简单的测试搜索查询，验证 API Key 与配置是否正确。

请求示例：
```bash
curl -X POST "http://localhost:4000/search_tools/test_connection" \
    -H "Authorization: Bearer <your_api_key>" \
    -H "Content-Type: application/json" \
    -d '{
        "litellm_params": {
            "search_provider": "perplexity",
            "api_key": "sk-..."
        }
    }'
```

响应示例（成功）：...

#### `GET` /search_tools/ui/available_providers

**获取可用的搜索提供商**

获取可用的搜索提供商列表及其配置字段。

会从各 transformation 配置中自动发现搜索提供商及其 UI 友好名称。

请求示例：
```bash
curl -X GET "http://localhost:4000/search_tools/ui/available_providers" \
    -H "Authorization: Bearer <your_api_key>"
```

响应示例：
```
{
    "providers": [
        {
            "provider_name": "perplexity",
            "ui_friendly_name": "Perplexity"
        }
        // ...更多提供商...
    ]
}
```

