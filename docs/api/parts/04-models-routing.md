## 模型与路由

### 模型管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/models` | 模型列表 |
| `GET` | `/v1/models` | 模型列表 |
| `GET` | `/models/{model_id}` | 模型信息 |
| `GET` | `/v1/models/{model_id}` | 模型信息 |
| `GET` | `/v1/model/info` | Model Info V1 |
| `GET` | `/model/info` | Model Info V1 |
| `GET` | `/model_group/info` | 模型组信息 |
| `GET` | `/public/model_hub` | Public Model Hub |
| `GET` | `/public/model_hub/info` | Public Model Hub Info |
| `GET` | `/public/litellm_model_cost_map` | 获取 Litellm Model Cost Map |
| `PATCH` | `/model/{model_id}/update` | Patch Model |
| `POST` | `/model/delete` | 删除模型 |
| `POST` | `/model/new` | 添加 New Model |
| `POST` | `/model/update` | 更新模型 |
| `POST` | `/model_group/make_public` | 更新 Public Model Groups |
| `POST` | `/model_hub/update_useful_links` | 更新 Useful Links |
| `POST` | `/access_group/new` | 创建 Model Group |
| `GET` | `/access_group/list` | 访问组列表 |
| `GET` | `/access_group/{access_group}/info` | 获取 Access Group Info |
| `PUT` | `/access_group/{access_group}/update` | 更新访问组 |
| `DELETE` | `/access_group/{access_group}/delete` | 删除访问组 |

#### `GET` /models

**模型列表**

Use `/model/info` - to get detailed model information, example - pricing, mode, etc.

This is just for compatibility with openai projects like aider.

Query 参数：
- include_metadata: Include additional metadata in the response with fallback information
- fallback_type: Type of fallbacks to include ("general", "context_window", "content_policy")
                Defaults to "general" when include_metadata=true
- scope: Optional scope parameter. Currently only accepts "expand".
         When ...

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `return_wildcard_routes` | 查询参数 | ❌ |  |
| `team_id` | 查询参数 | ❌ |  |
| `include_model_access_groups` | 查询参数 | ❌ |  |
| `only_model_access_groups` | 查询参数 | ❌ |  |
| `include_metadata` | 查询参数 | ❌ |  |
| `fallback_type` | 查询参数 | ❌ |  |
| `scope` | 查询参数 | ❌ |  |

#### `GET` /v1/models

**模型列表**

Use `/model/info` - to get detailed model information, example - pricing, mode, etc.

This is just for compatibility with openai projects like aider.

Query 参数：
- include_metadata: Include additional metadata in the response with fallback information
- fallback_type: Type of fallbacks to include ("general", "context_window", "content_policy")
                Defaults to "general" when include_metadata=true
- scope: Optional scope parameter. Currently only accepts "expand".
         When ...

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `return_wildcard_routes` | 查询参数 | ❌ |  |
| `team_id` | 查询参数 | ❌ |  |
| `include_model_access_groups` | 查询参数 | ❌ |  |
| `only_model_access_groups` | 查询参数 | ❌ |  |
| `include_metadata` | 查询参数 | ❌ |  |
| `fallback_type` | 查询参数 | ❌ |  |
| `scope` | 查询参数 | ❌ |  |

#### `GET` /models/{model_id}

**模型信息**

Retrieve information about a specific model accessible to your API key.

Returns model details only if the model is available to your API key/team.
Returns 404 if the model doesn't exist or is not accessible.

Follows OpenAI API specification for individual model retrieval.
https://platform.openai.com/docs/api-reference/models/retrieve

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `model_id` | 路径参数 | ✅ |  |

#### `GET` /v1/models/{model_id}

**模型信息**

Retrieve information about a specific model accessible to your API key.

Returns model details only if the model is available to your API key/team.
Returns 404 if the model doesn't exist or is not accessible.

Follows OpenAI API specification for individual model retrieval.
https://platform.openai.com/docs/api-reference/models/retrieve

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `model_id` | 路径参数 | ✅ |  |

#### `GET` /v1/model/info

**Model Info V1**

Provides more info about each model in /models, including config.yaml descriptions (except api key and api base)

参数：
    litellm_model_id: Optional[str] = None (this is the value of `x-litellm-model-id` returned in response headers)

    - When litellm_model_id is passed, it will return the info for that specific model
    - When litellm_model_id is not passed, it will return the info for all models

返回：
    Returns a dictionary containing information about each model.

Example Res...

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `litellm_model_id` | 查询参数 | ❌ |  |

#### `GET` /model/info

**Model Info V1**

Provides more info about each model in /models, including config.yaml descriptions (except api key and api base)

参数：
    litellm_model_id: Optional[str] = None (this is the value of `x-litellm-model-id` returned in response headers)

    - When litellm_model_id is passed, it will return the info for that specific model
    - When litellm_model_id is not passed, it will return the info for all models

返回：
    Returns a dictionary containing information about each model.

Example Res...

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `litellm_model_id` | 查询参数 | ❌ |  |

#### `GET` /model_group/info

**模型组信息**

Get information about all the deployments on litellm proxy, including config.yaml descriptions (except api key and api base)

- /model_group/info returns all model groups. End users of proxy should use /model_group/info since those models will be used for /chat/completions, /embeddings, etc.
- /model_group/info?model_group=rerank-english-v3.0 returns all model groups for a specific model group (`model_name` in config.yaml)



请求示例 (All Models):
```shell
curl -X 'GET'     'http://local...
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `model_group` | 查询参数 | ❌ |  |

#### `GET` /public/litellm_model_cost_map

**获取 Litellm Model Cost Map**

Public endpoint to get the LiteLLM model cost map.
Returns pricing information for all supported models.

#### `PATCH` /model/{model_id}/update

**Patch Model**

PATCH Endpoint for partial model updates.

Only updates the fields specified in the request while preserving other existing values.
Follows proper PATCH semantics by only modifying provided fields.

参数：
    model_id: The ID of the model to update
    patch_data: The fields to update and their new values
    user_api_key_dict: User authentication information

返回：
    Updated model information

Raises:
    ProxyException: For various error conditions including authentication and database er...

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `model_id` | 路径参数 | ✅ |  |

#### `POST` /model/delete

**删除模型**

Allows deleting models in the model list in the config.yaml

#### `POST` /model/new

**添加 New Model**

Allows adding new models to the model list in the config.yaml

#### `POST` /model/update

**更新模型**

Edit existing model params

#### `POST` /model_group/make_public

**更新 Public Model Groups**

Update which model groups are public

#### `POST` /model_hub/update_useful_links

**更新 Useful Links**

Update useful links

#### `POST` /access_group/new

**创建 Model Group**

Create a new access group containing multiple model names.

An access group is a named collection of model groups that can be referenced
by teams/keys for simplified access control.

示例：
```bash
curl -X POST 'http://localhost:4000/access_group/new' \
  -H 'Authorization: Bearer sk-1234' \
  -H 'Content-Type: application/json' \
  -d '{
    "access_group": "production-models",
    "model_names": ["gpt-4", "claude-3-opus", "gemini-pro"]
  }'
```

参数：
- access_group: str - The access g...

#### `GET` /access_group/list

**访问组列表**

List all access groups.

Returns a list of all access groups with their model names and deployment counts.

示例：
```bash
curl -X GET 'http://localhost:4000/access_group/list' \
  -H 'Authorization: Bearer sk-1234'
```

返回：
- ListAccessGroupsResponse with all access groups

#### `GET` /access_group/{access_group}/info

**获取 Access Group Info**

Get information about a specific access group.

示例：
```bash
curl -X GET 'http://localhost:4000/access_group/production-models/info' \
  -H 'Authorization: Bearer sk-1234'
```

参数：
- access_group: str - The access group name (URL path parameter)

返回：
- AccessGroupInfo with the access group details

Raises:
- HTTPException 404: If access group not found

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `access_group` | 路径参数 | ✅ |  |

#### `PUT` /access_group/{access_group}/update

**更新访问组**

Update an access group's model names.

This will:
1. Remove the access group from all current deployments
2. Add the access group to all deployments for the new model_names list

示例：
```bash
curl -X PUT 'http://localhost:4000/access_group/production-models/update' \
  -H 'Authorization: Bearer sk-1234' \
  -H 'Content-Type: application/json' \
  -d '{
    "model_names": ["gpt-4", "claude-3-sonnet"]
  }'
```

参数：
- access_group: str - The access group name (URL path parameter)
- mode...

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `access_group` | 路径参数 | ✅ |  |

#### `DELETE` /access_group/{access_group}/delete

**删除访问组**

Delete an access group.

Removes the access group from all deployments that have it.

示例：
```bash
curl -X DELETE 'http://localhost:4000/access_group/production-models/delete' \
  -H 'Authorization: Bearer sk-1234'
```

参数：
- access_group: str - The access group name (URL path parameter)

返回：
- DeleteModelGroupResponse with deletion details

Raises:
- HTTPException 404: If access group not found

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `access_group` | 路径参数 | ✅ |  |

### 路由设置

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/router/settings` | 获取 Router Settings |
| `GET` | `/router/fields` | 获取 Router Fields |

#### `GET` /router/settings

**获取 Router Settings**

Get router configuration and available settings.

返回：
- fields: List of all configurable router settings with their metadata (type, description, default, options)
          The routing_strategy field includes available options extracted from the Router class
- current_values: Current values of router settings from config

#### `GET` /router/fields

**获取 Router Fields**

Get router settings field definitions without values.

Returns only the field metadata (type, description, default, options) without
populating field_value. This is useful for UI components that need to know
what fields to render, but will get the actual values from a different endpoint.

返回：
- fields: List of all configurable router settings with their metadata (type, description, default, options)
          The routing_strategy field includes available options extracted from the Router cl...

### 降级管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/fallback` | 创建 Fallback |
| `GET` | `/fallback/{model}` | 获取 Fallback |
| `DELETE` | `/fallback/{model}` | 删除 Fallback |

#### `POST` /fallback

**创建 Fallback**

Create or update fallbacks for a specific model.

This endpoint allows you to configure fallback models separately from the general config.
Fallbacks are triggered when a model call fails after retries.

**请求示例：**
```json
{
    "model": "gpt-3.5-turbo",
    "fallback_models": ["gpt-4", "claude-3-haiku"],
    "fallback_type": "general"
}
```

**Fallback Types:**
- `general`: Standard fallbacks for any error (default)
- `context_window`: Fallbacks specifically for context window exceede...

#### `GET` /fallback/{model}

**获取 Fallback**

Get fallback configuration for a specific model.

**参数：**
- `model`: The model name to get fallbacks for
- `fallback_type`: Type of fallback to retrieve (query parameter)

**示例：**
```
GET /fallback/gpt-3.5-turbo?fallback_type=general
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `model` | 路径参数 | ✅ |  |
| `fallback_type` | 查询参数 | ❌ |  |

#### `DELETE` /fallback/{model}

**删除 Fallback**

Delete fallback configuration for a specific model.

**参数：**
- `model`: The model name to delete fallbacks for
- `fallback_type`: Type of fallback to delete (query parameter)

**示例：**
```
DELETE /fallback/gpt-3.5-turbo?fallback_type=general
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `model` | 路径参数 | ✅ |  |
| `fallback_type` | 查询参数 | ❌ |  |

### 供应商

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/public/providers` | 获取 Supported Providers |
| `GET` | `/public/providers/fields` | 获取 Provider Fields |

#### `GET` /public/providers

**获取 Supported Providers**

Return a sorted list of all providers supported by LiteLLM.

#### `GET` /public/providers/fields

**获取 Provider Fields**

Return provider metadata required by the dashboard create-model flow.

### 缓存

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/cache/ping` | Cache Ping |
| `POST` | `/cache/delete` | Cache Delete |
| `GET` | `/cache/redis/info` | Cache Redis Info |
| `POST` | `/cache/flushall` | Cache Flushall |

#### `GET` /cache/ping

**Cache Ping**

Endpoint for checking if cache can be pinged

#### `POST` /cache/delete

**Cache Delete**

Endpoint for deleting a key from the cache. All responses from litellm proxy have `x-litellm-cache-key` in the headers

参数：
- **keys**: *Optional[List[str]]* - A list of keys to delete from the cache. Example {"keys": ["key1", "key2"]}

```shell
curl -X POST "http://0.0.0.0:4000/cache/delete"     -H "Authorization: Bearer sk-1234"     -d '{"keys": ["key1", "key2"]}'
```

#### `GET` /cache/redis/info

**Cache Redis Info**

Endpoint for getting /redis/info

#### `POST` /cache/flushall

**Cache Flushall**

A function to flush all items from the cache. (All items will be deleted from the cache with this)
Raises HTTPException if the cache is not initialized or if the cache type does not support flushing.
Returns a dictionary with the status of the operation.

Usage:
```
curl -X POST http://0.0.0.0:4000/cache/flushall -H "Authorization: Bearer sk-1234"
```

### 缓存设置

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/cache/settings` | 获取 Cache Settings |
| `POST` | `/cache/settings` | 更新 Cache Settings |
| `POST` | `/cache/settings/test` | Test Cache Connection |

#### `GET` /cache/settings

**获取 Cache Settings**

Get cache configuration and available settings.

返回：
- fields: List of all configurable cache settings with their metadata (type, description, default, options)
- current_values: Current values of cache settings from database

#### `POST` /cache/settings

**更新 Cache Settings**

Save cache settings to database and initialize cache.

This endpoint:
1. Encrypts sensitive fields (passwords, etc.)
2. Saves to LiteLLM_CacheConfig table
3. Reinitializes cache with new settings

#### `POST` /cache/settings/test

**Test Cache Connection**

Test cache connection with provided credentials.

Creates a temporary cache instance and uses its test_connection method
to verify the credentials work without affecting global state.

