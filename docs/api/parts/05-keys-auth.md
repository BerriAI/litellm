## 密钥与认证

### 密钥管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/key/generate` | Generate Key Fn |
| `POST` | `/key/service-account/generate` | Generate Service Account Key Fn |
| `POST` | `/key/update` | 更新 Key Fn |
| `POST` | `/key/bulk_update` | Bulk Update Keys |
| `POST` | `/key/delete` | 删除 Key Fn |
| `GET` | `/key/info` | Info Key Fn |
| `POST` | `/key/regenerate` | Regenerate Key Fn |
| `POST` | `/key/{key}/regenerate` | Regenerate Key Fn |
| `POST` | `/key/{key}/reset_spend` | Reset Key Spend Fn |
| `GET` | `/key/list` | 密钥列表 |
| `GET` | `/key/aliases` | Key Aliases |
| `POST` | `/key/block` | 封禁 Key |
| `POST` | `/key/unblock` | 解封 Key |
| `POST` | `/key/health` | Key Health |

#### `POST` /key/generate

**生成 API Key**

根据提供的数据生成一个 API Key。

文档：https://docs.litellm.ai/docs/proxy/virtual_keys

参数：
- duration: Optional[str] - Specify the length of time the token is valid for. You can set duration as seconds ("30s"), minutes ("30m"), hours ("30h"), days ("30d").
- key_alias: Optional[str] - User defined key alias
- key: Optional[str] - User defined key value. If not set, a 16-digit unique sk-key is created for you.
- team_id: Optional[str] - The team id of the key
- user_id: O...

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `litellm-changed-by` | 请求头 | ❌ | The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of |

#### `POST` /key/service-account/generate

**生成服务账号 Key**

根据提供的数据生成一个服务账号（Service Account）API Key。该 Key 不归属任何用户，归属于团队。

为什么使用服务账号 Key？
- Prevent key from being deleted when user is deleted.
- Apply team limits, not team member limits to key.

文档：https://docs.litellm.ai/docs/proxy/virtual_keys

参数：
- duration: Optional[str] - Specify the length of time the token is valid for. You can set duration as seconds ("30s"), minutes ("30m"), hours ("30h"), days ("30d").
- ...

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `litellm-changed-by` | 请求头 | ❌ | The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of |

#### `POST` /key/update

**更新 Key Fn**

更新已有 API Key 的参数。

参数：
- key: str - The key to update
- key_alias: Optional[str] - User-friendly key alias
- user_id: Optional[str] - User ID associated with key
- team_id: Optional[str] - Team ID associated with key
- agent_id: Optional[str] - The agent id associated with the key.
- organization_id: Optional[str] - The organization id of the key.
- budget_id: Optional[str] - The budget id associated with the key. Created by calling `/budget/new`.
- models: Optiona...

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `litellm-changed-by` | 请求头 | ❌ | The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of |

#### `POST` /key/bulk_update

**批量更新 Key**

批量更新多个 Key。

该端点支持在单次请求中更新多个 Key。每个更新独立处理 —— 即使部分更新失败，其余更新仍会成功应用。

参数：
- keys: List[BulkUpdateKeyRequestItem] - List of key update requests, each containing:
    - key: str - The key identifier (token) to update
    - budget_id: Optional[str] - Budget ID associated with the key
    - max_budget: Optional[float] - Max budget for key
    - team_id: Optional[str] ...

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `litellm-changed-by` | 请求头 | ❌ | The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of |

#### `POST` /key/delete

**删除 Key Fn**

从 Key 管理系统中删除指定 Key。

参数：:
- keys (List[str]): A list of keys or hashed keys to delete. 示例 {"keys": ["sk-QWrxEynunsNpV1zT48HIrw", "837e17519f44683334df5291321d97b8bf1098cd490e49e215f6fea935aa28be"]}
- key_aliases (List[str]): A list of key aliases to delete. Can be passed instead of `keys`.示例 {"key_aliases": ["alias1", "alias2"]}

返回：
- deleted_keys (List[str]): A list of deleted keys. 示例 {"deleted_keys": ["sk-QWrxEynunsNpV1zT48HIrw", "837e1751...

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `litellm-changed-by` | 请求头 | ❌ | The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of |

#### `GET` /key/info

**Key 信息**

获取指定 Key 的信息。
参数：
    key: Optional[str] = Query parameter representing the key in the request
    user_api_key_dict: UserAPIKeyAuth = Dependency representing the user's API key
返回：
    Dict containing the key and its associated information

Curl 示例:
```
curl -X GET "http://0.0.0.0:4000/key/info?key=sk-test-example-key-123" -H "Authorization: Bearer sk-1234"
```

Curl 示例（若未显式传入 Key，则使用 Authorization Header 中的 Key）。
```
curl ...
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `key` | 查询参数 | ❌ | Key in the request parameters |

#### `POST` /key/regenerate

**重新生成 Key**

重新生成已有的 API Key，可选地同时更新其参数。

参数：
- key: str (path parameter) - The key to regenerate
- data: Optional[RegenerateKeyRequest] - Request body containing optional parameters to update
    - key: Optional[str] - The key to regenerate.
    - new_master_key: Optional[str] - The new master key to use, if key is the master key.
    - new_key: Optional[str] - The new key to use, if key is not the master key. If both set, new_master_key will be used.
   ...

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `key` | 查询参数 | ❌ |  |
| `litellm-changed-by` | 请求头 | ❌ | The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of |

#### `POST` /key/{key}/regenerate

**重新生成 Key**

重新生成已有的 API Key，可选地同时更新其参数。

参数：
- key: str (path parameter) - The key to regenerate
- data: Optional[RegenerateKeyRequest] - Request body containing optional parameters to update
    - key: Optional[str] - The key to regenerate.
    - new_master_key: Optional[str] - The new master key to use, if key is the master key.
    - new_key: Optional[str] - The new key to use, if key is not the master key. If both set, new_master_key will be used.
   ...

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `key` | 路径参数 | ✅ |  |
| `litellm-changed-by` | 请求头 | ❌ | The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of |

#### `POST` /key/{key}/reset_spend

**重置 Key 花费**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `key` | 路径参数 | ✅ |  |
| `litellm-changed-by` | 请求头 | ❌ | The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of |

#### `GET` /key/list

**密钥列表**

列出指定用户 / 团队 / 组织下的全部 Key。

参数：
    expand: Optional[List[str]] - Expand related objects (e.g. 'user' to include user information)
    status: Optional[str] - Filter by status. Currently supports "deleted" to query deleted keys.

返回：
    {
        "keys": List[str] or List[UserAPIKeyAuth],
        "total_count": int,
        "current_page": int,
        "total_pages": int,
    }

当 `expand` 包含 `"user"` 时，每个 key 对象会带上一个 `user` 字段（含用户详情）。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `page` | 查询参数 | ❌ | Page number |
| `size` | 查询参数 | ❌ | Page size |
| `user_id` | 查询参数 | ❌ | Filter keys by user ID |
| `team_id` | 查询参数 | ❌ | Filter keys by team ID |
| `organization_id` | 查询参数 | ❌ | Filter keys by organization ID |
| `key_hash` | 查询参数 | ❌ | Filter keys by key hash |
| `key_alias` | 查询参数 | ❌ | Filter keys by key alias |
| `return_full_object` | 查询参数 | ❌ | Return full key object |
| `include_team_keys` | 查询参数 | ❌ | Include all keys for teams that user is an admin of. |
| `include_created_by_keys` | 查询参数 | ❌ | Include keys created by the user |

#### `GET` /key/aliases

**Key 别名**

列出 key aliases with pagination and optional search.

非管理员用户只能看到自己拥有的 Key，或归属于自己的团队/组织的 Key 的别名。
用户所属团队的 Key。

返回：
    {
        "aliases": List[str],
        "total_count": int,
        "current_page": int,
        "total_pages": int,
        "size": int,
    }

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `page` | 查询参数 | ❌ | Page number |
| `size` | 查询参数 | ❌ | Page size |
| `search` | 查询参数 | ❌ | Search key aliases (case-insensitive partial match) |

#### `POST` /key/block

**封禁 Key**

阻断 Virtual Key，使其无法发起任何请求。

参数：
- key: str - The key to block. Can be either the unhashed key (sk-...) or the hashed key value

 示例：
```bash
curl --location 'http://0.0.0.0:4000/key/block'     --header 'Authorization: Bearer sk-1234'     --header 'Content-Type: application/json'     --data '{
    "key": "sk-Fn8Ej39NxjAXrvpUGKghGw"
}'
```

注意：该端点仅管理员可用（Proxy 管理员、团队管理员或组织管理员可阻断 Key）。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `litellm-changed-by` | 请求头 | ❌ | The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of |

#### `POST` /key/unblock

**解封 Key**

解除 Virtual Key 的阻断，使其可以再次发起请求。

参数：
- key: str - The key to unblock. Can be either the unhashed key (sk-...) or the hashed key value

示例：
```bash
curl --location 'http://0.0.0.0:4000/key/unblock'     --header 'Authorization: Bearer sk-1234'     --header 'Content-Type: application/json'     --data '{
    "key": "sk-Fn8Ej39NxjAXrvpUGKghGw"
}'
```

注意：该端点仅管理员可用（Proxy 管理员、团队管理员或组织管理员可解除阻断）。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `litellm-changed-by` | 请求头 | ❌ | The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of |

#### `POST` /key/health

**Key 健康检查**

检查 Key 的健康状态。

检查：
- If key based logging is configured correctly - sends a test log

用法 

在请求头中传入 Key。

```bash
curl -X POST "http://localhost:4000/key/health"      -H "Authorization: Bearer sk-1234"      -H "Content-Type: application/json"
```

日志回调配置正确时的响应:

```
{
  "key": "healthy",
  "logging_callbacks": {
    "callbacks": [
      "gcs_bucket"
    ],
    "status": "healthy",
    "details": "No logger excep...
```

### 凭据管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/credentials` | 获取 Credentials |
| `POST` | `/credentials` | 创建凭据 |
| `GET` | `/credentials/by_name/{credential_name}` | 获取 Credential By Name |
| `GET` | `/credentials/by_model/{model_id}` | 获取 Credential By Model |
| `DELETE` | `/credentials/{credential_name}` | 删除凭据 |
| `PATCH` | `/credentials/{credential_name}` | 更新凭据 |

#### `GET` /credentials

**获取 Credentials**

[BETA] 该端点处于 Beta 阶段，行为可能在后续版本中变化。

#### `POST` /credentials

**创建凭据**

[BETA] 该端点处于 Beta 阶段，行为可能在后续版本中变化。
将凭证持久化到数据库。
重新加载内存中的凭证。

#### `GET` /credentials/by_name/{credential_name}

**获取 Credential By Name**

[BETA] 该端点处于 Beta 阶段，行为可能在后续版本中变化。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `credential_name` | 路径参数 | ✅ | The credential name, percent-decoded; may contain slashes |

#### `GET` /credentials/by_model/{model_id}

**获取 Credential By Model**

[BETA] 该端点处于 Beta 阶段，行为可能在后续版本中变化。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `model_id` | 路径参数 | ✅ | The model ID to look up credentials for |

#### `DELETE` /credentials/{credential_name}

**删除凭据**

[BETA] 该端点处于 Beta 阶段，行为可能在后续版本中变化。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `credential_name` | 路径参数 | ✅ | The credential name, percent-decoded; may contain slashes |

#### `PATCH` /credentials/{credential_name}

**更新凭据**

[BETA] 该端点处于 Beta 阶段，行为可能在后续版本中变化。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `credential_name` | 路径参数 | ✅ | The credential name, percent-decoded; may contain slashes |

### JWT 密钥映射

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/jwt/key/mapping/new` | 创建 Jwt Key Mapping |
| `POST` | `/jwt/key/mapping/update` | 更新 Jwt Key Mapping |
| `POST` | `/jwt/key/mapping/delete` | 删除 Jwt Key Mapping |
| `GET` | `/jwt/key/mapping/list` | 列表 Jwt Key Mappings |
| `GET` | `/jwt/key/mapping/info` | Info Jwt Key Mapping |

#### `GET` /jwt/key/mapping/list

**列表 Jwt Key Mappings**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `page` | 查询参数 | ❌ | Page number |
| `size` | 查询参数 | ❌ | Page size |

#### `GET` /jwt/key/mapping/info

**JWT Key 映射信息**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `id` | 查询参数 | ✅ |  |

### SSO 单点登录设置

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/get/internal_user_settings` | 获取 Internal User Settings |
| `GET` | `/get/default_team_settings` | 获取 Default Team Settings |
| `PATCH` | `/update/internal_user_settings` | 更新 Internal User Settings |
| `PATCH` | `/update/default_team_settings` | 更新 Default Team Settings |
| `GET` | `/get/sso_settings` | 获取 Sso Settings |
| `PATCH` | `/update/sso_settings` | 更新 Sso Settings |

#### `GET` /get/internal_user_settings

**获取 Internal User Settings**

从 `litellm_settings` 配置中获取全部 SSO 配置。
返回一个结构化对象，包含取值与描述信息，用于 UI 展示。

#### `GET` /get/default_team_settings

**获取 Default Team Settings**

从 `litellm_settings` 配置中获取全部 SSO 配置。
返回一个结构化对象，包含取值与描述信息，用于 UI 展示。

#### `PATCH` /update/internal_user_settings

**更新 Internal User Settings**

更新 SSO 用户的默认内部用户参数。
该配置将对通过 SSO 登录的新用户生效。

#### `PATCH` /update/default_team_settings

**更新 Default Team Settings**

更新 SSO 用户的默认团队参数。
该配置将对通过 SSO 创建的新团队生效。

#### `GET` /get/sso_settings

**获取 Sso Settings**

从专用 SSO 表中获取全部 SSO 配置。
返回一个结构化对象，包含取值与描述信息，用于 UI 展示。

#### `PATCH` /update/sso_settings

**更新 Sso Settings**

更新 SSO 配置（持久化到专用 SSO 表）。

### ✨ SCIM v2（仅企业版）

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/scim/v2` | 获取 Scim Base |
| `GET` | `/scim/v2/ResourceTypes` | 获取 Resource Types |
| `GET` | `/scim/v2/ResourceTypes/{resource_type_id}` | 获取 Resource Type |
| `GET` | `/scim/v2/Schemas` | 获取 Schemas |
| `GET` | `/scim/v2/Schemas/{schema_id}` | 获取 Schema |
| `GET` | `/scim/v2/ServiceProviderConfig` | 获取 Service Provider Config |
| `GET` | `/scim/v2/Users` | 获取用户列表 |
| `POST` | `/scim/v2/Users` | 创建 User |
| `GET` | `/scim/v2/Users/{user_id}` | 获取 User |
| `PUT` | `/scim/v2/Users/{user_id}` | 更新用户 |
| `DELETE` | `/scim/v2/Users/{user_id}` | 删除用户 |
| `PATCH` | `/scim/v2/Users/{user_id}` | Patch User |
| `GET` | `/scim/v2/Groups` | 获取 Groups |
| `POST` | `/scim/v2/Groups` | 创建 Group |
| `GET` | `/scim/v2/Groups/{group_id}` | 获取 Group |
| `PUT` | `/scim/v2/Groups/{group_id}` | 更新 Group |
| `DELETE` | `/scim/v2/Groups/{group_id}` | 删除 Group |
| `PATCH` | `/scim/v2/Groups/{group_id}` | Patch Group |

#### `GET` /scim/v2

**获取 Scim Base**

SCIM v2 的基础发现端点（符合 RFC 7644 第 4 节）。

返回该 SCIM Service Provider 支持的 ResourceType 的 `ListResponse`。
身份提供商（Okta、Azure AD 等）通过该端点进行资源发现。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `feature` | 查询参数 | ❌ |  |

#### `GET` /scim/v2/ResourceTypes

**获取 Resource Types**

SCIM ResourceTypes 端点（符合 RFC 7644 第 4 节）。

返回该 Service Provider 支持的全部 ResourceType 的 `ListResponse`。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `feature` | 查询参数 | ❌ |  |

#### `GET` /scim/v2/ResourceTypes/{resource_type_id}

**获取 Resource Type**

按 RFC 7644 规范，根据 ID 获取单个 ResourceType。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `resource_type_id` | 路径参数 | ✅ |  |
| `feature` | 查询参数 | ❌ |  |

#### `GET` /scim/v2/Schemas

**获取 Schemas**

SCIM Schemas 端点（符合 RFC 7643 第 7 节）。

返回该 Service Provider 支持的全部 Schema 的 `ListResponse`。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `feature` | 查询参数 | ❌ |  |

#### `GET` /scim/v2/Schemas/{schema_id}

**获取 Schema**

按 RFC 7643 第 7 节规范，根据 URI 获取单个 Schema。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `schema_id` | 路径参数 | ✅ |  |
| `feature` | 查询参数 | ❌ |  |

#### `GET` /scim/v2/ServiceProviderConfig

**获取 Service Provider Config**

返回 SCIM Service Provider 配置。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `feature` | 查询参数 | ❌ |  |

#### `GET` /scim/v2/Users

**获取用户列表**

按照 SCIM v2 协议获取用户列表。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `startIndex` | 查询参数 | ❌ |  |
| `count` | 查询参数 | ❌ |  |
| `filter` | 查询参数 | ❌ |  |
| `feature` | 查询参数 | ❌ |  |

#### `POST` /scim/v2/Users

**创建 User**

按照 SCIM v2 协议创建用户。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `feature` | 查询参数 | ❌ |  |

#### `GET` /scim/v2/Users/{user_id}

**获取 User**

按照 SCIM v2 协议根据 ID 获取单个用户。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `user_id` | 路径参数 | ✅ |  |
| `feature` | 查询参数 | ❌ |  |

#### `PUT` /scim/v2/Users/{user_id}

**更新用户**

按照 SCIM v2 协议整体替换更新用户。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `user_id` | 路径参数 | ✅ |  |
| `feature` | 查询参数 | ❌ |  |

#### `DELETE` /scim/v2/Users/{user_id}

**删除用户**

按照 SCIM v2 协议删除用户。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `user_id` | 路径参数 | ✅ |  |
| `feature` | 查询参数 | ❌ |  |

#### `PATCH` /scim/v2/Users/{user_id}

**局部更新用户**

按照 SCIM v2 协议对用户执行 Patch 操作。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `user_id` | 路径参数 | ✅ |  |
| `feature` | 查询参数 | ❌ |  |

#### `GET` /scim/v2/Groups

**获取 Groups**

按照 SCIM v2 协议获取用户组列表。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `startIndex` | 查询参数 | ❌ |  |
| `count` | 查询参数 | ❌ |  |
| `filter` | 查询参数 | ❌ |  |
| `feature` | 查询参数 | ❌ |  |

#### `POST` /scim/v2/Groups

**创建 Group**

按照 SCIM v2 协议创建用户组。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `feature` | 查询参数 | ❌ |  |

#### `GET` /scim/v2/Groups/{group_id}

**获取 Group**

按照 SCIM v2 协议根据 ID 获取单个用户组。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `group_id` | 路径参数 | ✅ |  |
| `feature` | 查询参数 | ❌ |  |

#### `PUT` /scim/v2/Groups/{group_id}

**更新 Group**

按照 SCIM v2 协议更新用户组。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `group_id` | 路径参数 | ✅ |  |
| `feature` | 查询参数 | ❌ |  |

#### `DELETE` /scim/v2/Groups/{group_id}

**删除 Group**

按照 SCIM v2 协议删除用户组。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `group_id` | 路径参数 | ✅ |  |
| `feature` | 查询参数 | ❌ |  |

#### `PATCH` /scim/v2/Groups/{group_id}

**局部更新用户组**

按照 SCIM v2 协议对用户组执行 Patch 操作。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `group_id` | 路径参数 | ✅ |  |
| `feature` | 查询参数 | ❌ |  |

