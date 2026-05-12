## 用户与团队管理

### 内部用户管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/user/new` | 创建用户 |
| `GET` | `/user/info` | 用户信息 |
| `GET` | `/v2/user/info` | User Info V2 |
| `POST` | `/user/update` | User Update |
| `POST` | `/user/bulk_update` | Bulk User Update |
| `GET` | `/user/list` | 获取用户列表 |
| `POST` | `/user/delete` | 删除用户 |
| `GET` | `/user/daily/activity` | 获取 User Daily Activity |
| `GET` | `/user/daily/activity/aggregated` | 获取 User Daily Activity Aggregated |
| `GET` | `/user/available_users` | Available Enterprise Users |

#### `POST` /user/new

**创建用户**

用于创建一个带有预算的内部用户（INTERNAL）。
内部用户（Internal User）可访问 LiteLLM 管理后台，创建 Key、申请模型访问权限。
该接口会创建新用户并为其生成新的 API Key，并返回该新 Key。

返回 user id、预算信息以及新生成的 Key。

参数：
- user_id: Optional[str] - Specify a user id. If not set, a unique id will be generated.
- user_alias: Optional[str] - A descriptive name for you to know who this user id refers to.
- teams: Optional[list] - specify...

#### `GET` /user/info

**用户信息**

[10/07/2024]
注意：如需获取全部用户（并分页），请使用 `/user/list` 端点。


用于获取用户信息（用户记录 + 该用户名下的所有 Key 信息）。

请求示例
```
curl -X GET 'http://localhost:4000/user/info?user_id=krrish7%40berri.ai'     --header 'Authorization: Bearer sk-1234'
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `user_id` | 查询参数 | ❌ | User ID in the request parameters |

#### `GET` /v2/user/info

**用户信息 V2**

轻量版获取用户信息的端点，仅返回 user 对象（不包含 key/team 对象）。

`/user/info` 的 v2 版本，避免「万能端点」问题。
与旧端点不同，旧端点会将全部 Key 和团队数据载入内存。

访问控制：
- Proxy admins can query any user
- Team admins can query users within their teams
- Internal users can only query themselves (omit user_id or pass own)
- Returns 404 for non-existent users or unauthorized access

请求示例:
```
...
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `user_id` | 查询参数 | ❌ | User ID in the request parameters |

#### `POST` /user/update

**更新用户**

Curl 示例 

```
curl --location 'http://0.0.0.0:4000/user/update'     --header 'Authorization: Bearer sk-1234'     --header 'Content-Type: application/json'     --data '{
    "user_id": "test-litellm-user-4",
    "user_role": "proxy_admin_viewer"
}'
```

参数：
    - user_id: Optional[str] - Specify a user id. If not set, a unique id will be generated.
    - user_email: Optional[str] - Specify a user email.
    - password: Optional[str] - Specify a user password.
    - user_alias: Option...

#### `POST` /user/bulk_update

**批量更新用户**

批量更新多个用户。

该端点支持在单次请求中更新多个用户。每个更新
独立处理：即使部分更新失败，其余更新仍会成功应用。

参数：
- users: Optional[List[UpdateUserRequest]] - List of specific user update requests
- all_users: Optional[bool] - Set to true to update all users in the system
- user_updates: Optional[UpdateUserRequest] - Updates to apply when all_users=True

返回：
- results: List of individual update ...

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `litellm-changed-by` | 请求头 | ❌ | The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of |

#### `GET` /user/list

**获取用户列表**

分页获取用户列表（支持过滤与排序）。

参数：
    role: Optional[str]
        Filter users by role. Can be one of:
        - proxy_admin
        - proxy_admin_viewer
        - internal_user
        - internal_user_viewer
    user_ids: Optional[str]
        Get list of users by user_ids. Comma separated list of user_ids.
    sso_ids: Optional[str]
        Get list of users by sso_ids. Comma separated list of sso_ids.
    user_email: Optional[str]
        Filter us...

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `role` | 查询参数 | ❌ | Filter users by role |
| `user_ids` | 查询参数 | ❌ | Get list of users by user_ids |
| `sso_user_ids` | 查询参数 | ❌ | Get list of users by sso_user_id |
| `user_email` | 查询参数 | ❌ | Filter users by partial email match |
| `team` | 查询参数 | ❌ | Filter users by team id |
| `page` | 查询参数 | ❌ | Page number |
| `page_size` | 查询参数 | ❌ | Number of items per page |
| `sort_by` | 查询参数 | ❌ | Column to sort by (e.g. 'user_id', 'user_email', 'created_at', 'spend') |
| `sort_order` | 查询参数 | ❌ | Sort order ('asc' or 'desc') |
| `organization_ids` | 查询参数 | ❌ | Filter users by organization membership. Comma-separated list of org IDs. |

#### `POST` /user/delete

**删除用户**

删除用户及其关联的全部 Key。

```
curl --location 'http://0.0.0.0:4000/user/delete' 
--header 'Authorization: Bearer sk-1234' 
--header 'Content-Type: application/json' 
--data-raw '{
    "user_ids": ["45e3e396-ee08-4a61-a88e-16b3ce7e0849"]
}'
```

参数：
- user_ids: List[str] - The list of user id's to be deleted.

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `litellm-changed-by` | 请求头 | ❌ | The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of |

#### `GET` /user/daily/activity

**获取 User Daily Activity**

[BETA] This is a beta endpoint. It will change.

用于优化按用户查询花费数据以做分析。

返回：
(by date)
- spend
- prompt_tokens
- completion_tokens
- cache_read_input_tokens
- cache_creation_input_tokens
- total_tokens
- api_requests
- breakdown by model, api_key, provider

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `start_date` | 查询参数 | ❌ | Start date in YYYY-MM-DD format |
| `end_date` | 查询参数 | ❌ | End date in YYYY-MM-DD format |
| `model` | 查询参数 | ❌ | Filter by specific model |
| `api_key` | 查询参数 | ❌ | Filter by specific API key |
| `user_id` | 查询参数 | ❌ | Filter by specific user ID. Admins can filter by any user or omit for global view. Non-admins must p |
| `page` | 查询参数 | ❌ | Page number for pagination |
| `page_size` | 查询参数 | ❌ | Items per page |
| `timezone` | 查询参数 | ❌ | Timezone offset in minutes from UTC (e.g., 480 for PST). Matches JavaScript's Date.getTimezoneOffset |

#### `GET` /user/daily/activity/aggregated

**获取 User Daily Activity Aggregated**

返回用户每日活动的聚合分析数据（不分页）。
返回与分页端点一致的响应结构，但分页元信息固定为单页。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `start_date` | 查询参数 | ❌ | Start date in YYYY-MM-DD format |
| `end_date` | 查询参数 | ❌ | End date in YYYY-MM-DD format |
| `model` | 查询参数 | ❌ | Filter by specific model |
| `api_key` | 查询参数 | ❌ | Filter by specific API key |
| `user_id` | 查询参数 | ❌ | Filter by specific user ID. Admins can filter by any user or omit for global view. Non-admins must p |
| `timezone` | 查询参数 | ❌ | Timezone offset in minutes from UTC (e.g., 480 for PST). Matches JavaScript's Date.getTimezoneOffset |

#### `GET` /user/available_users

**可用企业版用户**

对于设置了 `max_users` 的 Key，返回允许使用该 Key 的用户列表。

### 团队管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/team/new` | 创建团队 |
| `POST` | `/team/update` | 更新团队 |
| `POST` | `/team/member_add` | 添加团队成员 |
| `POST` | `/team/member_delete` | 删除团队成员 |
| `POST` | `/team/member_update` | 更新团队成员 |
| `POST` | `/team/bulk_member_add` | Bulk Team Member Add |
| `POST` | `/team/delete` | 删除团队 |
| `GET` | `/team/info` | 团队信息 |
| `POST` | `/team/block` | 封禁团队 |
| `POST` | `/team/unblock` | 解封团队 |
| `GET` | `/v2/team/list` | 列表 Team V2 |
| `GET` | `/team/list` | 团队列表 |
| `POST` | `/team/model/add` | Team Model Add |
| `POST` | `/team/model/delete` | Team Model Delete |
| `GET` | `/team/permissions_list` | Team Member Permissions |
| `POST` | `/team/permissions_update` | 更新 Team Member Permissions |
| `GET` | `/team/daily/activity` | 获取 Team Daily Activity |
| `POST` | `/team/{team_id}/callback` | 添加 Team Callbacks |
| `GET` | `/team/{team_id}/callback` | 获取 Team Callbacks |
| `POST` | `/team/{team_id}/disable_logging` | Disable Team Logging |

#### `POST` /team/new

**创建团队**

允许用户创建新团队，并将用户权限应用到新团队。

👉 [Detailed Doc on setting team budgets](https://docs.litellm.ai/docs/proxy/team_budgets)


参数：
- team_alias: Optional[str] - User defined team alias
- team_id: Optional[str] - The team id of the user. If none passed, we'll generate it.
- members_with_roles: List[{"role": "admin" or "user", "user_id": "<user-id>"}] - A list of users and their roles in the team. Get user_id when making a new user via `/user/new`.
- t...

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `litellm-changed-by` | 请求头 | ❌ | The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of |

#### `POST` /team/update

**更新团队**

请使用 `/team/member_add` 与 `/team/member/delete` 来新增/移除团队成员。

现在可以通过 `/team/update` 更新团队的预算与限流配置。

参数：
- team_id: str - The team id of the user. Required param.
- team_alias: Optional[str] - User defined team alias
- team_member_permissions: Optional[List[str]] - A list of routes that non-admin team members can access. example: ["/key/generate", "/key/update", "/key/delete"]
- metadata: Optional[dict] - Metadata for team, store information for...

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `litellm-changed-by` | 请求头 | ❌ | The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of |

#### `POST` /team/member_add

**添加团队成员**

向团队中添加新成员（通过 `user_email` 或 `user_id`）。

若用户不存在，将在 User 表中新建一条用户记录。

仅 `proxy_admin` 或对应团队的管理员可访问该端点。
```

curl -X POST 'http://0.0.0.0:4000/team/member_add'     -H 'Authorization: Bearer sk-1234'     -H 'Content-Type: application/json'     -d '{"team_id": "45e3e396-ee08-4a61-a88e-16b3ce7e0849", "member": {"role": "user", "user_id": "krrish247652@berri.ai"}}'

```

#### `POST` /team/member_delete

**删除团队成员**

[BETA]

从团队中删除成员（通过 `user_email` 或 `user_id`）。

若用户不存在，将抛出异常。
```
curl -X POST 'http://0.0.0.0:8000/team/member_delete' 
-H 'Authorization: Bearer sk-1234' 
-H 'Content-Type: application/json' 
-d '{
    "team_id": "45e3e396-ee08-4a61-a88e-16b3ce7e0849",
    "user_id": "krrish247652@berri.ai"
}'
```

#### `POST` /team/member_update

**更新团队成员**

[BETA]

更新团队成员预算与角色。

#### `POST` /team/bulk_member_add

**批量添加团队成员**

一次性批量向团队添加多个成员。

该端点复用 `/team/member_add` 的逻辑，并提供更适合批量使用的响应格式。

参数：
- team_id: str - The ID of the team to add members to
- members: List[Member] - List of members to add to the team
- all_users: Optional[bool] - Flag to add all users on Proxy to the team
- max_budget_in_team: Optional[float] - Maximum budget allocated to each user within the team

返回：
- results: List of individual member addition r...

#### `POST` /team/delete

**删除团队**

删除团队及其关联的全部 Key。

参数：
- team_ids: List[str] - Required. List of team IDs to delete. 示例： ["team-1234", "team-5678"]

```
curl --location 'http://0.0.0.0:4000/team/delete'     --header 'Authorization: Bearer sk-1234'     --header 'Content-Type: application/json'     --data-raw '{
    "team_ids": ["8d916b1c-510d-4894-a334-1c16a93344f5"]
}'
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `litellm-changed-by` | 请求头 | ❌ | The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of |

#### `GET` /team/info

**团队信息**

获取团队详情，以及其关联的 Key 列表。

参数：
- team_id: str - Required. The unique identifier of the team to get info on.

```
curl --location 'http://localhost:4000/team/info?team_id=your_team_id_here'     --header 'Authorization: Bearer your_api_key_here'
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `team_id` | 查询参数 | ❌ | Team ID in the request parameters |

#### `POST` /team/block

**封禁团队**

阻断持有该 team_id 的全部 Key 的调用。

参数：
- team_id: str - Required. The unique identifier of the team to block.

示例：
```
curl --location 'http://0.0.0.0:4000/team/block'     --header 'Authorization: Bearer sk-1234'     --header 'Content-Type: application/json'     --data '{
    "team_id": "team-1234"
}'
```

返回：
- The updated team record with blocked=True

#### `POST` /team/unblock

**解封团队**

阻断持有该 team_id 的全部 Key 的调用。

参数：
- team_id: str - Required. The unique identifier of the team to unblock.

示例：
```
curl --location 'http://0.0.0.0:4000/team/unblock'     --header 'Authorization: Bearer sk-1234'     --header 'Content-Type: application/json'     --data '{
    "team_id": "team-1234"
}'
```

#### `GET` /v2/team/list

**列表 Team V2**

分页获取团队列表（支持过滤与排序）。

参数：
    user_id: Optional[str]
        Only return teams which this user belongs to
    organization_id: Optional[str]
        Only return teams which belong to this organization
    team_id: Optional[str]
        Filter teams by exact team_id match
    team_alias: Optional[str]
        Filter teams by partial team_alias match
    page: int
        The page number to return
    page_size: int
        The number of items p...

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `user_id` | 查询参数 | ❌ | Only return teams which this 'user_id' belongs to |
| `organization_id` | 查询参数 | ❌ | Only return teams which this 'organization_id' belongs to |
| `team_id` | 查询参数 | ❌ | Only return teams which this 'team_id' belongs to |
| `team_alias` | 查询参数 | ❌ | Only return teams which this 'team_alias' belongs to. Supports partial matching. |
| `page` | 查询参数 | ❌ | Page number for pagination |
| `page_size` | 查询参数 | ❌ | Number of teams per page |
| `sort_by` | 查询参数 | ❌ | Column to sort by (e.g. 'team_id', 'team_alias', 'created_at') |
| `sort_order` | 查询参数 | ❌ | Sort order ('asc' or 'desc') |
| `status` | 查询参数 | ❌ | Filter by status (e.g. 'deleted') |

#### `GET` /team/list

**团队列表**

```
curl --location --request GET 'http://0.0.0.0:4000/team/list'         --header 'Authorization: Bearer sk-1234'
```

参数：
- user_id: str - Optional. If passed will only return teams that the user_id is a member of.
- organization_id: str - Optional. If passed will only return teams that belong to the organization_id. Pass 'default_organization' to get all teams without organization_id.

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `user_id` | 查询参数 | ❌ | Only return teams which this 'user_id' belongs to |
| `organization_id` | 查询参数 | ❌ |  |

#### `POST` /team/model/add

**为团队添加可用模型**

向团队的可用模型列表中添加模型（仅 Proxy 管理员或团队管理员可操作）。

参数：
- team_id: str - Required. The team to add models to
- models: List[str] - Required. List of models to add to the team

请求示例：
```
curl --location 'http://0.0.0.0:4000/team/model/add'     --header 'Authorization: Bearer sk-1234'     --header 'Content-Type: application/json'     --data '{
    "team_id": "team-1234",
    "models": ["gpt-4", "claude-2"]
}'
```

#### `POST` /team/model/delete

**从团队移除可用模型**

从团队的可用模型列表中移除模型（仅 Proxy 管理员或团队管理员可操作）。

参数：
- team_id: str - Required. The team to remove models from
- models: List[str] - Required. List of models to remove from the team

请求示例：
```
curl --location 'http://0.0.0.0:4000/team/model/delete'     --header 'Authorization: Bearer sk-1234'     --header 'Content-Type: application/json'     --data '{
    "team_id": "team-1234",
    "models": ["gpt-4"]
}'
```

#### `GET` /team/permissions_list

**团队成员权限**

获取团队成员的权限配置。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `team_id` | 查询参数 | ❌ | Team ID in the request parameters |

#### `POST` /team/permissions_update

**更新 Team Member Permissions**

更新团队成员的权限。

#### `GET` /team/daily/activity

**获取 Team Daily Activity**

按指定团队（或全部团队）获取每日活动数据。

参数：
    team_ids (Optional[str]): Comma-separated list of team IDs to filter by. If not provided, returns data for all teams.
    start_date (Optional[str]): Start date for the activity period (YYYY-MM-DD).
    end_date (Optional[str]): End date for the activity period (YYYY-MM-DD).
    model (Optional[str]): Filter by model name.
    api_key (Optional[str]): Filter by API key.
    page (int): Page number for pagination.
    page_size (int):...

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `team_ids` | 查询参数 | ❌ |  |
| `start_date` | 查询参数 | ❌ |  |
| `end_date` | 查询参数 | ❌ |  |
| `model` | 查询参数 | ❌ |  |
| `api_key` | 查询参数 | ❌ |  |
| `page` | 查询参数 | ❌ |  |
| `page_size` | 查询参数 | ❌ |  |
| `exclude_team_ids` | 查询参数 | ❌ |  |

#### `POST` /team/{team_id}/callback

**添加 Team Callbacks**

为团队新增成功/失败回调。

若希望不同团队配置不同的成功/失败回调，请使用该接口。

参数：
- callback_name (Literal["langfuse", "langsmith", "gcs"], required): The name of the callback to add
- callback_type (Literal["success", "failure", "success_and_failure"], required): The type of callback to add. One of:
    - "success": Callback for successful LLM calls
    - "failure": Callback for failed LLM calls
    - "success_and_failure": Callback for b...

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `team_id` | 路径参数 | ✅ |  |
| `litellm-changed-by` | 请求头 | ❌ | The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of |

#### `GET` /team/{team_id}/callback

**获取 Team Callbacks**

获取团队的成功/失败回调与相关变量。

参数：
- team_id (str, required): The unique identifier for the team

请求示例：
```
curl -X GET 'http://localhost:4000/team/dbe2f686-a686-4896-864a-4c3924458709/callback'         -H 'Authorization: Bearer sk-1234'
```

该接口返回 `team_id = dbe2f686-a686-4896-864a-4c3924458709` 的团队回调配置。

返回 `{`
        "status": "success",
        "data": {
            "team_id": team_id,
            "success_call...

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `team_id` | 路径参数 | ✅ |  |

#### `POST` /team/{team_id}/disable_logging

**停用团队日志**

为团队禁用全部日志回调。

参数：
- team_id (str, required): The unique identifier for the team

请求示例：
```
curl -X POST 'http://localhost:4000/team/dbe2f686-a686-4896-864a-4c3924458709/disable_logging'         -H 'Authorization: Bearer sk-1234'
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `team_id` | 路径参数 | ✅ |  |

### 组织管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/organization/new` | 创建组织 |
| `GET` | `/organization/daily/activity` | 获取 Organization Daily Activity |
| `PATCH` | `/organization/update` | 更新组织 |
| `DELETE` | `/organization/delete` | 删除组织 |
| `GET` | `/organization/list` | 列表 Organization |
| `GET` | `/organization/info` | Info Organization |
| `POST` | `/organization/info` | Deprecated Info Organization |
| `POST` | `/organization/member_add` | 添加组织成员 |
| `PATCH` | `/organization/member_update` | 更新组织成员 |
| `DELETE` | `/organization/member_delete` | 删除组织成员 |

#### `POST` /organization/new

**创建组织**

允许组织拥有团队。

设置组织级别的预算和模型访问权限。

仅管理员可创建组织。

# Parameters

- organization_alias: *str* - The name of the organization.
- models: *List* - The models the organization has access to.
- budget_id: *Optional[str]* - The id for a budget (tpm/rpm/max budget) for the organization.
### IF NO BUDGET ID - CREATE ONE WITH THESE PARAMS ###
- max_budget: *Optional[float]* - Max budget for org
- tpm_limit: *Optional[int]* - Max tpm limit for org
- rpm_limit: *O...

#### `GET` /organization/daily/activity

**获取 Organization Daily Activity**

获取指定组织（或当前用户可访问的全部组织）的每日活动数据。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `organization_ids` | 查询参数 | ❌ |  |
| `start_date` | 查询参数 | ❌ |  |
| `end_date` | 查询参数 | ❌ |  |
| `model` | 查询参数 | ❌ |  |
| `api_key` | 查询参数 | ❌ |  |
| `page` | 查询参数 | ❌ |  |
| `page_size` | 查询参数 | ❌ |  |
| `exclude_organization_ids` | 查询参数 | ❌ |  |

#### `PATCH` /organization/update

**更新组织**

更新组织。

#### `DELETE` /organization/delete

**删除组织**

删除组织。

# 参数：

- organization_ids: List[str] - The organization ids to delete.

#### `GET` /organization/list

**列表 Organization**

获取组织列表（支持可选过滤条件）。

参数：
    org_id: Optional[str]
        Filter organizations by exact organization_id match
    org_alias: Optional[str]
        Filter organizations by partial organization_alias match (case-insensitive)

示例：
```
curl --location --request GET 'http://0.0.0.0:4000/organization/list?org_alias=my-org'         --header 'Authorization: Bearer sk-1234'
```

带上 `org_id` 的示例:
```
curl --location --request GET 'http://0.0.0.0:4000/orga...
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `org_id` | 查询参数 | ❌ | Filter organizations by exact organization_id match |
| `org_alias` | 查询参数 | ❌ | Filter organizations by partial organization_alias match. Supports case-insensitive search. |

#### `GET` /organization/info

**组织信息**

获取组织的详细信息。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `organization_id` | 查询参数 | ✅ |  |

#### `POST` /organization/info

**[已废弃] 获取组织信息**

已废弃。请改用 `GET /organization/info`。

#### `POST` /organization/member_add

**添加组织成员**

[BETA]

向组织中添加新成员（通过 `user_email` 或 `user_id`）。

若用户不存在，将在 User 表中新建一条用户记录。

仅 `proxy_admin` 或对应组织的 `org_admin` 可访问该端点。

# 参数：

- organization_id: str (required)
- member: Union[List[Member], Member] (required)
    - role: Literal[LitellmUserRoles] (required)
    - user_id: Optional[str]
    - user_email: Optional[str]

注意：每次调用必须提供 `user_id` 或 `user_email` 其一……

#### `PATCH` /organization/member_update

**更新组织成员**

更新组织成员角色。

#### `DELETE` /organization/member_delete

**删除组织成员**

从组织中删除某个成员。

### 项目管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/project/new` | 新建 Project |
| `POST` | `/project/update` | 更新项目 |
| `DELETE` | `/project/delete` | 删除项目 |
| `GET` | `/project/info` | Project Info |
| `GET` | `/project/list` | 项目列表 |

#### `POST` /project/new

**新建 Project**

创建新的项目。项目在层级结构中位于团队与 Key 之间。

仅管理员或团队管理员可创建项目。

# Parameters

- project_alias: *Optional[str]* - The name of the project.
- description: *Optional[str]* - Description of the project's purpose and use case.
- team_id: *str* - The team id that this project belongs to. Required.
- models: *List* - The models the project has access to.
- budget_id: *Optional[str]* - The id for a budget (tpm/rpm/max budget) for the project....

#### `POST` /project/update

**更新项目**

更新项目。

参数：
- project_id: *str* - The project id to update. Required.
- project_alias: *Optional[str]* - Updated name for the project
- description: *Optional[str]* - Updated description for the project
- team_id: *Optional[str]* - Updated team_id for the project
- metadata: *Optional[dict]* - Updated metadata for project
- models: *Optional[list]* - Updated list of models for the project
- blocked: *Optional[bool]* - Updated blocked status
- max_budget: *Optional[float]* - Upd...

#### `DELETE` /project/delete

**删除项目**

删除项目。

参数：
- project_ids: *List[str]* - List of project ids to delete

示例：
```bash
curl --location --request DELETE 'http://0.0.0.0:4000/project/delete' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{
    "project_ids": ["project-123", "project-456"]
}'
```

#### `GET` /project/info

**项目信息**

获取指定项目的信息。

参数：
- project_id: *str* - The project id to fetch info for

示例：
```bash
curl --location 'http://0.0.0.0:4000/project/info?project_id=project-123' \
--header 'Authorization: Bearer sk-1234'
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `project_id` | 查询参数 | ✅ |  |

#### `GET` /project/list

**项目列表**

列出当前用户可访问的全部项目。

示例：
```bash
curl --location 'http://0.0.0.0:4000/project/list' \
--header 'Authorization: Bearer sk-1234'
```

### 客户管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/customer/block` | 封禁 User |
| `POST` | `/customer/unblock` | 解封 User |
| `POST` | `/customer/new` | 新建 End User |
| `GET` | `/customer/info` | End User Info |
| `POST` | `/customer/update` | 更新 End User |
| `POST` | `/customer/delete` | 删除 End User |
| `GET` | `/customer/list` | 列表 End User |
| `GET` | `/customer/daily/activity` | 获取 Customer Daily Activity |

#### `POST` /customer/block

**封禁 User**

[BETA] Reject calls with this end-user id

参数：
- user_ids (List[str], required): The unique `user_id`s for the users to block

    (any /chat/completion call with this user={end-user-id} param, will be rejected.)

    ```
    curl -X POST "http://0.0.0.0:8000/user/block"
    -H "Authorization: Bearer sk-1234"
    -d '{
    "user_ids": [<user_id>, ...]
    }'
    ```

#### `POST` /customer/unblock

**解封 User**

[BETA] Unblock calls with this user id

示例
```
curl -X POST "http://0.0.0.0:8000/user/unblock"
-H "Authorization: Bearer sk-1234"
-d '{
"user_ids": [<user_id>, ...]
}'
```

#### `POST` /customer/new

**新建 End User**

允许创建新的客户（Customer）。 


参数：
- user_id: str - The unique identifier for the user.
- alias: Optional[str] - A human-friendly alias for the user.
- blocked: bool - Flag to allow or disallow requests for this end-user. Default is False.
- max_budget: Optional[float] - The maximum budget allocated to the user. Either 'max_budget' or 'budget_id' should be provided, not both.
- budget_id: Optional[str] - The identifier for an existing budget allocated to the user. Either 'max_budget' o...

#### `GET` /customer/info

**终端用户信息**

获取终端用户信息。`end_user` 指 Proxy 的外部客户。

参数：
- end_user_id (str, required): The unique identifier for the end-user

请求示例：
```
curl -X GET 'http://localhost:4000/customer/info?end_user_id=test-litellm-user-4'         -H 'Authorization: Bearer sk-1234'
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `end_user_id` | 查询参数 | ✅ | End User ID in the request parameters |

#### `POST` /customer/update

**更新 End User**

Curl 示例 

参数：
- user_id: str
- alias: Optional[str] = None  # human-friendly alias
- blocked: bool = False  # allow/disallow requests for this end-user
- max_budget: Optional[float] = None
- budget_id: Optional[str] = None  # give either a budget_id or max_budget
- allowed_model_region: Optional[AllowedModelRegion] = (
    None  # require all user requests to use models in this specific region
)
- default_model: Optional[str] = (
    None  # if no equivalent model in allowed region ...

#### `POST` /customer/delete

**删除 End User**

批量删除终端用户。

参数：
- user_ids (List[str], required): The unique `user_id`s for the users to delete

请求示例：
```
curl --location 'http://0.0.0.0:4000/customer/delete'         --header 'Authorization: Bearer sk-1234'         --header 'Content-Type: application/json'         --data '{
        "user_ids" :["ishaan-jaff-5"]
}'

全部参数见下方。 
```

#### `GET` /customer/list

**列表 End User**

[Admin-only] List all available customers

请求示例：
```
curl --location --request GET 'http://0.0.0.0:4000/customer/list'         --header 'Authorization: Bearer sk-1234'
```

#### `GET` /customer/daily/activity

**获取 Customer Daily Activity**

获取指定组织（或当前用户可访问的全部组织）的每日活动数据。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `end_user_ids` | 查询参数 | ❌ |  |
| `start_date` | 查询参数 | ❌ |  |
| `end_date` | 查询参数 | ❌ |  |
| `model` | 查询参数 | ❌ |  |
| `api_key` | 查询参数 | ❌ |  |
| `page` | 查询参数 | ❌ |  |
| `page_size` | 查询参数 | ❌ |  |
| `exclude_end_user_ids` | 查询参数 | ❌ |  |

### 访问组管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/v1/access_group` | 访问组列表 |
| `POST` | `/v1/access_group` | 创建访问组 |
| `GET` | `/v1/access_group/{access_group_id}` | 获取 Access Group |
| `PUT` | `/v1/access_group/{access_group_id}` | 更新访问组 |
| `DELETE` | `/v1/access_group/{access_group_id}` | 删除访问组 |
| `GET` | `/v1/unified_access_group` | 访问组列表 |
| `POST` | `/v1/unified_access_group` | 创建访问组 |
| `GET` | `/v1/unified_access_group/{access_group_id}` | 获取 Access Group |
| `PUT` | `/v1/unified_access_group/{access_group_id}` | 更新访问组 |
| `DELETE` | `/v1/unified_access_group/{access_group_id}` | 删除访问组 |

#### `GET` /v1/access_group/{access_group_id}

**获取 Access Group**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `access_group_id` | 路径参数 | ✅ |  |

#### `PUT` /v1/access_group/{access_group_id}

**更新访问组**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `access_group_id` | 路径参数 | ✅ |  |

#### `DELETE` /v1/access_group/{access_group_id}

**删除访问组**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `access_group_id` | 路径参数 | ✅ |  |

#### `GET` /v1/unified_access_group/{access_group_id}

**获取 Access Group**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `access_group_id` | 路径参数 | ✅ |  |

#### `PUT` /v1/unified_access_group/{access_group_id}

**更新访问组**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `access_group_id` | 路径参数 | ✅ |  |

#### `DELETE` /v1/unified_access_group/{access_group_id}

**删除访问组**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `access_group_id` | 路径参数 | ✅ |  |

