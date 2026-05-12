## 预算与计费

### 预算管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/budget/new` | 新建 Budget |
| `POST` | `/budget/update` | 更新预算 |
| `POST` | `/budget/info` | Info Budget |
| `GET` | `/budget/settings` | Budget Settings |
| `GET` | `/budget/list` | 列表 Budget |
| `POST` | `/budget/delete` | 删除预算 |

#### `POST` /budget/new

**新建 Budget**

创建新的预算对象，可应用于团队、组织、终端用户、Key。

参数：
- budget_duration: Optional[str] - Budget reset period ("30d", "1h", etc.)
- budget_id: Optional[str] - The id of the budget. If not provided, a new id will be generated.
- max_budget: Optional[float] - The max budget for the budget.
- soft_budget: Optional[float] - The soft budget for the budget.
- max_parallel_requests: Optional[int] - The max number of parallel requests for the budget.
- tpm_limit: Option...

#### `POST` /budget/update

**更新预算**

更新已有预算对象。

参数：
- budget_duration: Optional[str] - Budget reset period ("30d", "1h", etc.)
- budget_id: Optional[str] - The id of the budget. If not provided, a new id will be generated.
- max_budget: Optional[float] - The max budget for the budget.
- soft_budget: Optional[float] - The soft budget for the budget.
- max_parallel_requests: Optional[int] - The max number of parallel requests for the budget.
- tpm_limit: Optional[int] - The tokens per minute limit for ...

#### `POST` /budget/info

**预算信息**

根据 ID 获取预算信息。

参数：
- budgets: List[str] - The list of budget ids to get information for

#### `GET` /budget/settings

**预算配置**

获取预算条目的可配置参数列表、当前值以及每个字段的说明。

由管理后台 UI 使用。

Query 参数：
- budget_id: str - The budget id to get information for

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `budget_id` | 查询参数 | ✅ |  |

#### `GET` /budget/list

**列表 Budget**

列出 Proxy 数据库中已创建的全部预算。 由管理后台 UI 使用。

#### `POST` /budget/delete

**删除预算**

删除预算。

参数：
- id: str - The budget id to delete

### 预算与支出跟踪

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/user/daily/activity` | 获取 User Daily Activity |
| `GET` | `/user/daily/activity/aggregated` | 获取 User Daily Activity Aggregated |
| `GET` | `/spend/tags` | View Spend Tags |
| `GET` | `/global/spend/report` | 获取 Global Spend Report |
| `GET` | `/global/spend/tags` | Global View Spend Tags |
| `POST` | `/spend/calculate` | Calculate Spend |
| `GET` | `/spend/logs/v2` | Ui View Spend Logs |
| `GET` | `/spend/logs` | 查看支出日志 |
| `POST` | `/global/spend/reset` | Global Spend Reset |
| `POST` | `/usage/ai/chat` | 用法 Ai Chat |
| `POST` | `/add/allowed_ip` | 添加 Allowed Ip |
| `POST` | `/delete/allowed_ip` | 删除 Allowed Ip |

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

#### `GET` /spend/tags

**按标签查看花费**

LiteLLM 企业版 —— 按请求标签查看花费。

请求示例：
```
curl -X GET "http://0.0.0.0:8000/spend/tags" -H "Authorization: Bearer sk-1234"
```

指定起止日期范围的花费统计。
```
curl -X GET "http://0.0.0.0:8000/spend/tags?start_date=2022-01-01&end_date=2022-02-01" -H "Authorization: Bearer sk-1234"
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `start_date` | 查询参数 | ❌ | Time from which to start viewing key spend |
| `end_date` | 查询参数 | ❌ | Time till which to view key spend |

#### `GET` /global/spend/report

**获取 Global Spend Report**

按 `startTime`、`endTime` 返回每个团队的每日花费；对每个团队可按 Key、模型查看具体使用情况。
[
    {
        "group-by-day": "2024-05-10",
        "teams": [
            {
                "team_name": "team-1"
                "spend": 10,
                "keys": [
                    "key": "1213",
                    "usage": {
                        "model-1": {
                                "cost": 12.50,
                                "input_tokens": 1000,
                 ...

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `start_date` | 查询参数 | ❌ | Time from which to start viewing spend |
| `end_date` | 查询参数 | ❌ | Time till which to view spend |
| `group_by` | 查询参数 | ❌ | Group spend by internal team or customer or api_key |
| `api_key` | 查询参数 | ❌ | View spend for a specific api_key. 示例 api_key='sk-1234 |
| `internal_user_id` | 查询参数 | ❌ | View spend for a specific internal_user_id. 示例 internal_user_id='1234 |
| `team_id` | 查询参数 | ❌ | View spend for a specific team_id. 示例 team_id='1234 |
| `customer_id` | 查询参数 | ❌ | View spend for a specific customer_id. 示例 customer_id='1234. Can be used in conjunction with te |

#### `GET` /global/spend/tags

**全局按标签查看花费**

LiteLLM 企业版 —— 按请求标签查看花费（LiteLLM UI 使用）。

请求示例：
```
curl -X GET "http://0.0.0.0:4000/spend/tags" -H "Authorization: Bearer sk-1234"
```

指定起止日期范围的花费统计。
```
curl -X GET "http://0.0.0.0:4000/spend/tags?start_date=2022-01-01&end_date=2022-02-01" -H "Authorization: Bearer sk-1234"
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `start_date` | 查询参数 | ❌ | Time from which to start viewing key spend |
| `end_date` | 查询参数 | ❌ | Time till which to view key spend |
| `tags` | 查询参数 | ❌ | comman separated tags to filter on |

#### `POST` /spend/calculate

**计算花费**

接收 `completion_cost` 的全部参数。

在真正发起调用**之前**预估花费：

注意：若看到 `spend = $0.0`，需要为模型配置 `custom_pricing`：https://docs.litellm.ai/docs/proxy/custom_pricing

```
curl --location 'http://localhost:4000/spend/calculate'
--header 'Authorization: Bearer sk-1234'
--header 'Content-Type: application/json'
--data '{
    "model": "anthropic.claude-v2",
    "messages": [{"role": "user", "content": "Hey, how'''s it going?"}]
}'
```

计算花费……

#### `GET` /spend/logs/v2

**UI 查看花费日志**

分页查看花费日志。
同时可通过 `/spend/logs/v2`（公开 API）与 `/spend/logs/ui`（内部 UI）访问。

返回分页响应，包含 `data`、`total`、`page`、`page_size` 和 `total_pages`。

示例：
```
curl -X GET "http://0.0.0.0:8000/spend/logs/v2?start_date=2025-11-25%2000:00:00&end_date=2025-11-26%2023:59:59&page=1&page_size=50" -H "Authorization: Bearer sk-1234"
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `api_key` | 查询参数 | ❌ | Get spend logs based on api key |
| `user_id` | 查询参数 | ❌ | Get spend logs based on user_id |
| `request_id` | 查询参数 | ❌ | request_id to get spend logs for specific request_id |
| `team_id` | 查询参数 | ❌ | Filter spend logs by team_id |
| `min_spend` | 查询参数 | ❌ | Filter logs with spend greater than or equal to this value |
| `max_spend` | 查询参数 | ❌ | Filter logs with spend less than or equal to this value |
| `start_date` | 查询参数 | ❌ | Time from which to start viewing key spend |
| `end_date` | 查询参数 | ❌ | Time till which to view key spend |
| `page` | 查询参数 | ❌ | Page number for pagination |
| `page_size` | 查询参数 | ❌ | Number of items per page |

#### `GET` /spend/logs

**查看支出日志**

[已废弃] 该端点不支持分页，可能引发性能问题。
请改用 `/spend/logs/v2` 以分页访问花费日志。

查看全部花费日志；若提供 `request_id`，则仅返回该请求的日志。

当同时提供 `start_date` 和 `end_date` 时：
- summarize=true (default): Returns aggregated spend data grouped by date (maintains backward compatibility)
- summarize=false: Returns filtered individual log entries within the date range

请求示例 for...

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `api_key` | 查询参数 | ❌ | Get spend logs based on api key |
| `user_id` | 查询参数 | ❌ | Get spend logs based on user_id |
| `request_id` | 查询参数 | ❌ | request_id to get spend logs for specific request_id. If none passed then pass spend logs for all re |
| `start_date` | 查询参数 | ❌ | Time from which to start viewing key spend |
| `end_date` | 查询参数 | ❌ | Time till which to view key spend |
| `summarize` | 查询参数 | ❌ | When start_date and end_date are provided, summarize=true returns aggregated data by date (legacy be |

#### `POST` /global/spend/reset

**全局花费重置**

仅限管理员 / Master Key 的端点

全局重置全部 API Key 与团队的花费（保留 `LiteLLM_SpendLogs`）。

1. LiteLLM_SpendLogs will maintain the logs on spend, no data gets deleted from there
2. LiteLLM_VerificationTokens spend will be set = 0
3. LiteLLM_TeamTable spend will be set = 0

#### `POST` /usage/ai/chat

**用法 Ai Chat**

关于用量数据的 AI 对话，通过 SSE 流式返回 AI 响应。
该 AI Agent 可调用用于查询每日聚合活动数据的工具。

### 成本跟踪

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/config/cost_discount_config` | 获取 Cost Discount Config |
| `PATCH` | `/config/cost_discount_config` | 更新 Cost Discount Config |
| `GET` | `/config/cost_margin_config` | 获取 Cost Margin Config |
| `PATCH` | `/config/cost_margin_config` | 更新 Cost Margin Config |
| `POST` | `/cost/estimate` | Estimate Cost |

#### `GET` /config/cost_discount_config

**获取 Cost Discount Config**

获取当前的成本折扣配置。

返回 `litellm_settings` 中的 `cost_discount_config`。

#### `PATCH` /config/cost_discount_config

**更新 Cost Discount Config**

更新成本折扣配置。

更新 `litellm_settings` 中的 `cost_discount_config`。
折扣值应介于 0 和 1 之间（例如 `0.05` 表示 5% 折扣）。

示例：
```json
{
    "vertex_ai": 0.05,
    "gemini": 0.05,
    "openai": 0.01
}
```

#### `GET` /config/cost_margin_config

**获取 Cost Margin Config**

获取当前的成本溢价配置。

返回 `litellm_settings` 中的 `cost_margin_config`。

#### `PATCH` /config/cost_margin_config

**更新 Cost Margin Config**

更新成本溢价配置。

更新 `litellm_settings` 中的 `cost_margin_config`。
Margin 可以为：
- Percentage: {"openai": 0.10} = 10% margin
- Fixed amount: {"openai": {"fixed_amount": 0.001}} = $0.001 per request
- Combined: {"vertex_ai": {"percentage": 0.08, "fixed_amount": 0.0005}}
- Global: {"global": 0.05} = 5% global margin on all providers

示例：
```
{
    "global": 0.05,
    "openai": 0.10,
    "anthropic": {"fixed_amount": 0.001},
    "vertex_ai": {"percentage": 0.08, "fixe...
```

#### `POST` /cost/estimate

**估算成本**

按给定的模型与 Token 数量估算成本。

该端点使用与实际请求相同的成本计算逻辑（包含已配置的利润率与折扣）。

参数：
- model: Model name (e.g., "gpt-4", "claude-3-opus")
- input_tokens: Expected input tokens per request
- output_tokens: Expected output tokens per request
- num_requests_per_day: Number of requests per day (optional)
- num_requests_per_month: Number of requests per month (optional)

返回成本拆解信息（包括……）。

### 标签管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/tag/new` | 新建 Tag |
| `POST` | `/tag/update` | 更新标签 |
| `POST` | `/tag/info` | Info Tag |
| `GET` | `/tag/list` | 标签列表 |
| `POST` | `/tag/delete` | 删除标签 |
| `GET` | `/tag/daily/activity` | 获取 Tag Daily Activity |
| `GET` | `/tag/distinct` | 获取 Distinct User Agent Tags |
| `GET` | `/tag/dau` | 获取 Daily Active Users |
| `GET` | `/tag/wau` | 获取 Weekly Active Users |
| `GET` | `/tag/mau` | 获取 Monthly Active Users |
| `GET` | `/tag/summary` | 获取 Tag Summary |
| `GET` | `/tag/user-agent/per-user-analytics` | 获取 Per User Analytics |

#### `POST` /tag/new

**新建 Tag**

创建新的标签。

参数：
- name: str - The name of the tag
- description: Optional[str] - Description of what this tag represents
- models: List[str] - List of either 'model_id' or 'model_name' allowed for this tag
- budget_id: Optional[str] - The id for a budget (tpm/rpm/max budget) for the tag

### IF NO BUDGET ID - CREATE ONE WITH THESE PARAMS ###
- max_budget: Optional[float] - Max budget for tag
- tpm_limit: Optional[int] - Max tpm limit for tag
- rpm_limit: Optional[int] - Max rpm li...

#### `POST` /tag/update

**更新标签**

更新已有标签。

参数：
- name: str - The name of the tag to update
- description: Optional[str] - Updated description
- models: List[str] - Updated list of allowed LLM models
- budget_id: Optional[str] - The id for a budget to associate with the tag

### BUDGET UPDATE PARAMS ###
- max_budget: Optional[float] - Max budget for tag
- tpm_limit: Optional[int] - Max tpm limit for tag
- rpm_limit: Optional[int] - Max rpm limit for tag
- max_parallel_requests: Optional[int] - Max parallel...

#### `POST` /tag/info

**标签信息**

获取指定标签的信息。

参数：
- names: List[str] - List of tag names to get information for

#### `GET` /tag/list

**标签列表**

列出全部可用标签及其预算信息。

#### `POST` /tag/delete

**删除标签**

删除标签。

参数：
- name: str - The name of the tag to delete

#### `GET` /tag/daily/activity

**获取 Tag Daily Activity**

按指定标签（或全部标签）获取每日活动数据。

参数：
    tags (Optional[str]): Comma-separated list of tags to filter by. If not provided, returns data for all tags.
    start_date (Optional[str]): Start date for the activity period (YYYY-MM-DD).
    end_date (Optional[str]): End date for the activity period (YYYY-MM-DD).
    model (Optional[str]): Filter by model name.
    api_key (Optional[str]): Filter by API key.
    page (int): Page number for pagination.
    page_size (int): Number of ...

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `tags` | 查询参数 | ❌ |  |
| `start_date` | 查询参数 | ❌ |  |
| `end_date` | 查询参数 | ❌ |  |
| `model` | 查询参数 | ❌ |  |
| `api_key` | 查询参数 | ❌ |  |
| `page` | 查询参数 | ❌ |  |
| `page_size` | 查询参数 | ❌ |  |

#### `GET` /tag/distinct

**获取 Distinct User Agent Tags**

获取去重后的 user agent 标签，最多返回 `{MAX_TAGS}` 个。

该端点返回数据库中全部去重后的 user agent 标签。
按使用频次排序。

返回：
    DistinctTags响应： List of distinct user agent tags

#### `GET` /tag/dau

**获取 Daily Active Users**

按标签统计最近 `{MAX_DAYS}` 天（截至 UTC 今天 +1 天）的日活用户（DAU）。

该端点高效地统计最近 {MAX_DAYS} 天内每个标签对应的去重用户数。
使用单条优化后的 SQL 查询完成，非常适合仪表盘的时间序列展示。

参数：
    tag_filter: Optional filter to specific tag (legacy)
    tag_filters: Optional filter to multiple specific tags (takes precedence over tag_filter)

返回：
    ActiveUsersAnalytics响应： DAU data by tag for ea...

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `tag_filter` | 查询参数 | ❌ | Filter by specific tag (optional) |
| `tag_filters` | 查询参数 | ❌ | Filter by multiple specific tags (optional, takes precedence over tag_filter) |

#### `GET` /tag/wau

**获取 Weekly Active Users**

按标签统计最近 `{MAX_WEEKS}` 周（截至 UTC 今天 +1 天）的周活用户（WAU）。

按周维度展开：
- Week 1 (Jan 1): Earliest week (7 weeks ago)
- Week 2 (Jan 8): Next week (6 weeks ago)
- Week 3 (Jan 15): Next week (5 weeks ago)
- ... and so on for {MAX_WEEKS} weeks total
- Week 7: Most recent week ending on UTC today + 1 day

参数：
    tag_filter: Optional filter to specific tag (legacy)
    tag_filters: Optional filter to multiple specific tags (takes precedence ...

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `tag_filter` | 查询参数 | ❌ | Filter by specific tag (optional) |
| `tag_filters` | 查询参数 | ❌ | Filter by multiple specific tags (optional, takes precedence over tag_filter) |

#### `GET` /tag/mau

**获取 Monthly Active Users**

按标签统计最近 `{MAX_MONTHS}` 月（截至 UTC 今天 +1 天）的月活用户（MAU）。

按月份维度展开：
- Month 1 (Nov): Earliest month (7 months ago, 30-day period)
- Month 2 (Dec): Next month (6 months ago)
- Month 3 (Jan): Next month (5 months ago)
- ... and so on for {MAX_MONTHS} months total
- Month 7: Most recent month ending on UTC today + 1 day

参数：
    tag_filter: Optional filter to specific tag (legacy)
    tag_filters: Optional filter to multiple specif...

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `tag_filter` | 查询参数 | ❌ | Filter by specific tag (optional) |
| `tag_filters` | 查询参数 | ❌ | Filter by multiple specific tags (optional, takes precedence over tag_filter) |

#### `GET` /tag/summary

**获取 Tag Summary**

返回标签维度的汇总分析数据，包括去重用户数、请求数、Token 数与花费。

参数：
    start_date: Start date for the analytics period (YYYY-MM-DD)
    end_date: End date for the analytics period (YYYY-MM-DD)
    tag_filter: Optional filter to specific tag (legacy)
    tag_filters: Optional filter to multiple specific tags (takes precedence over tag_filter)

返回：
    TagSummary响应： Summary analytics data by tag

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `start_date` | 查询参数 | ✅ | Start date in YYYY-MM-DD format |
| `end_date` | 查询参数 | ✅ | End date in YYYY-MM-DD format |
| `tag_filter` | 查询参数 | ❌ | Filter by specific tag (optional) |
| `tag_filters` | 查询参数 | ❌ | Filter by multiple specific tags (optional, takes precedence over tag_filter) |

#### `GET` /tag/user-agent/per-user-analytics

**获取 Per User Analytics**

返回按用户维度的分析数据，包含每个用户的成功请求数、Token 数与花费。

该端点按单个用户维度返回使用指标（基于用户所属）。
最近 30 天（截至 UTC 今天 +1 天）的标签活动情况。

参数：
    tag_filter: Optional filter to specific tag (legacy)
    tag_filters: Optional filter to multiple specific tags (takes precedence over tag_filter)
    page: Page number for pagination
    page_size: Number of items per page

返回：
    PerUser...

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `tag_filter` | 查询参数 | ❌ | Filter by specific tag (optional) |
| `tag_filters` | 查询参数 | ❌ | Filter by multiple specific tags (optional, takes precedence over tag_filter) |
| `page` | 查询参数 | ❌ | Page number for pagination |
| `page_size` | 查询参数 | ❌ | Items per page |

