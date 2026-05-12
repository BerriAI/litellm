## 评估与分析

### OpenAI 评估 API

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/evals` | 创建 Eval |
| `GET` | `/v1/evals` | 列表 Evals |
| `GET` | `/v1/evals/{eval_id}` | 获取 Eval |
| `POST` | `/v1/evals/{eval_id}` | 更新 Eval |
| `DELETE` | `/v1/evals/{eval_id}` | 删除 Eval |
| `POST` | `/v1/evals/{eval_id}/cancel` | 取消 Eval |

#### `POST` /v1/evals

**创建 Eval**

创建一个新的 evaluation。

基于模型的路由（用于多账号场景）：
- 通过 Header 传入模型：`x-litellm-model: gpt-4-account-1`
- 通过 Query 传入模型：`?model=gpt-4-account-1`
- 通过 Body 传入模型：`{"model": "gpt-4-account-1"}`

请求示例：
```bash
curl -X POST "http://localhost:4000/v1/evals"       -H "Authorization: Bearer your-key"       -H "Content-Type: application/json"       -d '{
    "name": "Test Eval",
    "data_source_config": {"type": "file", "file_id": "file-abc123"},
 ...
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `custom_llm_provider` | 查询参数 | ❌ |  |

#### `GET` /v1/evals

**列表 Evals**

分页列出所有 evaluation。

基于模型的路由（用于多账号场景）：
- 通过 Header 传入模型：`x-litellm-model: gpt-4-account-1`
- 通过 Query 传入模型：`?model=gpt-4-account-1`
- 通过 Body 传入模型：`{"model": "gpt-4-account-1"}`

请求示例：
```bash
curl "http://localhost:4000/v1/evals?limit=10"       -H "Authorization: Bearer your-key"
```

返回 `ListEvalsResponse`，包含 evaluation 列表。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `limit` | 查询参数 | ❌ |  |
| `after` | 查询参数 | ❌ |  |
| `before` | 查询参数 | ❌ |  |
| `order` | 查询参数 | ❌ |  |
| `order_by` | 查询参数 | ❌ |  |
| `custom_llm_provider` | 查询参数 | ❌ |  |

#### `GET` /v1/evals/{eval_id}

**获取 Eval**

根据 ID 获取指定 evaluation。

基于模型的路由（用于多账号场景）：
- 通过 Header 传入模型：`x-litellm-model: gpt-4-account-1`
- 通过 Query 传入模型：`?model=gpt-4-account-1`
- 通过 Body 传入模型：`{"model": "gpt-4-account-1"}`

请求示例：
```bash
curl "http://localhost:4000/v1/evals/eval_123"       -H "Authorization: Bearer your-key"
```

返回 `Eval` 对象。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `eval_id` | 路径参数 | ✅ |  |
| `custom_llm_provider` | 查询参数 | ❌ |  |

#### `POST` /v1/evals/{eval_id}

**更新 Eval**

更新指定 evaluation。

基于模型的路由（用于多账号场景）：
- 通过 Header 传入模型：`x-litellm-model: gpt-4-account-1`
- 通过 Query 传入模型：`?model=gpt-4-account-1`
- 通过 Body 传入模型：`{"model": "gpt-4-account-1"}`

请求示例：
```bash
curl -X POST "http://localhost:4000/v1/evals/eval_123"       -H "Authorization: Bearer your-key"       -H "Content-Type: application/json"       -d '{"name": "Updated Name"}'
```

返回更新后的 `Eval` 对象。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `eval_id` | 路径参数 | ✅ |  |
| `custom_llm_provider` | 查询参数 | ❌ |  |

#### `DELETE` /v1/evals/{eval_id}

**删除 Eval**

删除指定 evaluation。

基于模型的路由（用于多账号场景）：
- 通过 Header 传入模型：`x-litellm-model: gpt-4-account-1`
- 通过 Query 传入模型：`?model=gpt-4-account-1`
- 通过 Body 传入模型：`{"model": "gpt-4-account-1"}`

请求示例：
```bash
curl -X DELETE "http://localhost:4000/v1/evals/eval_123"       -H "Authorization: Bearer your-key"
```

返回 `DeleteEvalResponse`，包含删除确认信息。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `eval_id` | 路径参数 | ✅ |  |
| `custom_llm_provider` | 查询参数 | ❌ |  |

#### `POST` /v1/evals/{eval_id}/cancel

**取消 Eval**

取消正在运行的 evaluation。

基于模型的路由（用于多账号场景）：
- 通过 Header 传入模型：`x-litellm-model: gpt-4-account-1`
- 通过 Query 传入模型：`?model=gpt-4-account-1`
- 通过 Body 传入模型：`{"model": "gpt-4-account-1"}`

请求示例：
```bash
curl -X POST "http://localhost:4000/v1/evals/eval_123/cancel"       -H "Authorization: Bearer your-key"
```

返回 `CancelEvalResponse`，包含取消确认信息。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `eval_id` | 路径参数 | ✅ |  |
| `custom_llm_provider` | 查询参数 | ❌ |  |

### OpenAI 评估 API - 运行

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/evals/{eval_id}/runs` | 创建 Run |
| `GET` | `/v1/evals/{eval_id}/runs` | 列表 Runs |
| `GET` | `/v1/evals/{eval_id}/runs/{run_id}` | 获取运行状态 |
| `POST` | `/v1/evals/{eval_id}/runs/{run_id}` | 取消 Run |
| `DELETE` | `/v1/evals/{eval_id}/runs/{run_id}` | 删除 Run |

#### `POST` /v1/evals/{eval_id}/runs

**创建 Run**

为指定 evaluation 创建一个新的 run。

基于模型的路由（用于多账号场景）：
- 通过 Header 传入模型：`x-litellm-model: gpt-4-account-1`
- 通过 Query 传入模型：`?model=gpt-4-account-1`
- 通过 Body 传入模型：`{"model": "gpt-4-account-1"}`
- 通过 `completion.model` 传入：`{"completion": {"model": "gpt-4-account-1"}}`

请求示例：
```bash
curl -X POST "http://localhost:4000/v1/evals/eval_123/runs"       -H "Authorization: Bearer your-key"       -H "Content-Type: application/json"  ...
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `eval_id` | 路径参数 | ✅ |  |
| `custom_llm_provider` | 查询参数 | ❌ |  |

#### `GET` /v1/evals/{eval_id}/runs

**列表 Runs**

分页列出指定 evaluation 下的所有 run。

基于模型的路由（用于多账号场景）：
- 通过 Header 传入模型：`x-litellm-model: gpt-4-account-1`
- 通过 Query 传入模型：`?model=gpt-4-account-1`

请求示例：
```bash
curl "http://localhost:4000/v1/evals/eval_123/runs?limit=10"       -H "Authorization: Bearer your-key"
```

返回 `ListRunsResponse`，包含 run 列表。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `eval_id` | 路径参数 | ✅ |  |
| `limit` | 查询参数 | ❌ |  |
| `after` | 查询参数 | ❌ |  |
| `before` | 查询参数 | ❌ |  |
| `order` | 查询参数 | ❌ |  |
| `custom_llm_provider` | 查询参数 | ❌ |  |

#### `GET` /v1/evals/{eval_id}/runs/{run_id}

**获取运行状态**

根据 ID 获取指定 run。

基于模型的路由（用于多账号场景）：
- 通过 Header 传入模型：`x-litellm-model: gpt-4-account-1`
- 通过 Query 传入模型：`?model=gpt-4-account-1`

请求示例：
```bash
curl "http://localhost:4000/v1/evals/eval_123/runs/run_456"       -H "Authorization: Bearer your-key"
```

返回包含完整详情的 `Run` 对象。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `eval_id` | 路径参数 | ✅ |  |
| `run_id` | 路径参数 | ✅ |  |
| `custom_llm_provider` | 查询参数 | ❌ |  |

#### `POST` /v1/evals/{eval_id}/runs/{run_id}

**取消 Run**

取消正在运行的 run。

基于模型的路由（用于多账号场景）：
- 通过 Header 传入模型：`x-litellm-model: gpt-4-account-1`
- 通过 Query 传入模型：`?model=gpt-4-account-1`

请求示例：
```bash
curl -X POST "http://localhost:4000/v1/evals/eval_123/runs/run_456/cancel"       -H "Authorization: Bearer your-key"
```

返回 `CancelRunResponse`，包含取消确认信息。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `eval_id` | 路径参数 | ✅ |  |
| `run_id` | 路径参数 | ✅ |  |
| `custom_llm_provider` | 查询参数 | ❌ |  |

#### `DELETE` /v1/evals/{eval_id}/runs/{run_id}

**删除 Run**

删除指定 run。

基于模型的路由（用于多账号场景）：
- 通过 Header 传入模型：`x-litellm-model: gpt-4-account-1`
- 通过 Query 传入模型：`?model=gpt-4-account-1`

请求示例：
```bash
curl -X DELETE "http://localhost:4000/v1/evals/eval_123/runs/run_456"       -H "Authorization: Bearer your-key"
```

返回 `RunDeleteResponse`，包含删除确认信息。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `eval_id` | 路径参数 | ✅ |  |
| `run_id` | 路径参数 | ✅ |  |
| `custom_llm_provider` | 查询参数 | ❌ |  |

### 用户代理分析

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/tag/distinct` | 获取不重复的用户代理 Tag |
| `GET` | `/tag/dau` | 获取日活用户（DAU） |
| `GET` | `/tag/wau` | 获取周活用户（WAU） |
| `GET` | `/tag/mau` | 获取月活用户（MAU） |
| `GET` | `/tag/summary` | 获取 Tag 汇总统计 |
| `GET` | `/tag/user-agent/per-user-analytics` | 获取单用户级分析 |

#### `GET` /tag/distinct

**获取不重复的用户代理 Tag**

获取所有不重复的用户代理（user agent）Tag，最多返回 `{MAX_TAGS}` 个。

该端点返回数据库中所有唯一的用户代理 Tag，按使用频次排序。

返回 `DistinctTagsResponse`，包含不重复的用户代理 Tag 列表。

#### `GET` /tag/dau

**获取日活用户（DAU）**

按 Tag 获取最近 `{MAX_DAYS}` 天的日活用户（DAU），截止时间为 UTC 当天 + 1 天。

该端点通过一条经过优化的 SQL 高效计算每个 Tag 的每日单人数，适用于 dashboard 的时间序列可视化。

参数：
- `tag_filter`：指定的单个 Tag 过滤（旧版）
- `tag_filters`：指定的多个 Tag 过滤，优先级高于 `tag_filter`

返回 `ActiveUsersAnalyticsResponse`，包含按 Tag 组织的每日 DAU 数据。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `tag_filter` | 查询参数 | ❌ | 按单个 Tag 过滤（可选） |
| `tag_filters` | 查询参数 | ❌ | 按多个 Tag 过滤（可选，优先级高于 `tag_filter`） |

#### `GET` /tag/wau

**获取周活用户（WAU）**

按 Tag 获取最近 `{MAX_WEEKS}` 周的周活用户（WAU），截止时间为 UTC 当天 + 1 天。

按周返回明细：
- Week 1（例如 Jan 1）：最早的一周（7 周前）
- Week 2（例如 Jan 8）：下一周（6 周前）
- Week 3（例如 Jan 15）：再下一周（5 周前）
- 以此类推，共 `{MAX_WEEKS}` 周
- Week 7：最近一周，截止于 UTC 当天 + 1 天

参数：
- `tag_filter`：指定的单个 Tag 过滤（旧版）
- `tag_filters`：指定的多个 Tag 过滤，优先级高于 `tag_filter`

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `tag_filter` | 查询参数 | ❌ | 按单个 Tag 过滤（可选） |
| `tag_filters` | 查询参数 | ❌ | 按多个 Tag 过滤（可选，优先级高于 `tag_filter`） |

#### `GET` /tag/mau

**获取月活用户（MAU）**

按 Tag 获取最近 `{MAX_MONTHS}` 个月的月活用户（MAU），截止时间为 UTC 当天 + 1 天。

按月返回明细：
- Month 1（例如 Nov）：最早的一个月（7 个月前，按 30 天为一个周期）
- Month 2（例如 Dec）：下一个月（6 个月前）
- Month 3（例如 Jan）：再下一个月（5 个月前）
- 以此类推，共 `{MAX_MONTHS}` 个月
- Month 7：最近一个月，截止于 UTC 当天 + 1 天

参数：
- `tag_filter`：指定的单个 Tag 过滤（旧版）
- `tag_filters`：指定的多个 Tag 过滤，优先级高于 `tag_filter`

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `tag_filter` | 查询参数 | ❌ | 按单个 Tag 过滤（可选） |
| `tag_filters` | 查询参数 | ❌ | 按多个 Tag 过滤（可选，优先级高于 `tag_filter`） |

#### `GET` /tag/summary

**获取 Tag 汇总统计**

按 Tag 获取汇总分析数据，包括唯一用户数、请求数、token 数及消费。

参数：
- `start_date`：分析窗口的开始日期（`YYYY-MM-DD`）
- `end_date`：分析窗口的结束日期（`YYYY-MM-DD`）
- `tag_filter`：指定的单个 Tag 过滤（旧版）
- `tag_filters`：指定的多个 Tag 过滤，优先级高于 `tag_filter`

返回 `TagSummaryResponse`，包含按 Tag 汇总的分析数据。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `start_date` | 查询参数 | ✅ | 开始日期（`YYYY-MM-DD`） |
| `end_date` | 查询参数 | ✅ | 结束日期（`YYYY-MM-DD`） |
| `tag_filter` | 查询参数 | ❌ | 按单个 Tag 过滤（可选） |
| `tag_filters` | 查询参数 | ❌ | 按多个 Tag 过滤（可选，优先级高于 `tag_filter`） |

#### `GET` /tag/user-agent/per-user-analytics

**获取单用户级分析**

按单个用户级别返回使用情况，包括成功请求数、token 数以及消费。

该端点统计最近 30 天（截止于 UTC 当天 + 1 天）每个用户的 Tag 活动使用指标。

参数：
- `tag_filter`：指定的单个 Tag 过滤（旧版）
- `tag_filters`：指定的多个 Tag 过滤，优先级高于 `tag_filter`
- `page`：分页页码
- `page_size`：每页条数

返回 `PerUserAnalyticsResponse`（结果被截断，见官方定义）。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `tag_filter` | 查询参数 | ❌ | 按单个 Tag 过滤（可选） |
| `tag_filters` | 查询参数 | ❌ | 按多个 Tag 过滤（可选，优先级高于 `tag_filter`） |
| `page` | 查询参数 | ❌ | 分页页码 |
| `page_size` | 查询参数 | ❌ | 每页条数 |

### CloudZero

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/cloudzero/settings` | 获取 CloudZero 配置 |
| `PUT` | `/cloudzero/settings` | 更新 CloudZero 配置 |
| `POST` | `/cloudzero/init` | 初始化 CloudZero 配置 |
| `POST` | `/cloudzero/dry-run` | CloudZero 演练（Dry Run）导出 |
| `POST` | `/cloudzero/export` | CloudZero 导出 |
| `DELETE` | `/cloudzero/delete` | 删除 CloudZero 配置 |

#### `GET` /cloudzero/settings

**获取 CloudZero 配置**

查看当前 CloudZero 配置。

返回当前 CloudZero 配置，API Key 会被掩码。仅显示前 4 位和后 4 位，其余覆盖。
若配置尚未设置，返回空值/null（与其他设置类端点行为一致）。

仅管理员用户可以查看 CloudZero 配置。

#### `PUT` /cloudzero/settings

**更新 CloudZero 配置**

更新已有的 CloudZero 配置。

支持增量更新，无需一次性传入所有字段；仅传入的字段会被更新，其余保持不变。

参数：
- `api_key`（可选）：CloudZero 认证用的 API Key
- `connection_id`（可选）：CloudZero 数据提交用的 connection ID
- `timezone`（可选）：日期处理使用的时区

仅管理员用户可以更新 CloudZero 配置。

#### `POST` /cloudzero/init

**初始化 CloudZero 配置**

初始化 CloudZero 配置并将其写入数据库。

该端点将 CloudZero 的 API Key、connection ID、时区写入 proxy 数据库，供 CloudZero logger 使用。

参数：
- `api_key`：CloudZero 认证用的 API Key
- `connection_id`：CloudZero 数据提交用的 connection ID
- `timezone`：日期处理使用的时区（默认：UTC）

仅管理员用户可以配置 CloudZero。

#### `POST` /cloudzero/dry-run

**CloudZero 演练（Dry Run）导出**

使用 CloudZero logger 进行一次演练（dry run）导出。

该端点仅返回将会导出的数据详情，不会真正向 CloudZero 发送。

参数：
- `limit`（可选）：处理的记录条数上限（默认 10000）

返回：
- `usage_data`：原始使用数据样本（前 50 条）
- `cbf_data`：已按 CloudZero CBF 格式格式化后的待导出数据
- `summary`：汇总统计，包含总消费、token 数、记录条数等

仅管理员用户可以执行。

#### `POST` /cloudzero/export

**CloudZero 导出**

使用 CloudZero logger 执行实际的数据导出。

该端点会将使用数据导出到 CloudZero AnyCost API。

参数：
- `limit`（可选）：导出的记录条数上限
- `operation`：CloudZero 操作类型（`replace_hourly` 或 `sum`，默认 `replace_hourly`）

仅管理员用户可以执行 CloudZero 导出。

#### `DELETE` /cloudzero/delete

**删除 CloudZero 配置**

从数据库中删除 CloudZero 配置。

该端点会从 proxy 数据库中移除 CloudZero 配置（API Key、connection ID、时区），
仅删除 CloudZero 相关记录，其他配置值不受影响。

仅管理员用户可以删除 CloudZero 配置。

### Vantage

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/vantage/settings` | 获取 Vantage 配置 |
| `PUT` | `/vantage/settings` | 更新 Vantage 配置 |
| `POST` | `/vantage/init` | 初始化 Vantage 配置 |
| `POST` | `/vantage/dry-run` | Vantage 演练（Dry Run）导出 |
| `POST` | `/vantage/export` | Vantage 导出 |
| `DELETE` | `/vantage/delete` | 删除 Vantage 配置 |

#### `GET` /vantage/settings

**获取 Vantage 配置**

查看当前 Vantage 配置。

返回当前 Vantage 配置，API Key 会被掩码。仅管理员用户可以查看。

#### `PUT` /vantage/settings

**更新 Vantage 配置**

更新已有的 Vantage 配置。

支持增量更新，无需传入所有字段。仅管理员用户可以更新。

#### `POST` /vantage/init

**初始化 Vantage 配置**

初始化 Vantage 配置并写入数据库。

参数：
- `api_key`：Vantage 认证用的 API Key
- `integration_token`：Vantage 成本导入端点所需的集成 token
- `base_url`：Vantage API 基础 URL（默认：`https://api.vantage.sh`）

仅管理员用户可以配置。

#### `POST` /vantage/dry-run

**Vantage 演练（Dry Run）导出**

使用 Vantage logger 执行一次演练导出。

仅返回将会导出的数据，不会真正向 Vantage 发送。

参数：
- `limit`：预览的记录条数上限（默认 500）

仅管理员用户可以执行。

#### `POST` /vantage/export

**Vantage 导出**

使用 Vantage logger 执行实际导出。

会以 FOCUS CSV 格式将使用数据导出到 Vantage API。

参数：
- `limit`（可选）：导出记录条数上限
- `start_time_utc`（可选）：导出数据的开始时间（UTC）
- `end_time_utc`（可选）：导出数据的结束时间（UTC）

仅管理员用户可以执行。

#### `DELETE` /vantage/delete

**删除 Vantage 配置**

从数据库中删除 Vantage 配置。仅管理员用户可以删除。

