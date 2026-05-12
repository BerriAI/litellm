## 安全与合规

### 安全护栏

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/guardrails/list` | 安全护栏列表 |
| `GET` | `/v2/guardrails/list` | 列表 Guardrails V2 |
| `POST` | `/guardrails` | 创建安全护栏 |
| `PUT` | `/guardrails/{guardrail_id}` | 更新安全护栏 |
| `DELETE` | `/guardrails/{guardrail_id}` | 删除安全护栏 |
| `PATCH` | `/guardrails/{guardrail_id}` | Patch Guardrail |
| `GET` | `/guardrails/{guardrail_id}` | 获取安全护栏信息 |
| `POST` | `/guardrails/register` | Register Guardrail |
| `GET` | `/guardrails/submissions` | 列表 Guardrail Submissions |
| `GET` | `/guardrails/submissions/{guardrail_id}` | 获取 Guardrail Submission |
| `POST` | `/guardrails/submissions/{guardrail_id}/approve` | Approve Guardrail Submission |
| `POST` | `/guardrails/submissions/{guardrail_id}/reject` | Reject Guardrail Submission |
| `GET` | `/guardrails/{guardrail_id}/info` | 获取安全护栏信息 |
| `GET` | `/guardrails/ui/add_guardrail_settings` | 获取 Guardrail Ui Settings |
| `GET` | `/guardrails/ui/category_yaml/{category_name}` | 获取 Category Yaml |
| `GET` | `/guardrails/ui/major_airlines` | 获取 Major Airlines |
| `POST` | `/guardrails/validate_blocked_words_file` | Validate Blocked Words File |
| `GET` | `/guardrails/ui/provider_specific_params` | 获取 Provider Specific Params |
| `POST` | `/guardrails/test_custom_code` | Test Custom Code Guardrail |
| `GET` | `/guardrails/usage/overview` | Guardrails 用法 Overview |
| `GET` | `/guardrails/usage/detail/{guardrail_id}` | Guardrails 用法 Detail |
| `GET` | `/guardrails/usage/logs` | Guardrails 用法 Logs |

#### `GET` /guardrails/list

**安全护栏列表**

列出 Proxy 服务端当前可用的 Guardrail。

👉 [Guardrail docs](https://docs.litellm.ai/docs/proxy/guardrails/quick_start)

请求示例：
```bash
curl -X GET "http://localhost:4000/guardrails/list" -H "Authorization: Bearer <your_api_key>"
```

响应示例：
```
{
    "guardrails": [
        {
        "guardrail_name": "bedrock-pre-guard",
        "guardrail_info": {
            "params": [
            {
                "name": "toxicity_score",
                "type": ...
```

#### `GET` /v2/guardrails/list

**列表 Guardrails V2**

通过 GuardrailRegistry 列出数据库中可用的 Guardrail。

👉 [Guardrail docs](https://docs.litellm.ai/docs/proxy/guardrails/quick_start)

请求示例：
```bash
curl -X GET "http://localhost:4000/v2/guardrails/list" -H "Authorization: Bearer <your_api_key>"
```

响应示例：
```
{
    "guardrails": [
        {
            "guardrail_id": "123e4567-e89b-12d3-a456-426614174000",
            "guardrail_name": "my-bedrock-guard",
            "litellm_params": {
      ...
```

#### `POST` /guardrails

**创建安全护栏**

创建新的 Guardrail。

👉 [Guardrail docs](https://docs.litellm.ai/docs/proxy/guardrails/quick_start)

请求示例：
```bash
curl -X POST "http://localhost:4000/guardrails" \
    -H "Authorization: Bearer <your_api_key>" \
    -H "Content-Type: application/json" \
    -d '{
        "guardrail": {
            "guardrail_name": "my-bedrock-guard",
            "litellm_params": {
                "guardrail": "bedrock",
                "mode": "pre_call",
                "guardrailIdentifier": "f...
```

#### `PUT` /guardrails/{guardrail_id}

**更新安全护栏**

更新已有 Guardrail。

👉 [Guardrail docs](https://docs.litellm.ai/docs/proxy/guardrails/quick_start)

请求示例：
```bash
curl -X PUT "http://localhost:4000/guardrails/123e4567-e89b-12d3-a456-426614174000" \
    -H "Authorization: Bearer <your_api_key>" \
    -H "Content-Type: application/json" \
    -d '{
        "guardrail": {
            "guardrail_name": "updated-bedrock-guard",
            "litellm_params": {
                "guardrail": "bedrock",
                "mode": "pre_c...
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `guardrail_id` | 路径参数 | ✅ |  |

#### `DELETE` /guardrails/{guardrail_id}

**删除安全护栏**

删除 Guardrail。

👉 [Guardrail docs](https://docs.litellm.ai/docs/proxy/guardrails/quick_start)

请求示例：
```bash
curl -X DELETE "http://localhost:4000/guardrails/123e4567-e89b-12d3-a456-426614174000" \
    -H "Authorization: Bearer <your_api_key>"
```

响应示例：
```json
{
    "message": "Guardrail 123e4567-e89b-12d3-a456-426614174000 deleted successfully"
}
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `guardrail_id` | 路径参数 | ✅ |  |

#### `PATCH` /guardrails/{guardrail_id}

**局部更新 Guardrail**

对已有 Guardrail 执行局部更新。

👉 [Guardrail docs](https://docs.litellm.ai/docs/proxy/guardrails/quick_start)

该端点支持只更新 Guardrail 的部分字段，无需提交整个对象。
仅可更新以下字段：
- guardrail_name: The name of the guardrail
- default_on: Whether the guardrail is enabled by default
- guardrail_info: Additional information about the guardrail

请求示例：
```bash
curl -X PATCH "http://localhost:4000/guardrails/123e45...
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `guardrail_id` | 路径参数 | ✅ |  |

#### `GET` /guardrails/{guardrail_id}

**获取安全护栏信息**

根据 ID 获取指定 Guardrail 的详细信息。

👉 [Guardrail docs](https://docs.litellm.ai/docs/proxy/guardrails/quick_start)

请求示例：
```bash
curl -X GET "http://localhost:4000/guardrails/123e4567-e89b-12d3-a456-426614174000/info" \
    -H "Authorization: Bearer <your_api_key>"
```

响应示例：
```
{
    "guardrail_id": "123e4567-e89b-12d3-a456-426614174000",
    "guardrail_name": "my-bedrock-guard",
    "litellm_params": {
        "guardrail": "bedrock",
        "...
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `guardrail_id` | 路径参数 | ✅ |  |

#### `POST` /guardrails/register

**注册 Guardrail**

提交一个用于接入的 Guardrail（团队提交）。

接收一个 Guardrail 配置（位于
[Generic Guardrail API](https://docs.litellm.ai/docs/adding_provider/generic_guardrail_api) format.
提交后状态为 `pending_review`，待管理员审批后生效。

#### `GET` /guardrails/submissions

**列表 Guardrail Submissions**

列出团队的 Guardrail 提交（仅管理员），仅返回带有 `team_id` 的 Guardrail。

状态取值：`pending_review`（团队已提交、待审批）、`active`（已通过）、`rejected`（已驳回）。

可选过滤条件：
- status: pending_review | active | rejected
- team_id: filter by specific team
- search: name/description

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `status` | 查询参数 | ❌ |  |
| `team_id` | 查询参数 | ❌ |  |
| `search` | 查询参数 | ❌ |  |

#### `GET` /guardrails/submissions/{guardrail_id}

**获取 Guardrail Submission**

根据 ID 获取单个 Guardrail 提交（仅管理员）。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `guardrail_id` | 路径参数 | ✅ |  |

#### `POST` /guardrails/submissions/{guardrail_id}/approve

**审批 Guardrail 提交**

审批通过待审核的 Guardrail 提交：将状态置为 `active` 并在内存中初始化（仅管理员）。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `guardrail_id` | 路径参数 | ✅ |  |

#### `POST` /guardrails/submissions/{guardrail_id}/reject

**驳回 Guardrail 提交**

驳回 Guardrail 提交（仅管理员）。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `guardrail_id` | 路径参数 | ✅ |  |

#### `GET` /guardrails/{guardrail_id}/info

**获取安全护栏信息**

根据 ID 获取指定 Guardrail 的详细信息。

👉 [Guardrail docs](https://docs.litellm.ai/docs/proxy/guardrails/quick_start)

请求示例：
```bash
curl -X GET "http://localhost:4000/guardrails/123e4567-e89b-12d3-a456-426614174000/info" \
    -H "Authorization: Bearer <your_api_key>"
```

响应示例：
```
{
    "guardrail_id": "123e4567-e89b-12d3-a456-426614174000",
    "guardrail_name": "my-bedrock-guard",
    "litellm_params": {
        "guardrail": "bedrock",
        "...
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `guardrail_id` | 路径参数 | ✅ |  |

#### `GET` /guardrails/ui/add_guardrail_settings

**获取 Guardrail Ui Settings**

获取 Guardrail 的 UI 配置。

返回：
- Supported entities for guardrails
- Supported modes for guardrails
- PII entity categories for UI organization
- Content filter settings (patterns and categories)

#### `GET` /guardrails/ui/category_yaml/{category_name}

**获取 Category Yaml**

获取指定内容过滤类别的 YAML 或 JSON 内容。

参数：
    category_name: The name of the category (e.g., "bias_gender", "harmful_self_harm")

返回：
    The raw YAML or JSON content of the category file with file type indicator

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `category_name` | 路径参数 | ✅ |  |

#### `GET` /guardrails/ui/major_airlines

**获取 Major Airlines**

从 IATA 获取主要航空公司列表（用于竞争对手/航司类型识别）。
返回航司 ID、匹配变体（以竖线分隔）以及标签。

#### `POST` /guardrails/validate_blocked_words_file

**校验 blocked_words 文件**

校验 `blocked_words` YAML 文件内容。

参数：
    request: Dictionary with 'file_content' key containing the YAML string

返回：
    Dictionary with 'valid' boolean and either 'message'/'errors' depending on result

请求示例：
```json
{
    "file_content": "blocked_words:\n  - keyword: \"test\"\n    action: \"BLOCK\""
}
```

成功示例 响应：
```json
{
    "valid": true,
    "message": "Valid YAML file with 2 blocked words"
}
```

错误示例 响应：
```
{
    "valid...
```

#### `GET` /guardrails/ui/provider_specific_params

**获取 Provider Specific Params**

获取不同 Guardrail 类型对应的提供商专属参数。

返回一个将 Guardrail 提供商映射到其专属参数的字典，包含参数名、描述，以及是否必填。

响应示例：
```
{
    "bedrock": {
        "guardrailIdentifier": {
            "description": "The ID of your guardrail on Bedrock",
            "required": true,
            "type": null
        },
        "guardrailVersion": {
            "description": "The version of ...
```

#### `POST` /guardrails/test_custom_code

**测试自定义代码 Guardrail**

在不创建 Guardrail 的情况下测试自定义代码 Guardrail 的逻辑。

该端点允许管理员通过以下方式试用自定义代码 Guardrail：
1. Compiling the provided code in a sandbox
2. Executing the apply_guardrail function with test input
3. Returning the result (allow/block/modify)

👉 [Custom Code Guardrail docs](https://docs.litellm.ai/docs/proxy/guardrails/custom_code_guardrail)

请求示例：
```bash
curl -X POST "http://localhost:4000/guardrails/test_custom_code" \
    -H "Authorization...
```

#### `GET` /guardrails/usage/overview

**Guardrails 用法 Overview**

返回给管理后台使用的 Guardrail 性能概览。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `start_date` | 查询参数 | ❌ | YYYY-MM-DD |
| `end_date` | 查询参数 | ❌ | YYYY-MM-DD |

#### `GET` /guardrails/usage/detail/{guardrail_id}

**Guardrails 用法 Detail**

返回单个 Guardrail 的用量指标与时间序列数据。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `guardrail_id` | 路径参数 | ✅ |  |
| `start_date` | 查询参数 | ❌ |  |
| `end_date` | 查询参数 | ❌ |  |

#### `GET` /guardrails/usage/logs

**Guardrails 用法 Logs**

通过索引从 `SpendLogs` 获取 Guardrail（或策略）的分页执行日志。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `guardrail_id` | 查询参数 | ❌ |  |
| `policy_id` | 查询参数 | ❌ |  |
| `page` | 查询参数 | ❌ |  |
| `page_size` | 查询参数 | ❌ |  |
| `action` | 查询参数 | ❌ |  |
| `start_date` | 查询参数 | ❌ |  |
| `end_date` | 查询参数 | ❌ |  |

### 策略

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/policies/usage/overview` | Policies 用法 Overview |
| `GET` | `/policies/list` | 列表 Policies |
| `POST` | `/policies` | 创建 Policy |
| `GET` | `/policies/name/{policy_name}/versions` | 列表 Policy Versions |
| `POST` | `/policies/name/{policy_name}/versions` | 创建 Policy Version |
| `PUT` | `/policies/{policy_id}/status` | 更新 Policy Version Status |
| `GET` | `/policies/compare` | Compare Policy Versions |
| `DELETE` | `/policies/name/{policy_name}/all-versions` | 删除 All Policy Versions |
| `GET` | `/policies/{policy_id}` | 获取 Policy |
| `PUT` | `/policies/{policy_id}` | 更新 Policy |
| `DELETE` | `/policies/{policy_id}` | 删除 Policy |
| `GET` | `/policies/{policy_id}/resolved-guardrails` | 获取 Resolved Guardrails |
| `POST` | `/policies/test-pipeline` | Test Pipeline |
| `GET` | `/policies/attachments/list` | 列表 Policy Attachments |
| `POST` | `/policies/attachments` | 创建 Policy Attachment |
| `GET` | `/policies/attachments/{attachment_id}` | 获取 Policy Attachment |
| `DELETE` | `/policies/attachments/{attachment_id}` | 删除 Policy Attachment |
| `POST` | `/policies/resolve` | Resolve Policies For Context |
| `POST` | `/policies/attachments/estimate-impact` | Estimate Attachment Impact |

#### `GET` /policies/usage/overview

**Policies 用法 Overview**

返回给管理后台使用的策略性能概览。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `start_date` | 查询参数 | ❌ | YYYY-MM-DD |
| `end_date` | 查询参数 | ❌ | YYYY-MM-DD |

#### `GET` /policies/list

**列表 Policies**

从数据库列出全部策略，可按 `version_status` 过滤。

查询参数：
- version_status: Optional. One of "draft", "published", "production".
  If omitted, all versions are returned.

请求示例：
```bash
curl -X GET "http://localhost:4000/policies/list" \
    -H "Authorization: Bearer <your_api_key>"
curl -X GET "http://localhost:4000/policies/list?version_status=production" \
    -H "Authorization: Bearer <your_api_key>"
```

响应示例：
```
{
    "policies": [
...
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `version_status` | 查询参数 | ❌ |  |

#### `POST` /policies

**创建 Policy**

创建新的策略。

请求示例：
```bash
curl -X POST "http://localhost:4000/policies" \
    -H "Authorization: Bearer <your_api_key>" \
    -H "Content-Type: application/json" \
    -d '{
        "policy_name": "global-baseline",
        "description": "Base guardrails for all requests",
        "guardrails_add": ["pii_masking", "prompt_injection"],
        "guardrails_remove": []
    }'
```

响应示例：
```
{
    "policy_id": "123e4567-e89b-12d3-a456-426614174000",
    "policy_...
```

#### `GET` /policies/name/{policy_name}/versions

**列表 Policy Versions**

按名称列出策略的全部版本，按 `version_number` 倒序排列。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `policy_name` | 路径参数 | ✅ |  |

#### `POST` /policies/name/{policy_name}/versions

**创建 Policy Version**

为策略创建一个新的 Draft 版本，并从源版本复制全部字段。
若未提供 `source_policy_id`，则以当前生产版本作为来源。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `policy_name` | 路径参数 | ✅ |  |

#### `PUT` /policies/{policy_id}/status

**更新 Policy Version Status**

更新策略版本的状态，可用状态流转如下：
- draft -> published
- published -> production (demotes current production to published)
- production -> published (demotes, policy becomes inactive)

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `policy_id` | 路径参数 | ✅ |  |

#### `GET` /policies/compare

**对比策略版本**

比较两个策略版本。查询参数：`version_a`、`version_b`（策略版本 ID）。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `version_a` | 查询参数 | ✅ |  |
| `version_b` | 查询参数 | ✅ |  |

#### `DELETE` /policies/name/{policy_name}/all-versions

**删除 All Policy Versions**

删除策略的全部版本，并同步从内存注册表中移除。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `policy_name` | 路径参数 | ✅ |  |

#### `GET` /policies/{policy_id}

**获取 Policy**

根据 ID 获取策略。

请求示例：
```bash
curl -X GET "http://localhost:4000/policies/123e4567-e89b-12d3-a456-426614174000" \
    -H "Authorization: Bearer <your_api_key>"
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `policy_id` | 路径参数 | ✅ |  |

#### `PUT` /policies/{policy_id}

**更新 Policy**

更新已有策略。

请求示例：
```bash
curl -X PUT "http://localhost:4000/policies/123e4567-e89b-12d3-a456-426614174000" \
    -H "Authorization: Bearer <your_api_key>" \
    -H "Content-Type: application/json" \
    -d '{
        "description": "Updated description",
        "guardrails_add": ["pii_masking", "toxicity_filter"]
    }'
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `policy_id` | 路径参数 | ✅ |  |

#### `DELETE` /policies/{policy_id}

**删除 Policy**

删除策略。

请求示例：
```bash
curl -X DELETE "http://localhost:4000/policies/123e4567-e89b-12d3-a456-426614174000" \
    -H "Authorization: Bearer <your_api_key>"
```

响应示例：
```json
{
    "message": "Policy 123e4567-e89b-12d3-a456-426614174000 deleted successfully"
}
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `policy_id` | 路径参数 | ✅ |  |

#### `GET` /policies/{policy_id}/resolved-guardrails

**获取 Resolved Guardrails**

获取某策略解析后的 Guardrail 集合（包含继承得到的 Guardrail）。

该端点会解析完整的继承链并返回最终
返回该策略最终会应用的 Guardrail 集合。

请求示例：
```bash
curl -X GET "http://localhost:4000/policies/123e4567-e89b-12d3-a456-426614174000/resolved-guardrails" \
    -H "Authorization: Bearer <your_api_key>"
```

响应示例：
```
{
    "policy_id": "123e4567-e89b-12d3-a456-426614174000",
    "policy_name": "healthcar...
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `policy_id` | 路径参数 | ✅ |  |

#### `POST` /policies/test-pipeline

**测试 Guardrail 流水线**

使用示例消息测试 Guardrail 流水线。

对传入的测试消息执行 Guardrail 流水线，并返回结果。
逐步返回测试结果：包含每个 Guardrail 的通过 / 失败状态、执行的动作等信息。
以及耗时信息。

请求示例：
```bash
curl -X POST "http://localhost:4000/policies/test-pipeline" \
    -H "Authorization: Bearer <your_api_key>" \
    -H "Content-Type: application/json" \
    -d '{
        "pipeline": {
            "mode": "pre_call",
            "steps": [
           ...
```

#### `GET` /policies/attachments/list

**列表 Policy Attachments**

从数据库列出全部策略 Attachment。

请求示例：
```bash
curl -X GET "http://localhost:4000/policies/attachments/list" \
    -H "Authorization: Bearer <your_api_key>"
```

响应示例：
```
{
    "attachments": [
        {
            "attachment_id": "123e4567-e89b-12d3-a456-426614174000",
            "policy_name": "global-baseline",
            "scope": "*",
            "teams": [],
            "keys": [],
            "models": [],
            "created_at": "2024-01-01...
```

#### `POST` /policies/attachments

**创建 Policy Attachment**

创建新的策略 Attachment。

请求示例：
```bash
curl -X POST "http://localhost:4000/policies/attachments" \
    -H "Authorization: Bearer <your_api_key>" \
    -H "Content-Type: application/json" \
    -d '{
        "policy_name": "global-baseline",
        "scope": "*"
    }'
```

带上 team 专属 attachment 的示例:
```bash
curl -X POST "http://localhost:4000/policies/attachments" \
    -H "Authorization: Bearer <your_api_key>" \
    -H "Content-Type: application/json" \
    -d '{
...
```

#### `GET` /policies/attachments/{attachment_id}

**获取 Policy Attachment**

根据 ID 获取策略 Attachment。

请求示例：
```bash
curl -X GET "http://localhost:4000/policies/attachments/123e4567-e89b-12d3-a456-426614174000" \
    -H "Authorization: Bearer <your_api_key>"
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `attachment_id` | 路径参数 | ✅ |  |

#### `DELETE` /policies/attachments/{attachment_id}

**删除 Policy Attachment**

删除策略 Attachment。

请求示例：
```bash
curl -X DELETE "http://localhost:4000/policies/attachments/123e4567-e89b-12d3-a456-426614174000" \
    -H "Authorization: Bearer <your_api_key>"
```

响应示例：
```json
{
    "message": "Attachment 123e4567-e89b-12d3-a456-426614174000 deleted successfully"
}
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `attachment_id` | 路径参数 | ✅ |  |

#### `POST` /policies/resolve

**解析上下文命中的策略**

解析给定上下文会命中哪些策略与 Guardrail。

用于调试「某次请求将命中哪些 Guardrail」。
对于给定的 team / key / model / tags 组合，会命中哪些 Guardrail？」

请求示例：
```bash
curl -X POST "http://localhost:4000/policies/resolve" \
    -H "Authorization: Bearer <your_api_key>" \
    -H "Content-Type: application/json" \
    -d '{
        "tags": ["healthcare"],
        "model": "gpt-4"
    }'
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `force_sync` | 查询参数 | ❌ | Force a DB sync before resolving. Default uses in-memory cache. |

#### `POST` /policies/attachments/estimate-impact

**评估 Attachment 影响范围**

估算某次策略 Attachment 会影响多少个 Key 与团队。

在创建 attachment 之前使用该接口预览影响范围。

请求示例：
```bash
curl -X POST "http://localhost:4000/policies/attachments/estimate-impact" \
    -H "Authorization: Bearer <your_api_key>" \
    -H "Content-Type: application/json" \
    -d '{
        "policy_name": "hipaa-compliance",
        "tags": ["healthcare", "health-*"]
    }'
```

### 策略管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/policy/validate` | Validate Policy |
| `GET` | `/policy/list` | 列表 Policies |
| `GET` | `/policy/info/{policy_name}` | 获取 Policy Info |
| `POST` | `/policy/test` | Test Policy Matching |
| `GET` | `/policy/templates` | 获取 Policy Templates |
| `POST` | `/policy/templates/enrich` | Enrich Policy Template |
| `POST` | `/policy/templates/enrich/stream` | Enrich Policy Template Stream |
| `POST` | `/policy/templates/suggest` | Suggest Policy Templates |
| `POST` | `/policy/templates/test` | Test Policy Template |

#### `POST` /policy/validate

**校验策略**

在应用之前校验策略配置。

检查：
- All referenced guardrails exist in the guardrail registry
- All non-wildcard team aliases exist in the database
- All non-wildcard key aliases exist in the database
- Inheritance chains are valid (no cycles, parents exist)
- Scope patterns are syntactically valid

返回：
- valid: True if the policy configuration is valid (no blocking errors)
- errors: List of blocking validation errors
- warnings: List of non-blocking validation wa...

#### `GET` /policy/list

**列表 Policies**

列出全部已加载策略及其解析后的 Guardrail 集合。

返回每个策略的详细信息，包括:
- Inheritance configuration
- Scope (teams, keys, models)
- Guardrails to add/remove
- Resolved guardrails (after inheritance)
- Inheritance chain

#### `GET` /policy/info/{policy_name}

**获取 Policy Info**

获取指定策略的详细信息。

返回：
- Policy configuration
- Resolved guardrails (after inheritance)
- Inheritance chain

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `policy_name` | 路径参数 | ✅ |  |

#### `POST` /policy/test

**测试策略命中**

测试给定请求上下文会命中哪些策略。

适合用于调试与理解策略行为。

请求体：
```json
{
    "team_alias": "healthcare-team",
    "key_alias": "my-api-key",
    "model": "gpt-4"
}
```

返回：
- matching_policies: List of policy names that match
- resolved_guardrails: Final list of guardrails that would be applied

#### `GET` /policy/templates

**获取 Policy Templates**

获取给管理 UI 使用的策略模板（预配置的 Guardrail 组合）。

优先从 GitHub 拉取；失败时自动回退到本地备份。
设置 `LITELLM_LOCAL_POLICY_TEMPLATES=true` 可跳过 GitHub，仅使用本地备份。

#### `POST` /policy/templates/enrich

**丰富策略模板**

使用 LLM 发现的数据（如竞品名称）丰富策略模板。

调用已接入的 LLM，为给定品牌名挖掘竞争对手。
然后返回填充了新发现数据的 `guardrailDefinitions`。

#### `POST` /policy/templates/enrich/stream

**流式丰富策略模板**

以 SSE 事件流的方式实时返回 LLM 生成的竞品名称。

事件：
- data: {"type": "competitor", "name": "..."}  — each competitor as discovered
- data: {"type": "done", "competitors": [...], "competitor_variations": {...}, "guardrailDefinitions": [...]}

#### `POST` /policy/templates/suggest

**推荐策略模板**

基于攻击示例与描述，使用 AI 推荐策略模板。

调用带 Tool Calling 的 LLM，将用户需求匹配到可用模板。

#### `POST` /policy/templates/test

**测试策略模板**

在不实际创建的情况下，使用文本输入测试策略模板对应的 Guardrail。

根据模板定义实例化临时 Guardrail 并执行。
针对传入的文本运行，并返回每个 Guardrail 的结果，方便用户
让用户在正式创建之前确认模板是否能解决自己的问题。

### 合规

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/compliance/eu-ai-act` | Check Eu Ai Act Compliance |
| `POST` | `/compliance/gdpr` | Check Gdpr Compliance |

#### `POST` /compliance/eu-ai-act

**EU AI Act 合规检查**

检查某条花费日志的 EU AI Act 合规情况。

检查：
- Art. 9: Guardrails applied (any guardrail)
- Art. 5: Content screened before LLM (pre-call guardrails)
- Art. 12: Audit record complete (user_id, model, timestamp, guardrail_results)

#### `POST` /compliance/gdpr

**GDPR 合规检查**

检查某条花费日志的 GDPR 合规情况。

检查：
- Art. 32: Data protection applied (pre-call guardrails)
- Art. 5(1)(c): Sensitive data protected (masked/blocked or no issues)
- Art. 30: Audit record complete (user_id, model, timestamp, guardrail_results)

### 审计日志

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/audit` | 获取 Audit Logs |
| `GET` | `/audit/{id}` | 获取 Audit Log By Id |

#### `GET` /audit

**获取 Audit Logs**

获取全部审计日志（支持过滤与分页）。

返回与指定过滤条件匹配的审计日志（分页响应）。

注意：`object_team_id` 与 `object_key_hash` 使用 Prisma 的 JSON 路径过滤。
需要 PostgreSQL 支持。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `page` | 查询参数 | ❌ |  |
| `page_size` | 查询参数 | ❌ |  |
| `changed_by` | 查询参数 | ❌ | Filter by user or system that performed the action |
| `changed_by_api_key` | 查询参数 | ❌ | Filter by API key hash that performed the action |
| `action` | 查询参数 | ❌ | Filter by action type (create, update, delete) |
| `table_name` | 查询参数 | ❌ | Filter by table name that was modified |
| `object_id` | 查询参数 | ❌ | Filter by ID of the object that was modified |
| `start_date` | 查询参数 | ❌ | Filter logs after this date |
| `end_date` | 查询参数 | ❌ | Filter logs before this date |
| `object_team_id` | 查询参数 | ❌ | Filter by team_id present in before_value or updated_values JSON (PostgreSQL only) |

#### `GET` /audit/{id}

**获取 Audit Log By Id**

根据 ID 获取指定审计日志条目的详细信息。

参数：
    id (str): The unique identifier of the audit log entry

返回：
    AuditLog响应： Detailed information about the audit log entry

异常:
    HTTPException: If the audit log is not found or if there's a database connection error

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `id` | 路径参数 | ✅ |  |

