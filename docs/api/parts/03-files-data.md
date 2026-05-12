## 文件与数据

### 文件管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/files` | 创建 File |
| `GET` | `/files` | 文件列表 |
| `POST` | `/v1/files` | 创建 File |
| `GET` | `/v1/files` | 文件列表 |
| `POST` | `/{provider}/v1/files` | 创建 File |
| `GET` | `/{provider}/v1/files` | 文件列表 |
| `GET` | `/files/{file_id}/content` | 获取文件内容 |
| `GET` | `/v1/files/{file_id}/content` | 获取文件内容 |
| `GET` | `/{provider}/v1/files/{file_id}/content` | 获取文件内容 |
| `GET` | `/files/{file_id}` | 获取文件 |
| `DELETE` | `/files/{file_id}` | 删除文件 |
| `GET` | `/v1/files/{file_id}` | 获取文件 |
| `DELETE` | `/v1/files/{file_id}` | 删除文件 |
| `GET` | `/{provider}/v1/files/{file_id}` | 获取文件 |
| `DELETE` | `/{provider}/v1/files/{file_id}` | 删除文件 |

#### `POST` /files

**创建 File**

上传可在 Assistants API、Batch API 之间共享使用的文件。
等效于 `POST https://api.openai.com/v1/files`。

与该规范参数一致：https://platform.openai.com/docs/api-reference/files/create

Curl 示例
```
curl http://localhost:4000/v1/files         -H "Authorization: Bearer sk-1234"         -F purpose="batch"         -F file="@mydata.jsonl"
    -F expires_after[anchor]="created_at"         -F expires_after[seconds]=2592000
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `provider` | 查询参数 | ❌ |  |

#### `GET` /files

**文件列表**

返回指定文件的信息；该文件可在 Assistants API、Batch API 之间共享使用。
等效于 `GET https://api.openai.com/v1/files/`。

与该规范参数一致：https://platform.openai.com/docs/api-reference/files/list

Curl 示例
```
curl http://localhost:4000/v1/files        -H "Authorization: Bearer sk-1234"

```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `provider` | 查询参数 | ❌ |  |
| `target_model_names` | 查询参数 | ❌ |  |
| `purpose` | 查询参数 | ❌ |  |

#### `POST` /v1/files

**创建 File**

上传可在 Assistants API、Batch API 之间共享使用的文件。
等效于 `POST https://api.openai.com/v1/files`。

与该规范参数一致：https://platform.openai.com/docs/api-reference/files/create

Curl 示例
```
curl http://localhost:4000/v1/files         -H "Authorization: Bearer sk-1234"         -F purpose="batch"         -F file="@mydata.jsonl"
    -F expires_after[anchor]="created_at"         -F expires_after[seconds]=2592000
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `provider` | 查询参数 | ❌ |  |

#### `GET` /v1/files

**文件列表**

返回指定文件的信息；该文件可在 Assistants API、Batch API 之间共享使用。
等效于 `GET https://api.openai.com/v1/files/`。

与该规范参数一致：https://platform.openai.com/docs/api-reference/files/list

Curl 示例
```
curl http://localhost:4000/v1/files        -H "Authorization: Bearer sk-1234"

```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `provider` | 查询参数 | ❌ |  |
| `target_model_names` | 查询参数 | ❌ |  |
| `purpose` | 查询参数 | ❌ |  |

#### `POST` /{provider}/v1/files

**创建 File**

上传可在 Assistants API、Batch API 之间共享使用的文件。
等效于 `POST https://api.openai.com/v1/files`。

与该规范参数一致：https://platform.openai.com/docs/api-reference/files/create

Curl 示例
```
curl http://localhost:4000/v1/files         -H "Authorization: Bearer sk-1234"         -F purpose="batch"         -F file="@mydata.jsonl"
    -F expires_after[anchor]="created_at"         -F expires_after[seconds]=2592000
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `provider` | 路径参数 | ✅ |  |

#### `GET` /{provider}/v1/files

**文件列表**

返回指定文件的信息；该文件可在 Assistants API、Batch API 之间共享使用。
等效于 `GET https://api.openai.com/v1/files/`。

与该规范参数一致：https://platform.openai.com/docs/api-reference/files/list

Curl 示例
```
curl http://localhost:4000/v1/files        -H "Authorization: Bearer sk-1234"

```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `provider` | 路径参数 | ✅ |  |
| `target_model_names` | 查询参数 | ❌ |  |
| `purpose` | 查询参数 | ❌ |  |

#### `GET` /files/{file_id}/content

**获取文件内容**

返回指定文件的信息；该文件可在 Assistants API、Batch API 之间共享使用。
等效于 `GET https://api.openai.com/v1/files/{file_id}/content`。

与该规范参数一致：https://platform.openai.com/docs/api-reference/files/retrieve-contents

Curl 示例
```
curl http://localhost:4000/v1/files/file-abc123/content         -H "Authorization: Bearer sk-1234"

```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `file_id` | 路径参数 | ✅ |  |
| `provider` | 查询参数 | ❌ |  |

#### `GET` /v1/files/{file_id}/content

**获取文件内容**

返回指定文件的信息；该文件可在 Assistants API、Batch API 之间共享使用。
等效于 `GET https://api.openai.com/v1/files/{file_id}/content`。

与该规范参数一致：https://platform.openai.com/docs/api-reference/files/retrieve-contents

Curl 示例
```
curl http://localhost:4000/v1/files/file-abc123/content         -H "Authorization: Bearer sk-1234"

```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `file_id` | 路径参数 | ✅ |  |
| `provider` | 查询参数 | ❌ |  |

#### `GET` /{provider}/v1/files/{file_id}/content

**获取文件内容**

返回指定文件的信息；该文件可在 Assistants API、Batch API 之间共享使用。
等效于 `GET https://api.openai.com/v1/files/{file_id}/content`。

与该规范参数一致：https://platform.openai.com/docs/api-reference/files/retrieve-contents

Curl 示例
```
curl http://localhost:4000/v1/files/file-abc123/content         -H "Authorization: Bearer sk-1234"

```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `file_id` | 路径参数 | ✅ |  |
| `provider` | 路径参数 | ✅ |  |

#### `GET` /files/{file_id}

**获取文件**

返回指定文件的信息；该文件可在 Assistants API、Batch API 之间共享使用。
等效于 `GET https://api.openai.com/v1/files/{file_id}`。

与该规范参数一致：https://platform.openai.com/docs/api-reference/files/retrieve

Curl 示例
```
curl http://localhost:4000/v1/files/file-abc123         -H "Authorization: Bearer sk-1234"

```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `file_id` | 路径参数 | ✅ |  |
| `provider` | 查询参数 | ❌ |  |

#### `DELETE` /files/{file_id}

**删除文件**

删除指定文件，该文件可在 Assistants API、Batch API 之间共享使用。
等效于 `DELETE https://api.openai.com/v1/files/{file_id}`。

与该规范参数一致：https://platform.openai.com/docs/api-reference/files/delete

Curl 示例
```
curl http://localhost:4000/v1/files/file-abc123     -X DELETE     -H "Authorization: Bearer $OPENAI_API_KEY"

```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `file_id` | 路径参数 | ✅ |  |
| `provider` | 查询参数 | ❌ |  |

#### `GET` /v1/files/{file_id}

**获取文件**

返回指定文件的信息；该文件可在 Assistants API、Batch API 之间共享使用。
等效于 `GET https://api.openai.com/v1/files/{file_id}`。

与该规范参数一致：https://platform.openai.com/docs/api-reference/files/retrieve

Curl 示例
```
curl http://localhost:4000/v1/files/file-abc123         -H "Authorization: Bearer sk-1234"

```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `file_id` | 路径参数 | ✅ |  |
| `provider` | 查询参数 | ❌ |  |

#### `DELETE` /v1/files/{file_id}

**删除文件**

删除指定文件，该文件可在 Assistants API、Batch API 之间共享使用。
等效于 `DELETE https://api.openai.com/v1/files/{file_id}`。

与该规范参数一致：https://platform.openai.com/docs/api-reference/files/delete

Curl 示例
```
curl http://localhost:4000/v1/files/file-abc123     -X DELETE     -H "Authorization: Bearer $OPENAI_API_KEY"

```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `file_id` | 路径参数 | ✅ |  |
| `provider` | 查询参数 | ❌ |  |

#### `GET` /{provider}/v1/files/{file_id}

**获取文件**

返回指定文件的信息；该文件可在 Assistants API、Batch API 之间共享使用。
等效于 `GET https://api.openai.com/v1/files/{file_id}`。

与该规范参数一致：https://platform.openai.com/docs/api-reference/files/retrieve

Curl 示例
```
curl http://localhost:4000/v1/files/file-abc123         -H "Authorization: Bearer sk-1234"

```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `file_id` | 路径参数 | ✅ |  |
| `provider` | 路径参数 | ✅ |  |

#### `DELETE` /{provider}/v1/files/{file_id}

**删除文件**

删除指定文件，该文件可在 Assistants API、Batch API 之间共享使用。
等效于 `DELETE https://api.openai.com/v1/files/{file_id}`。

与该规范参数一致：https://platform.openai.com/docs/api-reference/files/delete

Curl 示例
```
curl http://localhost:4000/v1/files/file-abc123     -X DELETE     -H "Authorization: Bearer $OPENAI_API_KEY"

```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `file_id` | 路径参数 | ✅ |  |
| `provider` | 路径参数 | ✅ |  |

### 批处理

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/batches` | 创建批处理 |
| `GET` | `/batches` | 批处理列表 |
| `POST` | `/v1/batches` | 创建批处理 |
| `GET` | `/v1/batches` | 批处理列表 |
| `POST` | `/{provider}/v1/batches` | 创建批处理 |
| `GET` | `/{provider}/v1/batches` | 批处理列表 |
| `GET` | `/batches/{batch_id}` | Retrieve Batch |
| `GET` | `/v1/batches/{batch_id}` | Retrieve Batch |
| `GET` | `/{provider}/v1/batches/{batch_id}` | Retrieve Batch |
| `POST` | `/batches/{batch_id}/cancel` | 取消批处理 |
| `POST` | `/v1/batches/{batch_id}/cancel` | 取消批处理 |
| `POST` | `/{provider}/v1/batches/{batch_id}/cancel` | 取消批处理 |

#### `POST` /batches

**创建批处理**

创建 API 请求的批量作业，用于异步处理。
等效于 `POST https://api.openai.com/v1/batch`。
与该规范参数一致：https://platform.openai.com/docs/api-reference/batch

Curl 示例
```
curl http://localhost:4000/v1/batches         -H "Authorization: Bearer sk-1234"         -H "Content-Type: application/json"         -d '{
        "input_file_id": "file-abc123",
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h"
}'
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `provider` | 查询参数 | ❌ |  |

#### `GET` /batches

**批处理列表**

列出 
等效于 `GET https://api.openai.com/v1/batches/`。
与该规范参数一致：https://platform.openai.com/docs/api-reference/batch/list

Curl 示例
```
curl http://localhost:4000/v1/batches?limit=2     -H "Authorization: Bearer sk-1234"     -H "Content-Type: application/json" 
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `provider` | 查询参数 | ❌ |  |
| `limit` | 查询参数 | ❌ |  |
| `after` | 查询参数 | ❌ |  |
| `target_model_names` | 查询参数 | ❌ |  |

#### `POST` /v1/batches

**创建批处理**

创建 API 请求的批量作业，用于异步处理。
等效于 `POST https://api.openai.com/v1/batch`。
与该规范参数一致：https://platform.openai.com/docs/api-reference/batch

Curl 示例
```
curl http://localhost:4000/v1/batches         -H "Authorization: Bearer sk-1234"         -H "Content-Type: application/json"         -d '{
        "input_file_id": "file-abc123",
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h"
}'
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `provider` | 查询参数 | ❌ |  |

#### `GET` /v1/batches

**批处理列表**

列出 
等效于 `GET https://api.openai.com/v1/batches/`。
与该规范参数一致：https://platform.openai.com/docs/api-reference/batch/list

Curl 示例
```
curl http://localhost:4000/v1/batches?limit=2     -H "Authorization: Bearer sk-1234"     -H "Content-Type: application/json" 
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `provider` | 查询参数 | ❌ |  |
| `limit` | 查询参数 | ❌ |  |
| `after` | 查询参数 | ❌ |  |
| `target_model_names` | 查询参数 | ❌ |  |

#### `POST` /{provider}/v1/batches

**创建批处理**

创建 API 请求的批量作业，用于异步处理。
等效于 `POST https://api.openai.com/v1/batch`。
与该规范参数一致：https://platform.openai.com/docs/api-reference/batch

Curl 示例
```
curl http://localhost:4000/v1/batches         -H "Authorization: Bearer sk-1234"         -H "Content-Type: application/json"         -d '{
        "input_file_id": "file-abc123",
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h"
}'
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `provider` | 路径参数 | ✅ |  |

#### `GET` /{provider}/v1/batches

**批处理列表**

列出 
等效于 `GET https://api.openai.com/v1/batches/`。
与该规范参数一致：https://platform.openai.com/docs/api-reference/batch/list

Curl 示例
```
curl http://localhost:4000/v1/batches?limit=2     -H "Authorization: Bearer sk-1234"     -H "Content-Type: application/json" 
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `provider` | 路径参数 | ✅ |  |
| `limit` | 查询参数 | ❌ |  |
| `after` | 查询参数 | ❌ |  |
| `target_model_names` | 查询参数 | ❌ |  |

#### `GET` /batches/{batch_id}

**获取批处理作业**

获取批处理作业详情。
等效于 `GET https://api.openai.com/v1/batches/{batch_id}`。
与该规范参数一致：https://platform.openai.com/docs/api-reference/batch/retrieve

Curl 示例
```
curl http://localhost:4000/v1/batches/batch_abc123     -H "Authorization: Bearer sk-1234"     -H "Content-Type: application/json" 
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `batch_id` | 路径参数 | ✅ | The ID of the batch to retrieve |
| `provider` | 查询参数 | ❌ |  |

#### `GET` /v1/batches/{batch_id}

**获取批处理作业**

获取批处理作业详情。
等效于 `GET https://api.openai.com/v1/batches/{batch_id}`。
与该规范参数一致：https://platform.openai.com/docs/api-reference/batch/retrieve

Curl 示例
```
curl http://localhost:4000/v1/batches/batch_abc123     -H "Authorization: Bearer sk-1234"     -H "Content-Type: application/json" 
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `batch_id` | 路径参数 | ✅ | The ID of the batch to retrieve |
| `provider` | 查询参数 | ❌ |  |

#### `GET` /{provider}/v1/batches/{batch_id}

**获取批处理作业**

获取批处理作业详情。
等效于 `GET https://api.openai.com/v1/batches/{batch_id}`。
与该规范参数一致：https://platform.openai.com/docs/api-reference/batch/retrieve

Curl 示例
```
curl http://localhost:4000/v1/batches/batch_abc123     -H "Authorization: Bearer sk-1234"     -H "Content-Type: application/json" 
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `provider` | 路径参数 | ✅ |  |
| `batch_id` | 路径参数 | ✅ | The ID of the batch to retrieve |

#### `POST` /batches/{batch_id}/cancel

**取消批处理**

取消批处理作业。
等效于 `POST https://api.openai.com/v1/batches/{batch_id}/cancel`。

与该规范参数一致：https://platform.openai.com/docs/api-reference/batch/cancel

Curl 示例
```
curl http://localhost:4000/v1/batches/batch_abc123/cancel         -H "Authorization: Bearer sk-1234"         -H "Content-Type: application/json"         -X POST

```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `batch_id` | 路径参数 | ✅ |  |
| `provider` | 查询参数 | ❌ |  |

#### `POST` /v1/batches/{batch_id}/cancel

**取消批处理**

取消批处理作业。
等效于 `POST https://api.openai.com/v1/batches/{batch_id}/cancel`。

与该规范参数一致：https://platform.openai.com/docs/api-reference/batch/cancel

Curl 示例
```
curl http://localhost:4000/v1/batches/batch_abc123/cancel         -H "Authorization: Bearer sk-1234"         -H "Content-Type: application/json"         -X POST

```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `batch_id` | 路径参数 | ✅ |  |
| `provider` | 查询参数 | ❌ |  |

#### `POST` /{provider}/v1/batches/{batch_id}/cancel

**取消批处理**

取消批处理作业。
等效于 `POST https://api.openai.com/v1/batches/{batch_id}/cancel`。

与该规范参数一致：https://platform.openai.com/docs/api-reference/batch/cancel

Curl 示例
```
curl http://localhost:4000/v1/batches/batch_abc123/cancel         -H "Authorization: Bearer sk-1234"         -H "Content-Type: application/json"         -X POST

```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `batch_id` | 路径参数 | ✅ |  |
| `provider` | 路径参数 | ✅ |  |

### 微调

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/fine_tuning/jobs` | ✨ (Enterprise) Create Fine-Tuning Job |
| `GET` | `/fine_tuning/jobs` | ✨ (Enterprise) List Fine-Tuning Jobs |
| `POST` | `/v1/fine_tuning/jobs` | ✨ (Enterprise) Create Fine-Tuning Job |
| `GET` | `/v1/fine_tuning/jobs` | ✨ (Enterprise) List Fine-Tuning Jobs |
| `GET` | `/fine_tuning/jobs/{fine_tuning_job_id}` | ✨ (Enterprise) Retrieve Fine-Tuning Job |
| `GET` | `/v1/fine_tuning/jobs/{fine_tuning_job_id}` | ✨ (Enterprise) Retrieve Fine-Tuning Job |
| `POST` | `/fine_tuning/jobs/{fine_tuning_job_id}/cancel` | ✨ (Enterprise) Cancel Fine-Tuning Jobs |
| `POST` | `/v1/fine_tuning/jobs/{fine_tuning_job_id}/cancel` | ✨ (Enterprise) Cancel Fine-Tuning Jobs |

#### `POST` /fine_tuning/jobs

**✨ (Enterprise) Create Fine-Tuning Job**

创建微调作业，从给定数据集开始训练一个新模型。
等效于 `POST https://api.openai.com/v1/fine_tuning/jobs`。

与该规范参数一致：https://platform.openai.com/docs/api-reference/fine-tuning/create

Curl 示例:
```
curl http://localhost:4000/v1/fine_tuning/jobs       -H "Content-Type: application/json"       -H "Authorization: Bearer sk-1234"       -d '{
    "model": "gpt-3.5-turbo",
    "training_file": "file-abc123",
    "...
```

#### `GET` /fine_tuning/jobs

**✨ (Enterprise) List Fine-Tuning Jobs**

列出该组织下的微调作业。
等效于 `GET https://api.openai.com/v1/fine_tuning/jobs`。

支持的查询参数：
- `custom_llm_provider`: Name of the LiteLLM provider
- `after`: Identifier for the last job from the previous pagination request.
- `limit`: Number of fine-tuning jobs to retrieve (default is 20).

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `custom_llm_provider` | 查询参数 | ❌ |  |
| `target_model_names` | 查询参数 | ❌ | Comma separated list of model names to filter by. 示例： 'gpt-4o,gpt-4o-mini' |
| `after` | 查询参数 | ❌ |  |
| `limit` | 查询参数 | ❌ |  |

#### `POST` /v1/fine_tuning/jobs

**✨ (Enterprise) Create Fine-Tuning Job**

创建微调作业，从给定数据集开始训练一个新模型。
等效于 `POST https://api.openai.com/v1/fine_tuning/jobs`。

与该规范参数一致：https://platform.openai.com/docs/api-reference/fine-tuning/create

Curl 示例:
```
curl http://localhost:4000/v1/fine_tuning/jobs       -H "Content-Type: application/json"       -H "Authorization: Bearer sk-1234"       -d '{
    "model": "gpt-3.5-turbo",
    "training_file": "file-abc123",
    "...
```

#### `GET` /v1/fine_tuning/jobs

**✨ (Enterprise) List Fine-Tuning Jobs**

列出该组织下的微调作业。
等效于 `GET https://api.openai.com/v1/fine_tuning/jobs`。

支持的查询参数：
- `custom_llm_provider`: Name of the LiteLLM provider
- `after`: Identifier for the last job from the previous pagination request.
- `limit`: Number of fine-tuning jobs to retrieve (default is 20).

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `custom_llm_provider` | 查询参数 | ❌ |  |
| `target_model_names` | 查询参数 | ❌ | Comma separated list of model names to filter by. 示例： 'gpt-4o,gpt-4o-mini' |
| `after` | 查询参数 | ❌ |  |
| `limit` | 查询参数 | ❌ |  |

#### `GET` /fine_tuning/jobs/{fine_tuning_job_id}

**✨ (Enterprise) Retrieve Fine-Tuning Job**

获取微调作业详情。
等效于 `GET https://api.openai.com/v1/fine_tuning/jobs/{fine_tuning_job_id}`。

支持的查询参数：
- `custom_llm_provider`: Name of the LiteLLM provider
- `fine_tuning_job_id`: The ID of the fine-tuning job to retrieve.

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `fine_tuning_job_id` | 路径参数 | ✅ |  |
| `custom_llm_provider` | 查询参数 | ❌ |  |

#### `GET` /v1/fine_tuning/jobs/{fine_tuning_job_id}

**✨ (Enterprise) Retrieve Fine-Tuning Job**

获取微调作业详情。
等效于 `GET https://api.openai.com/v1/fine_tuning/jobs/{fine_tuning_job_id}`。

支持的查询参数：
- `custom_llm_provider`: Name of the LiteLLM provider
- `fine_tuning_job_id`: The ID of the fine-tuning job to retrieve.

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `fine_tuning_job_id` | 路径参数 | ✅ |  |
| `custom_llm_provider` | 查询参数 | ❌ |  |

#### `POST` /fine_tuning/jobs/{fine_tuning_job_id}/cancel

**✨ (Enterprise) Cancel Fine-Tuning Jobs**

取消微调作业。

等效于 `POST https://api.openai.com/v1/fine_tuning/jobs/{fine_tuning_job_id}/cancel`。

支持的查询参数：
- `custom_llm_provider`: Name of the LiteLLM provider
- `fine_tuning_job_id`: The ID of the fine-tuning job to cancel.

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `fine_tuning_job_id` | 路径参数 | ✅ |  |

#### `POST` /v1/fine_tuning/jobs/{fine_tuning_job_id}/cancel

**✨ (Enterprise) Cancel Fine-Tuning Jobs**

取消微调作业。

等效于 `POST https://api.openai.com/v1/fine_tuning/jobs/{fine_tuning_job_id}/cancel`。

支持的查询参数：
- `custom_llm_provider`: Name of the LiteLLM provider
- `fine_tuning_job_id`: The ID of the fine-tuning job to cancel.

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `fine_tuning_job_id` | 路径参数 | ✅ |  |

### 向量存储管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/vector_store/new` | 新建 Vector Store |
| `GET` | `/v1/vector_store/list` | 向量存储列表 |
| `GET` | `/vector_store/list` | 向量存储列表 |
| `POST` | `/vector_store/delete` | 删除向量存储 |
| `POST` | `/vector_store/info` | 获取 Vector Store Info |
| `POST` | `/vector_store/update` | 更新向量存储 |

#### `POST` /vector_store/new

**新建 Vector Store**

创建新的 Vector Store。

参数：
- vector_store_id: str - Unique identifier for the vector store
- custom_llm_provider: str - Provider of the vector store
- vector_store_name: Optional[str] - Name of the vector store
- vector_store_description: Optional[str] - Description of the vector store
- vector_store_metadata: Optional[Dict] - Additional metadata for the vector store

#### `GET` /v1/vector_store/list

**向量存储列表**

列出全部可用的 Vector Store，支持可选的过滤与分页。
同时包含内存中的 Vector Store 与数据库中存储的 Vector Store。
以数据库为准 —— 删除的 Vector Store 会同步从内存中移除，更新的 Vector Store 会同步到内存。

参数：
- page: int - Page number for pagination (default: 1)
- page_size: int - Number of items per page (default: 100)

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `page` | 查询参数 | ❌ |  |
| `page_size` | 查询参数 | ❌ |  |

#### `GET` /vector_store/list

**向量存储列表**

列出全部可用的 Vector Store，支持可选的过滤与分页。
同时包含内存中的 Vector Store 与数据库中存储的 Vector Store。
以数据库为准 —— 删除的 Vector Store 会同步从内存中移除，更新的 Vector Store 会同步到内存。

参数：
- page: int - Page number for pagination (default: 1)
- page_size: int - Number of items per page (default: 100)

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `page` | 查询参数 | ❌ |  |
| `page_size` | 查询参数 | ❌ |  |

#### `POST` /vector_store/delete

**删除向量存储**

同时从数据库和内存注册表中删除 Vector Store。

参数：
- vector_store_id: str - ID of the vector store to delete

#### `POST` /vector_store/info

**获取 Vector Store Info**

返回单个 Vector Store 的详情。

#### `POST` /vector_store/update

**更新向量存储**

同时在数据库和内存注册表中更新 Vector Store 信息。
更新后的数据会立即同步到内存注册表。

### 向量存储文件

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/vector_stores/{vector_store_id}/files` | Vector Store File Create |
| `GET` | `/vector_stores/{vector_store_id}/files` | Vector Store File List |
| `POST` | `/v1/vector_stores/{vector_store_id}/files` | Vector Store File Create |
| `GET` | `/v1/vector_stores/{vector_store_id}/files` | Vector Store File List |
| `GET` | `/vector_stores/{vector_store_id}/files/{file_id}` | Vector Store File Retrieve |
| `POST` | `/vector_stores/{vector_store_id}/files/{file_id}` | Vector Store File Update |
| `DELETE` | `/vector_stores/{vector_store_id}/files/{file_id}` | Vector Store File Delete |
| `GET` | `/v1/vector_stores/{vector_store_id}/files/{file_id}` | Vector Store File Retrieve |
| `POST` | `/v1/vector_stores/{vector_store_id}/files/{file_id}` | Vector Store File Update |
| `DELETE` | `/v1/vector_stores/{vector_store_id}/files/{file_id}` | Vector Store File Delete |
| `GET` | `/vector_stores/{vector_store_id}/files/{file_id}/content` | Vector Store File Content |
| `GET` | `/v1/vector_stores/{vector_store_id}/files/{file_id}/content` | Vector Store File Content |

#### `POST` /vector_stores/{vector_store_id}/files

**创建 Vector Store 文件**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `vector_store_id` | 路径参数 | ✅ |  |

#### `GET` /vector_stores/{vector_store_id}/files

**Vector Store 文件列表**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `vector_store_id` | 路径参数 | ✅ |  |

#### `POST` /v1/vector_stores/{vector_store_id}/files

**创建 Vector Store 文件**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `vector_store_id` | 路径参数 | ✅ |  |

#### `GET` /v1/vector_stores/{vector_store_id}/files

**Vector Store 文件列表**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `vector_store_id` | 路径参数 | ✅ |  |

#### `GET` /vector_stores/{vector_store_id}/files/{file_id}

**获取 Vector Store 文件**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `vector_store_id` | 路径参数 | ✅ |  |
| `file_id` | 路径参数 | ✅ |  |

#### `POST` /vector_stores/{vector_store_id}/files/{file_id}

**更新 Vector Store 文件**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `vector_store_id` | 路径参数 | ✅ |  |
| `file_id` | 路径参数 | ✅ |  |

#### `DELETE` /vector_stores/{vector_store_id}/files/{file_id}

**删除 Vector Store 文件**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `vector_store_id` | 路径参数 | ✅ |  |
| `file_id` | 路径参数 | ✅ |  |

#### `GET` /v1/vector_stores/{vector_store_id}/files/{file_id}

**获取 Vector Store 文件**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `vector_store_id` | 路径参数 | ✅ |  |
| `file_id` | 路径参数 | ✅ |  |

#### `POST` /v1/vector_stores/{vector_store_id}/files/{file_id}

**更新 Vector Store 文件**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `vector_store_id` | 路径参数 | ✅ |  |
| `file_id` | 路径参数 | ✅ |  |

#### `DELETE` /v1/vector_stores/{vector_store_id}/files/{file_id}

**删除 Vector Store 文件**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `vector_store_id` | 路径参数 | ✅ |  |
| `file_id` | 路径参数 | ✅ |  |

#### `GET` /vector_stores/{vector_store_id}/files/{file_id}/content

**Vector Store 文件内容**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `vector_store_id` | 路径参数 | ✅ |  |
| `file_id` | 路径参数 | ✅ |  |

#### `GET` /v1/vector_stores/{vector_store_id}/files/{file_id}/content

**Vector Store 文件内容**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `vector_store_id` | 路径参数 | ✅ |  |
| `file_id` | 路径参数 | ✅ |  |

### RAG 检索增强生成

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/rag/ingest` | Rag Ingest |
| `POST` | `/v1/rag/ingest` | Rag Ingest |
| `POST` | `/rag/query` | Rag Query |
| `POST` | `/v1/rag/query` | Rag Query |

#### `POST` /rag/ingest

**RAG 摄取**

RAG 摄取端点 —— 一体化的文档摄取流水线。

支持表单上传（文件）或 JSON 请求体（URL）。

## Form upload (for files):
```bash
curl -X POST "http://localhost:4000/v1/rag/ingest" \
    -H "Authorization: Bearer sk-1234" \
    -F file="@document.pdf" \
    -F 'ingest_options={"vector_store": {"custom_llm_provider": "openai"}}'
```

## JSON body (for URLs):
```bash
curl -X POST "http://localhost:4000/v1/rag/ingest" \
    -H "Authorization: Bearer sk-1234" \
    -H "Co...
```

#### `POST` /v1/rag/ingest

**RAG 摄取**

RAG 摄取端点 —— 一体化的文档摄取流水线。

支持表单上传（文件）或 JSON 请求体（URL）。

## Form upload (for files):
```bash
curl -X POST "http://localhost:4000/v1/rag/ingest" \
    -H "Authorization: Bearer sk-1234" \
    -F file="@document.pdf" \
    -F 'ingest_options={"vector_store": {"custom_llm_provider": "openai"}}'
```

## JSON body (for URLs):
```bash
curl -X POST "http://localhost:4000/v1/rag/ingest" \
    -H "Authorization: Bearer sk-1234" \
    -H "Co...
```

#### `POST` /rag/query

**RAG 查询**

RAG 查询端点 —— 检索 Vector Store，可选重排序，并生成 LLM 响应。

该端点：
1. Extracts the query from the last user message
2. Searches the vector store for relevant context
3. Optionally reranks the results
4. Generates an LLM response with the retrieved context

## 请求示例：
```bash
curl -X POST "http://localhost:4000/v1/rag/query" \
    -H "Authorization: Bearer sk-1234" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gpt-4o-mini",
 ...
```

#### `POST` /v1/rag/query

**RAG 查询**

RAG 查询端点 —— 检索 Vector Store，可选重排序，并生成 LLM 响应。

该端点：
1. Extracts the query from the last user message
2. Searches the vector store for relevant context
3. Optionally reranks the results
4. Generates an LLM response with the retrieved context

## 请求示例：
```bash
curl -X POST "http://localhost:4000/v1/rag/query" \
    -H "Authorization: Bearer sk-1234" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gpt-4o-mini",
 ...
```

### 容器

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/containers` | 列表 Containers |
| `POST` | `/containers` | 创建 Container |
| `GET` | `/v1/containers` | 列表 Containers |
| `POST` | `/v1/containers` | 创建 Container |
| `GET` | `/containers/{container_id}` | Retrieve Container |
| `DELETE` | `/containers/{container_id}` | 删除 Container |
| `GET` | `/v1/containers/{container_id}` | Retrieve Container |
| `DELETE` | `/v1/containers/{container_id}` | 删除 Container |
| `GET` | `/v1/containers/{container_id}/files` | Handler Container Id |
| `POST` | `/v1/containers/{container_id}/files` | Handler Multipart Upload |
| `GET` | `/containers/{container_id}/files` | Handler Container Id |
| `POST` | `/containers/{container_id}/files` | Handler Multipart Upload |
| `GET` | `/v1/containers/{container_id}/files/{file_id}` | Handler Container File |
| `DELETE` | `/v1/containers/{container_id}/files/{file_id}` | Handler Container File |
| `GET` | `/containers/{container_id}/files/{file_id}` | Handler Container File |
| `DELETE` | `/containers/{container_id}/files/{file_id}` | Handler Container File |
| `GET` | `/v1/containers/{container_id}/files/{file_id}/content` | Handler Binary Content |
| `GET` | `/containers/{container_id}/files/{file_id}/content` | Handler Binary Content |

#### `GET` /containers

**列表 Containers**

容器列表端点，用于获取容器列表。

与 OpenAI Containers API 规范一致：
https://platform.openai.com/docs/api-reference/containers

示例：
```bash
curl -X GET "http://localhost:4000/v1/containers?limit=20&order=desc"         -H "Authorization: Bearer sk-1234"
```

或通过请求头、查询参数指定提供商：
```bash
curl -X GET "http://localhost:4000/v1/containers?custom_llm_provider=azure"         -H "Authorization: Bearer sk-1234"
```

#### `POST` /containers

**创建 Container**

容器创建端点，用于创建新容器。

与 OpenAI Containers API 规范一致：
https://platform.openai.com/docs/api-reference/containers

示例：
```bash
curl -X POST "http://localhost:4000/v1/containers"         -H "Authorization: Bearer sk-1234"         -H "Content-Type: application/json"         -d '{
        "name": "My Container",
        "expires_after": {
            "anchor": "last_active_at",
            "minutes": 20
        }
    }'
```

或通过请求头指定提供商……

#### `GET` /v1/containers

**列表 Containers**

容器列表端点，用于获取容器列表。

与 OpenAI Containers API 规范一致：
https://platform.openai.com/docs/api-reference/containers

示例：
```bash
curl -X GET "http://localhost:4000/v1/containers?limit=20&order=desc"         -H "Authorization: Bearer sk-1234"
```

或通过请求头、查询参数指定提供商：
```bash
curl -X GET "http://localhost:4000/v1/containers?custom_llm_provider=azure"         -H "Authorization: Bearer sk-1234"
```

#### `POST` /v1/containers

**创建 Container**

容器创建端点，用于创建新容器。

与 OpenAI Containers API 规范一致：
https://platform.openai.com/docs/api-reference/containers

示例：
```bash
curl -X POST "http://localhost:4000/v1/containers"         -H "Authorization: Bearer sk-1234"         -H "Content-Type: application/json"         -d '{
        "name": "My Container",
        "expires_after": {
            "anchor": "last_active_at",
            "minutes": 20
        }
    }'
```

或通过请求头指定提供商……

#### `GET` /containers/{container_id}

**获取容器**

容器详情端点，用于获取指定容器的详细信息。

与 OpenAI Containers API 规范一致：
https://platform.openai.com/docs/api-reference/containers

示例：
```bash
curl -X GET "http://localhost:4000/v1/containers/cntr_123"         -H "Authorization: Bearer sk-1234"
```

或通过请求头指定提供商：
```bash
curl -X GET "http://localhost:4000/v1/containers/cntr_123"         -H "Authorization: Bearer sk-1234"         -H "custom-llm-provider: azure"
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `container_id` | 路径参数 | ✅ |  |

#### `DELETE` /containers/{container_id}

**删除 Container**

容器删除端点，用于删除指定容器。

与 OpenAI Containers API 规范一致：
https://platform.openai.com/docs/api-reference/containers

示例：
```bash
curl -X DELETE "http://localhost:4000/v1/containers/cntr_123"         -H "Authorization: Bearer sk-1234"
```

或通过请求头指定提供商：
```bash
curl -X DELETE "http://localhost:4000/v1/containers/cntr_123"         -H "Authorization: Bearer sk-1234"         -H "custom-llm-provider: azure"
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `container_id` | 路径参数 | ✅ |  |

#### `GET` /v1/containers/{container_id}

**获取容器**

容器详情端点，用于获取指定容器的详细信息。

与 OpenAI Containers API 规范一致：
https://platform.openai.com/docs/api-reference/containers

示例：
```bash
curl -X GET "http://localhost:4000/v1/containers/cntr_123"         -H "Authorization: Bearer sk-1234"
```

或通过请求头指定提供商：
```bash
curl -X GET "http://localhost:4000/v1/containers/cntr_123"         -H "Authorization: Bearer sk-1234"         -H "custom-llm-provider: azure"
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `container_id` | 路径参数 | ✅ |  |

#### `DELETE` /v1/containers/{container_id}

**删除 Container**

容器删除端点，用于删除指定容器。

与 OpenAI Containers API 规范一致：
https://platform.openai.com/docs/api-reference/containers

示例：
```bash
curl -X DELETE "http://localhost:4000/v1/containers/cntr_123"         -H "Authorization: Bearer sk-1234"
```

或通过请求头指定提供商：
```bash
curl -X DELETE "http://localhost:4000/v1/containers/cntr_123"         -H "Authorization: Bearer sk-1234"         -H "custom-llm-provider: azure"
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `container_id` | 路径参数 | ✅ |  |

#### `GET` /v1/containers/{container_id}/files

**按容器 ID 处理**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `container_id` | 路径参数 | ✅ |  |

#### `POST` /v1/containers/{container_id}/files

**处理 Multipart 上传**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `container_id` | 路径参数 | ✅ |  |

#### `GET` /containers/{container_id}/files

**按容器 ID 处理**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `container_id` | 路径参数 | ✅ |  |

#### `POST` /containers/{container_id}/files

**处理 Multipart 上传**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `container_id` | 路径参数 | ✅ |  |

#### `GET` /v1/containers/{container_id}/files/{file_id}

**处理容器文件**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `container_id` | 路径参数 | ✅ |  |
| `file_id` | 路径参数 | ✅ |  |

#### `DELETE` /v1/containers/{container_id}/files/{file_id}

**处理容器文件**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `container_id` | 路径参数 | ✅ |  |
| `file_id` | 路径参数 | ✅ |  |

#### `GET` /containers/{container_id}/files/{file_id}

**处理容器文件**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `container_id` | 路径参数 | ✅ |  |
| `file_id` | 路径参数 | ✅ |  |

#### `DELETE` /containers/{container_id}/files/{file_id}

**处理容器文件**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `container_id` | 路径参数 | ✅ |  |
| `file_id` | 路径参数 | ✅ |  |

#### `GET` /v1/containers/{container_id}/files/{file_id}/content

**处理二进制内容**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `container_id` | 路径参数 | ✅ |  |
| `file_id` | 路径参数 | ✅ |  |

#### `GET` /containers/{container_id}/files/{file_id}/content

**处理二进制内容**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `container_id` | 路径参数 | ✅ |  |
| `file_id` | 路径参数 | ✅ |  |

