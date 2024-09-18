import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# [BETA] Batches API

Covers Batches, Files

## **Supported Providers**:
- Azure OpenAI
- OpenAI

## Quick Start 

- Create File for Batch Completion

- Create Batch Request

- List Batches

- Retrieve the Specific Batch and File Content


<Tabs>
<TabItem value="proxy" label="LiteLLM PROXY Server">

```bash
$ export OPENAI_API_KEY="sk-..."

$ litellm

# RUNNING on http://0.0.0.0:4000
```

**Create File for Batch Completion**

```shell
curl http://localhost:4000/v1/files \
    -H "Authorization: Bearer sk-1234" \
    -F purpose="batch" \
    -F file="@mydata.jsonl"
```

**Create Batch Request**

```bash
curl http://localhost:4000/v1/batches \
        -H "Authorization: Bearer sk-1234" \
        -H "Content-Type: application/json" \
        -d '{
            "input_file_id": "file-abc123",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h"
    }'
```

**Retrieve the Specific Batch**

```bash
curl http://localhost:4000/v1/batches/batch_abc123 \
    -H "Authorization: Bearer sk-1234" \
    -H "Content-Type: application/json" \
```


**List Batches**

```bash
curl http://localhost:4000/v1/batches \
    -H "Authorization: Bearer sk-1234" \
    -H "Content-Type: application/json" \
```

</TabItem>
<TabItem value="sdk" label="SDK">

**Create File for Batch Completion**

```python
from litellm
import os 

os.environ["OPENAI_API_KEY"] = "sk-.."

file_name = "openai_batch_completions.jsonl"
_current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(_current_dir, file_name)
file_obj = await litellm.acreate_file(
    file=open(file_path, "rb"),
    purpose="batch",
    custom_llm_provider="openai",
)
print("Response from creating file=", file_obj)
```

**Create Batch Request**

```python
from litellm
import os 

create_batch_response = await litellm.acreate_batch(
    completion_window="24h",
    endpoint="/v1/chat/completions",
    input_file_id=batch_input_file_id,
    custom_llm_provider="openai",
    metadata={"key1": "value1", "key2": "value2"},
)

print("response from litellm.create_batch=", create_batch_response)
```

**Retrieve the Specific Batch and File Content**

```python

retrieved_batch = await litellm.aretrieve_batch(
    batch_id=create_batch_response.id, custom_llm_provider="openai"
)
print("retrieved batch=", retrieved_batch)
# just assert that we retrieved a non None batch

assert retrieved_batch.id == create_batch_response.id

# try to get file content for our original file

file_content = await litellm.afile_content(
    file_id=batch_input_file_id, custom_llm_provider="openai"
)

print("file content = ", file_content)
```

**List Batches**

```python
list_batches_response = litellm.list_batches(custom_llm_provider="openai", limit=2)
print("list_batches_response=", list_batches_response)
```

</TabItem>

</Tabs>

## [👉 Proxy API Reference](https://litellm-api.up.railway.app/#/batch)

## Azure Batches API 

Just add the azure env vars to your environment. 

```bash
export AZURE_API_KEY=""
export AZURE_API_BASE=""
```

AND use `/azure/*` for the Batches API calls

```bash
http://0.0.0.0:4000/azure/v1/batches
```
### Usage

**Setup**

- Add Azure API Keys to your environment

#### 1. Upload a File

```bash
curl http://localhost:4000/azure/v1/files \
    -H "Authorization: Bearer sk-1234" \
    -F purpose="batch" \
    -F file="@mydata.jsonl"
```

**Example File**

Note: `model` should be your azure deployment name.

```json
{"custom_id": "task-0", "method": "POST", "url": "/chat/completions", "body": {"model": "REPLACE-WITH-MODEL-DEPLOYMENT-NAME", "messages": [{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": "When was Microsoft founded?"}]}}
{"custom_id": "task-1", "method": "POST", "url": "/chat/completions", "body": {"model": "REPLACE-WITH-MODEL-DEPLOYMENT-NAME", "messages": [{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": "When was the first XBOX released?"}]}}
{"custom_id": "task-2", "method": "POST", "url": "/chat/completions", "body": {"model": "REPLACE-WITH-MODEL-DEPLOYMENT-NAME", "messages": [{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": "What is Altair Basic?"}]}}
```

#### 2. Create a batch 

```bash
curl http://0.0.0.0:4000/azure/v1/batches \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input_file_id": "file-abc123",
    "endpoint": "/v1/chat/completions",
    "completion_window": "24h"
  }'

```

#### 3. Retrieve batch


```bash
curl http://0.0.0.0:4000/azure/v1/batches/batch_abc123 \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
```

#### 4. Cancel batch 

```bash
curl http://0.0.0.0:4000/azure/v1/batches/batch_abc123/cancel \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
  -X POST
```

#### 5. List Batch

```bash
curl http://0.0.0.0:4000/v1/batches?limit=2 \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json"
```

### [👉 Health Check Azure Batch models](./proxy/health.md#batch-models-azure-only)


### [BETA] Loadbalance Multiple Azure Deployments 
In your config.yaml, set `enable_loadbalancing_on_batch_endpoints: true`

```yaml
model_list:
  - model_name: "batch-gpt-4o-mini"
    litellm_params:
      model: "azure/gpt-4o-mini"
      api_key: os.environ/AZURE_API_KEY
      api_base: os.environ/AZURE_API_BASE
    model_info:
      mode: batch

litellm_settings:
  enable_loadbalancing_on_batch_endpoints: true # 👈 KEY CHANGE
```

Note: This works on `{PROXY_BASE_URL}/v1/files` and `{PROXY_BASE_URL}/v1/batches`.
Note: Response is in the OpenAI-format. 

1. Upload a file 

Just set `model: batch-gpt-4o-mini` in your .jsonl.

```bash
curl http://localhost:4000/v1/files \
    -H "Authorization: Bearer sk-1234" \
    -F purpose="batch" \
    -F file="@mydata.jsonl"
```

**Example File**

Note: `model` should be your azure deployment name.

```json
{"custom_id": "task-0", "method": "POST", "url": "/chat/completions", "body": {"model": "batch-gpt-4o-mini", "messages": [{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": "When was Microsoft founded?"}]}}
{"custom_id": "task-1", "method": "POST", "url": "/chat/completions", "body": {"model": "batch-gpt-4o-mini", "messages": [{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": "When was the first XBOX released?"}]}}
{"custom_id": "task-2", "method": "POST", "url": "/chat/completions", "body": {"model": "batch-gpt-4o-mini", "messages": [{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": "What is Altair Basic?"}]}}
```

Expected Response (OpenAI-compatible)

```bash
{"id":"file-f0be81f654454113a922da60acb0eea6",...}
```

2. Create a batch 

```bash
curl http://0.0.0.0:4000/v1/batches \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input_file_id": "file-f0be81f654454113a922da60acb0eea6",
    "endpoint": "/v1/chat/completions",
    "completion_window": "24h",
    "model: "batch-gpt-4o-mini"
  }'
```

Expected Response: 

```bash
{"id":"batch_94e43f0a-d805-477d-adf9-bbb9c50910ed",...}
```

3. Retrieve a batch 

```bash
curl http://0.0.0.0:4000/v1/batches/batch_94e43f0a-d805-477d-adf9-bbb9c50910ed \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
```


Expected Response: 

```
{"id":"batch_94e43f0a-d805-477d-adf9-bbb9c50910ed",...}
```

4. List batch

```bash
curl http://0.0.0.0:4000/v1/batches?limit=2 \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json"
```

Expected Response:

```bash
{"data":[{"id":"batch_R3V...}
```