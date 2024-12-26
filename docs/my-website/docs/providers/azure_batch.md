
# Batches API

| Property | Details |
|-------|-------|
| Description | Azure OpenAI Batches API |
| `custom_llm_provider` on LiteLLM | `azure/` |
| Supported Operations | `/v1/batches`, `/v1/files` |
| Azure OpenAI Batches API | [Azure OpenAI Batches API ↗](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/batch) |
| Cost Tracking, Logging Support | ✅ LiteLLM will log, track cost for Batch API Requests |


### Quick Start

Just add the azure env vars to your environment. 

```bash
export AZURE_API_KEY=""
export AZURE_API_BASE=""
```

<Tabs>
<TabItem value="proxy" label="LiteLLM PROXY Server">

**1. Upload a File**

<Tabs>
<TabItem value="sdk" label="OpenAI Python SDK">

```python
from openai import OpenAI

# Initialize the client
client = OpenAI(
    base_url="http://localhost:4000",
    api_key="your-api-key"
)

batch_input_file = client.files.create(
    file=open("mydata.jsonl", "rb"),
    purpose="batch",
    extra_body={"custom_llm_provider": "azure"}
)
file_id = batch_input_file.id
```

</TabItem>
<TabItem value="curl" label="Curl">

```bash
curl http://localhost:4000/v1/files \
    -H "Authorization: Bearer sk-1234" \
    -F purpose="batch" \
    -F file="@mydata.jsonl"
```

</TabItem>
</Tabs>

**Example File Format**
```json
{"custom_id": "task-0", "method": "POST", "url": "/chat/completions", "body": {"model": "REPLACE-WITH-MODEL-DEPLOYMENT-NAME", "messages": [{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": "When was Microsoft founded?"}]}}
{"custom_id": "task-1", "method": "POST", "url": "/chat/completions", "body": {"model": "REPLACE-WITH-MODEL-DEPLOYMENT-NAME", "messages": [{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": "When was the first XBOX released?"}]}}
{"custom_id": "task-2", "method": "POST", "url": "/chat/completions", "body": {"model": "REPLACE-WITH-MODEL-DEPLOYMENT-NAME", "messages": [{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": "What is Altair Basic?"}]}}
```

**2. Create a Batch Request**

<Tabs>
<TabItem value="sdk" label="OpenAI Python SDK">

```python
batch = client.batches.create( # re use client from above
    input_file_id=file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={"description": "My batch job"},
    extra_body={"custom_llm_provider": "azure"}
)
```

</TabItem>
<TabItem value="curl" label="Curl">

```bash
curl http://localhost:4000/v1/batches \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input_file_id": "file-abc123",
    "endpoint": "/v1/chat/completions",
    "completion_window": "24h"
  }'
```
</TabItem>
</Tabs>

**3. Retrieve a Batch**

<Tabs>
<TabItem value="sdk" label="OpenAI Python SDK">

```python
retrieved_batch = client.batches.retrieve(
    batch.id,
    extra_body={"custom_llm_provider": "azure"}
)
```

</TabItem>
<TabItem value="curl" label="Curl">

```bash
curl http://localhost:4000/v1/batches/batch_abc123 \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
```

</TabItem>
</Tabs>

**4. Cancel a Batch**

<Tabs>
<TabItem value="sdk" label="OpenAI Python SDK">

```python
cancelled_batch = client.batches.cancel(
    batch.id,
    extra_body={"custom_llm_provider": "azure"}
)
```

</TabItem>
<TabItem value="curl" label="Curl">

```bash
curl http://localhost:4000/v1/batches/batch_abc123/cancel \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
  -X POST
```

</TabItem>
</Tabs>

**5. List Batches**

<Tabs>
<TabItem value="sdk" label="OpenAI Python SDK">

```python
client.batches.list(extra_body={"custom_llm_provider": "azure"})
```

</TabItem>
<TabItem value="curl" label="Curl">

```bash
curl http://localhost:4000/v1/batches?limit=2 \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json"
```
</TabItem>
</Tabs>
</TabItem>
<TabItem value="sdk" label="LiteLLM SDK">

**1. Create File for Batch Completion**

```python
from litellm
import os 

os.environ["AZURE_API_KEY"] = ""
os.environ["AZURE_API_BASE"] = ""

file_name = "azure_batch_completions.jsonl"
_current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(_current_dir, file_name)
file_obj = await litellm.acreate_file(
    file=open(file_path, "rb"),
    purpose="batch",
    custom_llm_provider="azure",
)
print("Response from creating file=", file_obj)
```

**2. Create Batch Request**

```python
create_batch_response = await litellm.acreate_batch(
    completion_window="24h",
    endpoint="/v1/chat/completions",
    input_file_id=batch_input_file_id,
    custom_llm_provider="azure",
    metadata={"key1": "value1", "key2": "value2"},
)

print("response from litellm.create_batch=", create_batch_response)
```

**3. Retrieve Batch and File Content**

```python
retrieved_batch = await litellm.aretrieve_batch(
    batch_id=create_batch_response.id, 
    custom_llm_provider="azure"
)
print("retrieved batch=", retrieved_batch)

# Get file content
file_content = await litellm.afile_content(
    file_id=batch_input_file_id, 
    custom_llm_provider="azure"
)
print("file content = ", file_content)
```

**4. List Batches**

```python
list_batches_response = litellm.list_batches(
    custom_llm_provider="azure", 
    limit=2
)
print("list_batches_response=", list_batches_response)
```

</TabItem>
</Tabs>

### [Health Check Azure Batch models](./proxy/health.md#batch-models-azure-only)


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