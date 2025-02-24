import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# [BETA] Batches API

Covers Batches, Files

| Feature | Supported | Notes | 
|-------|-------|-------|
| Supported Providers | OpenAI, Azure, Vertex | - |
| ✨ Cost Tracking | ✅ | LiteLLM Enterprise only |
| Logging | ✅ | Works across all logging integrations |

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


## **Supported Providers**:
### [Azure OpenAI](./providers/azure#azure-batches-api)
### [OpenAI](#quick-start)
### [Vertex AI](./providers/vertex#batch-apis)


## How Cost Tracking for Batches API Works

LiteLLM tracks batch processing costs by logging two key events:

| Event Type | Description | When it's Logged |
|------------|-------------|------------------|
| `acreate_batch` | Initial batch creation | When batch request is submitted |
| `batch_success` | Final usage and cost | When batch processing completes |

Cost calculation:

- LiteLLM polls the batch status until completion
- Upon completion, it aggregates usage and costs from all responses in the output file
- Total `token` and `response_cost` reflect the combined metrics across all batch responses


## Batches API with Self-Hosted Models

To use the batches API with self-hosted models, you'll need to:

1. Configure a storage location for batch files (S3, GCS, etc.)
2. Point to your self-hosted model endpoint


### Step 1: Configure Storage in config.yaml
First, set up where you want to store the batch files. You can use S3, GCS, or Azure Blob Storage

```yaml
model_list:
  - model_name: vllm-model
    litellm_params:
      model: openai/facebook/opt-125m # the `openai/` prefix tells litellm it's openai compatible
      api_base: http://0.0.0.0:4000/v1
      api_key: none

batch_settings:
  # Configure S3 for batch file storage
  model: vllm-model
  batch_storage_params:
    s3_bucket_name: my-batch-bucket   # AWS Bucket Name for S3
    s3_region_name: us-west-2         # AWS Region Name for S3
    s3_aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID           # AWS Access Key ID for S3
    s3_aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY   # AWS Secret Access Key for S3
```

### Step 2: Start the proxy

```bash
litellm --config config.yaml
```

### Step 3: Create a Batch Request


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


## [Swagger API Reference](https://litellm-api.up.railway.app/#/batch)
