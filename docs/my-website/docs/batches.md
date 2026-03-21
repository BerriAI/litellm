import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# /batches

Covers Batches, Files

| Feature | Supported | Notes | 
|-------|-------|-------|
| Supported Providers | OpenAI, Azure, Vertex, Bedrock, vLLM | - |
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
import litellm
import os 
import asyncio

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
import litellm
import os 
import asyncio

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
    # Maximum wait time before we give up
    MAX_WAIT_TIME = 300  

    # Time to wait between each status check
    POLL_INTERVAL = 5
    
    #Time waited till now 
    waited = 0

    # Wait for the batch to finish processing before trying to retrieve output
    # This loop checks the batch status every few seconds (polling)

    while True:
        retrieved_batch = await litellm.aretrieve_batch(
            batch_id=create_batch_response.id,
            custom_llm_provider="openai"
        )
        
        status = retrieved_batch.status
        print(f"⏳ Batch status: {status}")
        
        if status == "completed" and retrieved_batch.output_file_id:
            print("✅ Batch complete. Output file ID:", retrieved_batch.output_file_id)
            break
        elif status in ["failed", "cancelled", "expired"]:
            raise RuntimeError(f"❌ Batch failed with status: {status}")
        
        await asyncio.sleep(POLL_INTERVAL)
        waited += POLL_INTERVAL
        if waited > MAX_WAIT_TIME:
            raise TimeoutError("❌ Timed out waiting for batch to complete.")

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


## Multi-Account / Model-Based Routing

Route batch operations to different provider accounts using model-specific credentials from your `config.yaml`. This eliminates the need for environment variables and enables multi-tenant batch processing.

### How It Works

**Priority Order:**
1. **Encoded Batch/File ID** (highest) - Model info embedded in the ID
2. **Model Parameter** - Via header (`x-litellm-model`), query param, or request body
3. **Custom Provider** (fallback) - Uses environment variables

### Configuration

```yaml
model_list:
  - model_name: gpt-4o-account-1
    litellm_params:
      model: openai/gpt-4o
      api_key: sk-account-1-key
      api_base: https://api.openai.com/v1
  
  - model_name: gpt-4o-account-2
    litellm_params:
      model: openai/gpt-4o
      api_key: sk-account-2-key
      api_base: https://api.openai.com/v1
  
  - model_name: azure-batches
    litellm_params:
      model: azure/gpt-4
      api_key: azure-key-123
      api_base: https://my-resource.openai.azure.com
      api_version: "2024-02-01"
```

### Usage Examples

#### Scenario 1: Encoded File ID with Model

When you upload a file with a model parameter, LiteLLM encodes the model information in the file ID. All subsequent operations automatically use those credentials.

```bash
# Step 1: Upload file with model
curl http://localhost:4000/v1/files \
  -H "Authorization: Bearer sk-1234" \
  -H "x-litellm-model: gpt-4o-account-1" \
  -F purpose="batch" \
  -F file="@batch.jsonl"

# Response includes encoded file ID:
# {
#   "id": "file-bGl0ZWxsbTpmaWxlLUxkaUwzaVYxNGZRVlpYcU5KVEdkSjk7bW9kZWwsZ3B0LTRvLWFjY291bnQtMQ",
#   ...
# }

# Step 2: Create batch - automatically routes to gpt-4o-account-1
curl http://localhost:4000/v1/batches \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "input_file_id": "file-bGl0ZWxsbTpmaWxlLUxkaUwzaVYxNGZRVlpYcU5KVEdkSjk7bW9kZWwsZ3B0LTRvLWFjY291bnQtMQ",
    "endpoint": "/v1/chat/completions",
    "completion_window": "24h"
  }'

# Batch ID is also encoded with model:
# {
#   "id": "batch_bGl0ZWxsbTpiYXRjaF82OTIwM2IzNjg0MDQ4MTkwYTA3ODQ5NDY3YTFjMDJkYTttb2RlbCxncHQtNG8tYWNjb3VudC0x",
#   "input_file_id": "file-bGl0ZWxsbTpmaWxlLUxkaUwzaVYxNGZRVlpYcU5KVEdkSjk7bW9kZWwsZ3B0LTRvLWFjY291bnQtMQ",
#   ...
# }

# Step 3: Retrieve batch - automatically routes to gpt-4o-account-1
curl http://localhost:4000/v1/batches/batch_bGl0ZWxsbTpiYXRjaF82OTIwM2IzNjg0MDQ4MTkwYTA3ODQ5NDY3YTFjMDJkYTttb2RlbCxncHQtNG8tYWNjb3VudC0x \
  -H "Authorization: Bearer sk-1234"
```

**✅ Benefits:**
- No need to specify model on every request
- File and batch IDs "remember" which account created them
- Automatic routing for retrieve, cancel, and file content operations

#### Scenario 2: Model via Header/Query Parameter

Specify the model for each request without encoding it in the ID.

```bash
# Create batch with model header
curl http://localhost:4000/v1/batches \
  -H "Authorization: Bearer sk-1234" \
  -H "x-litellm-model: gpt-4o-account-2" \
  -H "Content-Type: application/json" \
  -d '{
    "input_file_id": "file-abc123",
    "endpoint": "/v1/chat/completions",
    "completion_window": "24h"
  }'

# Or use query parameter
curl "http://localhost:4000/v1/batches?model=gpt-4o-account-2" \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "input_file_id": "file-abc123",
    "endpoint": "/v1/chat/completions",
    "completion_window": "24h"
  }'

# List batches for specific model
curl "http://localhost:4000/v1/batches?model=gpt-4o-account-2" \
  -H "Authorization: Bearer sk-1234"
```

**✅ Use Case:**
- One-off batch operations
- Different models for different operations
- Explicit control over routing

#### Scenario 3: Environment Variables (Fallback)

Traditional approach using environment variables when no model is specified.

```bash
export OPENAI_API_KEY="sk-env-key"

curl http://localhost:4000/v1/batches \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "input_file_id": "file-abc123",
    "endpoint": "/v1/chat/completions",
    "completion_window": "24h"
  }'
```

**✅ Use Case:**
- Backward compatibility
- Simple single-account setups
- Quick prototyping

### Complete Multi-Account Example

```bash
# Upload file to Account 1
FILE_1=$(curl -s http://localhost:4000/v1/files \
  -H "x-litellm-model: gpt-4o-account-1" \
  -F purpose="batch" \
  -F file="@batch1.jsonl" | jq -r '.id')

# Upload file to Account 2
FILE_2=$(curl -s http://localhost:4000/v1/files \
  -H "x-litellm-model: gpt-4o-account-2" \
  -F purpose="batch" \
  -F file="@batch2.jsonl" | jq -r '.id')

# Create batch on Account 1 (auto-routed via encoded file ID)
BATCH_1=$(curl -s http://localhost:4000/v1/batches \
  -d "{\"input_file_id\": \"$FILE_1\", \"endpoint\": \"/v1/chat/completions\", \"completion_window\": \"24h\"}" | jq -r '.id')

# Create batch on Account 2 (auto-routed via encoded file ID)
BATCH_2=$(curl -s http://localhost:4000/v1/batches \
  -d "{\"input_file_id\": \"$FILE_2\", \"endpoint\": \"/v1/chat/completions\", \"completion_window\": \"24h\"}" | jq -r '.id')

# Retrieve both batches (auto-routed to correct accounts)
curl http://localhost:4000/v1/batches/$BATCH_1
curl http://localhost:4000/v1/batches/$BATCH_2

# List batches per account
curl "http://localhost:4000/v1/batches?model=gpt-4o-account-1"
curl "http://localhost:4000/v1/batches?model=gpt-4o-account-2"
```

### SDK Usage with Model Routing

```python
import litellm
import asyncio

# Upload file with model routing
file_obj = await litellm.acreate_file(
    file=open("batch.jsonl", "rb"),
    purpose="batch",
    model="gpt-4o-account-1",  # Route to specific account
)

print(f"File ID: {file_obj.id}")
# File ID is encoded with model info

# Create batch - automatically uses gpt-4o-account-1 credentials
batch = await litellm.acreate_batch(
    completion_window="24h",
    endpoint="/v1/chat/completions",
    input_file_id=file_obj.id,  # Model info embedded in ID
)

print(f"Batch ID: {batch.id}")
# Batch ID is also encoded

# Retrieve batch - automatically routes to correct account
retrieved = await litellm.aretrieve_batch(
    batch_id=batch.id,  # Model info embedded in ID
)

print(f"Batch status: {retrieved.status}")

# Or explicitly specify model
batch2 = await litellm.acreate_batch(
    completion_window="24h",
    endpoint="/v1/chat/completions",
    input_file_id="file-regular-id",
    model="gpt-4o-account-2",  # Explicit routing
)
```

### How ID Encoding Works

LiteLLM encodes model information into file and batch IDs using base64:

```
Original:  file-abc123
Encoded:   file-bGl0ZWxsbTpmaWxlLWFiYzEyMzttb2RlbCxncHQtNG8tdGVzdA
           └─┬─┘ └──────────────────┬──────────────────────┘
          prefix      base64(litellm:file-abc123;model,gpt-4o-test)

Original:  batch_xyz789
Encoded:   batch_bGl0ZWxsbTpiYXRjaF94eXo3ODk7bW9kZWwsZ3B0LTRvLXRlc3Q
           └──┬──┘ └──────────────────┬──────────────────────┘
           prefix       base64(litellm:batch_xyz789;model,gpt-4o-test)
```

The encoding:
- ✅ Preserves OpenAI-compatible prefixes (`file-`, `batch_`)
- ✅ Is transparent to clients
- ✅ Enables automatic routing without additional parameters
- ✅ Works across all batch and file endpoints

### Supported Endpoints

All batch and file endpoints support model-based routing:

| Endpoint | Method | Model Routing |
|----------|--------|---------------|
| `/v1/files` | POST | ✅ Via header/query/body |
| `/v1/files/{file_id}` | GET | ✅ Auto from encoded ID + header/query |
| `/v1/files/{file_id}/content` | GET | ✅ Auto from encoded ID + header/query |
| `/v1/files/{file_id}` | DELETE | ✅ Auto from encoded ID |
| `/v1/batches` | POST | ✅ Auto from file ID + header/query/body |
| `/v1/batches` | GET | ✅ Via header/query |
| `/v1/batches/{batch_id}` | GET | ✅ Auto from encoded ID |
| `/v1/batches/{batch_id}/cancel` | POST | ✅ Auto from encoded ID |

## **Supported Providers**:
### [Azure OpenAI](./providers/azure#azure-batches-api)
### [OpenAI](#quick-start)
### [Vertex AI](./providers/vertex#batch-apis)
### [Bedrock](./providers/bedrock_batches)
### [vLLM](./providers/vllm_batches)


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





## [Swagger API Reference](https://litellm-api.up.railway.app/#/batch)
