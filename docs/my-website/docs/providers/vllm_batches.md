import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# vLLM - Batch + Files API

LiteLLM supports vLLM's Batch and Files API for processing large volumes of requests asynchronously.

| Feature | Supported |
|---------|-----------|
| `/v1/files` | ✅ |
| `/v1/batches` | ✅ |
| Cost Tracking | ✅ |

## Quick Start

### 1. Setup config.yaml

Define your vLLM model in `config.yaml`. LiteLLM uses the model name to route batch requests to the correct vLLM server.

```yaml
model_list:
  - model_name: my-vllm-model
    litellm_params:
      model: hosted_vllm/meta-llama/Llama-2-7b-chat-hf
      api_base: http://localhost:8000  # your vLLM server
```

### 2. Start LiteLLM Proxy

```bash
litellm --config /path/to/config.yaml
```

### 3. Create Batch File

Create a JSONL file with your batch requests:

```jsonl
{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "my-vllm-model", "messages": [{"role": "user", "content": "Hello!"}]}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "my-vllm-model", "messages": [{"role": "user", "content": "How are you?"}]}}
```

### 4. Upload File & Create Batch

:::tip Model Routing
LiteLLM needs to know which model (and therefore which vLLM server) to use for batch operations. Specify the model using the `x-litellm-model` header when uploading files. LiteLLM will encode this model info into the file ID, so subsequent batch operations automatically route to the correct server.

See [Multi-Account / Model-Based Routing](../batches#multi-account--model-based-routing) for more details.
:::

<Tabs>
<TabItem value="curl" label="cURL">

**Upload File**

```bash
curl http://localhost:4000/v1/files \
  -H "Authorization: Bearer sk-1234" \
  -H "x-litellm-model: my-vllm-model" \
  -F purpose="batch" \
  -F file="@batch_requests.jsonl"
```

**Create Batch**

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

**Check Batch Status**

```bash
curl http://localhost:4000/v1/batches/batch_abc123 \
  -H "Authorization: Bearer sk-1234"
```

</TabItem>
<TabItem value="python" label="Python SDK">

```python
import litellm
import asyncio

async def run_vllm_batch():
    # Upload file
    file_obj = await litellm.acreate_file(
        file=open("batch_requests.jsonl", "rb"),
        purpose="batch",
        custom_llm_provider="hosted_vllm",
    )
    print(f"File uploaded: {file_obj.id}")

    # Create batch
    batch = await litellm.acreate_batch(
        completion_window="24h",
        endpoint="/v1/chat/completions",
        input_file_id=file_obj.id,
        custom_llm_provider="hosted_vllm",
    )
    print(f"Batch created: {batch.id}")

    # Poll for completion
    while True:
        batch_status = await litellm.aretrieve_batch(
            batch_id=batch.id,
            custom_llm_provider="hosted_vllm",
        )
        print(f"Status: {batch_status.status}")
        
        if batch_status.status == "completed":
            break
        elif batch_status.status in ["failed", "cancelled"]:
            raise Exception(f"Batch failed: {batch_status.status}")
        
        await asyncio.sleep(5)

    # Get results
    if batch_status.output_file_id:
        results = await litellm.afile_content(
            file_id=batch_status.output_file_id,
            custom_llm_provider="hosted_vllm",
        )
        print(f"Results: {results}")

asyncio.run(run_vllm_batch())
```

</TabItem>
</Tabs>

## Supported Operations

| Operation | Endpoint | Method |
|-----------|----------|--------|
| Upload file | `/v1/files` | POST |
| List files | `/v1/files` | GET |
| Retrieve file | `/v1/files/{file_id}` | GET |
| Delete file | `/v1/files/{file_id}` | DELETE |
| Get file content | `/v1/files/{file_id}/content` | GET |
| Create batch | `/v1/batches` | POST |
| List batches | `/v1/batches` | GET |
| Retrieve batch | `/v1/batches/{batch_id}` | GET |
| Cancel batch | `/v1/batches/{batch_id}/cancel` | POST |

## Environment Variables

```bash
# Set vLLM server endpoint
export HOSTED_VLLM_API_BASE="http://localhost:8000"

# Optional: API key if your vLLM server requires authentication
export HOSTED_VLLM_API_KEY="your-api-key"
```

## How Model Routing Works

When you upload a file with `x-litellm-model: my-vllm-model`, LiteLLM:

1. Encodes the model name into the returned file ID
2. Uses this encoded model info to automatically route subsequent batch operations to the correct vLLM server
3. No need to specify the model again when creating batches or retrieving results

This enables multi-tenant batch processing where different teams can use different vLLM deployments through the same LiteLLM proxy.

**Learn more:** [Multi-Account / Model-Based Routing](../batches#multi-account--model-based-routing)

## Related

- [vLLM Provider Overview](./vllm)
- [Batch API Overview](../batches)
- [Files API](../files_endpoints)
