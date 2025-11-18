import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# /batches

Covers Batches, Files

| Feature | Supported | Notes | 
|-------|-------|-------|
| Supported Providers | OpenAI, Azure, Vertex, Bedrock | - |
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


## **Supported Providers**:
### [Azure OpenAI](./providers/azure#azure-batches-api)
### [OpenAI](#quick-start)
### [Vertex AI](./providers/vertex#batch-apis)
### [Bedrock](./providers/bedrock_batches)


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
