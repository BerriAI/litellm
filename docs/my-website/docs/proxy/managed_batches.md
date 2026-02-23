# [BETA] LiteLLM Managed Files with Batches

:::info

This is a free LiteLLM Enterprise feature.

Available via the `litellm[proxy]` package or any `litellm` docker image.

:::


| Feature | Description | Comments |
| --- | --- | --- |
| Proxy | ✅ |  |
| SDK | ❌ | Requires postgres DB for storing file ids |
| Available across all [Batch providers](../batches#supported-providers) | ✅ |  |


## Overview

Use this to:

- Loadbalance across multiple Azure Batch deployments
- Control batch model access by key/user/team (same as chat completion models)


## (Proxy Admin) Usage

Here's how to give developers access to your Batch models.

### 1. Setup config.yaml

- specify `mode: batch` for each model: Allows developers to know this is a batch model.

```yaml showLineNumbers title="litellm_config.yaml"
model_list:
  - model_name: "gpt-4o-batch"
    litellm_params:
      model: azure/gpt-4o-mini-general-deployment
      api_base: os.environ/AZURE_API_BASE
      api_key: os.environ/AZURE_API_KEY
    model_info: 
      mode: batch # 👈 SPECIFY MODE AS BATCH, to tell user this is a batch model
  - model_name: "gpt-4o-batch"
    litellm_params:
      model: azure/gpt-4o-mini-special-deployment
      api_base: os.environ/AZURE_API_BASE_2
      api_key: os.environ/AZURE_API_KEY_2
    model_info: 
      mode: batch # 👈 SPECIFY MODE AS BATCH, to tell user this is a batch model

```

### 2. Create Virtual Key

```bash showLineNumbers title="create_virtual_key.sh"
curl -L -X POST 'https://{PROXY_BASE_URL}/key/generate' \
-H 'Authorization: Bearer ${PROXY_API_KEY}' \
-H 'Content-Type: application/json' \
-d '{"models": ["gpt-4o-batch"]}'
```


You can now use the virtual key to access the batch models (See Developer flow).

## (Developer) Usage

Here's how to create a LiteLLM managed file and execute Batch CRUD operations with the file. 

### 1. Create request.jsonl 

- Check models available via `/model_group/info`
- See all models with `mode: batch`
- Set `model` in .jsonl to the model from `/model_group/info`

```json showLineNumbers title="request.jsonl"
{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-batch", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-batch", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
```

Expectation:

- LiteLLM translates this to the azure deployment specific value (e.g. `gpt-4o-mini-general-deployment`)

### 2. Upload File 

Specify `target_model_names: "<model-name>"` to enable LiteLLM managed files and request validation.

model-name should be the same as the model-name in the request.jsonl

```python showLineNumbers title="create_batch.py"
from openai import OpenAI

client = OpenAI(
    base_url="http://0.0.0.0:4000",
    api_key="sk-1234",
)

# Upload file
batch_input_file = client.files.create(
    file=open("./request.jsonl", "rb"), # {"model": "gpt-4o-batch"} <-> {"model": "gpt-4o-mini-special-deployment"}
    purpose="batch",
    extra_body={"target_model_names": "gpt-4o-batch"}
)
print(batch_input_file)
```


**Where is the file written?**:

All gpt-4o-batch deployments (gpt-4o-mini-general-deployment, gpt-4o-mini-special-deployment) will be written to. This enables loadbalancing across all gpt-4o-batch deployments in Step 3.

### 3. Create + Retrieve the batch

```python showLineNumbers title="create_batch.py"
...
# Create batch
batch = client.batches.create( 
    input_file_id=batch_input_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={"description": "Test batch job"},
)
print(batch)

# Retrieve batch

batch_response = client.batches.retrieve(
    batch_id
)
status = batch_response.status
```

### 4. Retrieve Batch Content 

```python showLineNumbers title="create_batch.py"
...

file_id = batch_response.output_file_id

file_response = client.files.content(file_id)
print(file_response.text)
```

### 5. List batches

```python showLineNumbers title="create_batch.py"
...

client.batches.list(limit=10, extra_query={"target_model_names": "gpt-4o-batch"})
```

### [Coming Soon] Cancel a batch

```python showLineNumbers title="create_batch.py"
...

client.batches.cancel(batch_id)
```



## E2E Example

```python showLineNumbers title="create_batch.py"
import json
from pathlib import Path
from openai import OpenAI

"""
litellm yaml: 

model_list:
    - model_name: gpt-4o-batch
      litellm_params:
        model: azure/gpt-4o-my-special-deployment
        api_key: ..
        api_base: .. 

---
request.jsonl: 
{
    {
        ...,
        "body":{"model": "gpt-4o-batch", ...}}
    }
}
"""

client = OpenAI(
    base_url="http://0.0.0.0:4000",
    api_key="sk-1234",
)

# Upload file
batch_input_file = client.files.create(
    file=open("./request.jsonl", "rb"),
    purpose="batch",
    extra_body={"target_model_names": "gpt-4o-batch"}
)
print(batch_input_file) 


# Create batch
batch = client.batches.create( # UPDATE BATCH ID TO FILE ID 
    input_file_id=batch_input_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={"description": "Test batch job"},
)
print(batch)
batch_id = batch.id

# Retrieve batch

batch_response = client.batches.retrieve( # LOG VIRTUAL MODEL NAME
    batch_id
)
status = batch_response.status

print(f"status: {status}, output_file_id: {batch_response.output_file_id}")

# Download file
output_file_id = batch_response.output_file_id
print(f"output_file_id: {output_file_id}")
if not output_file_id:
    output_file_id = batch_response.error_file_id

if output_file_id:
    file_response = client.files.content(
        output_file_id
    )
    raw_responses = file_response.text.strip().split("\n")

    with open(
        Path.cwd().parent / "unified_batch_output.json", "w"
    ) as output_file:
        for raw_response in raw_responses:
            json.dump(json.loads(raw_response), output_file)
            output_file.write("\n")
## List Batch

list_batch_response = client.batches.list( # LOG VIRTUAL MODEL NAME
    extra_query={"target_model_names": "gpt-4o-batch"}
)

## Cancel Batch

batch_response = client.batches.cancel( # LOG VIRTUAL MODEL NAME
    batch_id
)
status = batch_response.status

print(f"status: {status}")
```

## FAQ

### Where are my files written?

When a `target_model_names` is specified, the file is written to all deployments that match the `target_model_names`.

No additional infrastructure is required.

## Could the batch be created at the eastus-01 deployment but a subsequent get of the batch could be routed to (a different) eastus2-01 deployment ?

**A.** You can loadbalance b/w multiple models for the initial create batch. Once that's created - we return a file id, which encodes the model deployment used, so it's sticky and only sends any get/delete to that deployment. 







