# [BETA] Unified File ID with Batches

:::info

This is a LiteLLM Enterprise feature.

Available for free via the `litellm[proxy]` package or any `litellm` docker image.

:::


| Feature | Description | Comments |
| --- | --- | --- |
| Proxy | ✅ |  |
| SDK | ❌ | Requires postgres DB for storing file ids |
| Available across all [Batch providers](../batches#supported-providers) | ✅ |  |


## Overview

Use this to:

- Loadbalance across multiple Azure Batch deployments
- Easily switch across Azure/OpenAI/VertexAI Batch APIs without needing to reupload files each time
- Control batch model access by key/user/team (same as chat completion models)

## (Proxy Admin) Usage

Here's how to give developers access to your Batch models.

### 1. Setup config.yaml

- specify `mode: batch` for each model: Allows developers to know this is a batch model.

```yaml
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

```bash
curl -L -X POST 'https://{PROXY_BASE_URL}/key/generate' \
-H 'Authorization: Bearer ${PROXY_API_KEY}' \
-H 'Content-Type: application/json' \
-d '{"models": ["gpt-4o-batch"]}'
```


You can now use the virtual key to access the batch models (See Developer flow).

## (Developer) Usage

### 1. Create request.jsonl 

- Check models available via `/model_group/info`
- See all models with `mode: batch`
- Set `model` in .jsonl to the model from `/model_group/info`

```json
{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-batch", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-batch", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
```

Expectation:

- LiteLLM translates this to the azure model specific value

### 2. Upload File 

Specify `target_model_names: "<model-name>"` to enable LiteLLM managed files and request validation.

model-name should be the same as the model-name in the request.jsonl

```python
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

Expectation:

- This is used to validate if user has model access. 
- This is used to write the file to the correct deployments (all gpt-4o-batch deployments will be written to).

### 3. Create + Retrieve the batch

```python
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

batch_response = client.batches.retrieve( # LOG VIRTUAL MODEL NAME
    batch_id
)
status = batch_response.status
```

### 4. Retrieve Batch Content 

```python
...

file_id = batch_response.output_file_id

file_response = client.files.content(file_id)
print(file_response.text)
```

### 5. Cancel a batch

```python
...

client.batches.cancel(batch_id)
```

### 6. List batches

```python
...

client.batches.list(limit=10, extra_body={"target_model_names": "gpt-4o-batch"})
```

