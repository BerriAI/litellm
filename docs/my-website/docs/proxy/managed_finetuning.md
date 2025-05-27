# ✨ [BETA] LiteLLM Managed Files with Finetuning


:::info

This is a free LiteLLM Enterprise feature.

Available via the `litellm[proxy]` package or any `litellm` docker image.

:::


| Property | Value | Comments |
| --- | --- | --- |
| Proxy | ✅ |  |
| SDK | ❌ | Requires postgres DB for storing file ids. |
| Available across all [Batch providers](../batches#supported-providers) | ✅ |  |
| Supported endpoints | `/fine_tuning/jobs` |  |

## Overview

Use this to:

- Create Finetuning jobs across OpenAI/Azure/Vertex AI in the OpenAI format (no additional `custom_llm_provider` param required). 
- Control finetuning model access by key/user/team (same as chat completion models)


## (Proxy Admin) Usage

Here's how to give developers access to your Finetuning models.

### 1. Setup config.yaml

Include `/fine_tuning` in the `supported_endpoints` list. Tells developers this model supports the `/fine_tuning` endpoint.

```yaml showLineNumbers title="litellm_config.yaml"
model_list:
  - model_name: "gpt-4.1-openai"
    litellm_params:
      model: gpt-4.1
      api_key: os.environ/OPENAI_API_KEY
    model_info:
      supported_endpoints: ["/chat/completions", "/fine_tuning"]
```

### 2. Create Virtual Key

```bash showLineNumbers title="create_virtual_key.sh"
curl -L -X POST 'https://{PROXY_BASE_URL}/key/generate' \
-H 'Authorization: Bearer ${PROXY_API_KEY}' \
-H 'Content-Type: application/json' \
-d '{"models": ["gpt-4.1-openai"]}'
```


You can now use the virtual key to access the finetuning models (See Developer flow).

## (Developer) Usage

Here's how to create a LiteLLM managed file and execute Finetuning CRUD operations with the file. 

### 1. Create request.jsonl 


```json showLineNumbers title="request.jsonl"
{"messages": [{"role": "system", "content": "Clippy is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "Paris, as if everyone doesn't know that already."}]}
{"messages": [{"role": "system", "content": "Clippy is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "Who wrote 'Romeo and Juliet'?"}, {"role": "assistant", "content": "Oh, just some guy named William Shakespeare. Ever heard of him?"}]}
```

### 2. Upload File 

Specify `target_model_names: "<model-name>"` to enable LiteLLM managed files and request validation.

model-name should be the same as the model-name in the request.jsonl

```python showLineNumbers title="create_finetuning_job.py"
from openai import OpenAI

client = OpenAI(
    base_url="http://0.0.0.0:4000",
    api_key="sk-1234",
)

# Upload file
finetuning_input_file = client.files.create(
    file=open("./request.jsonl", "rb"),
    purpose="fine-tune",
    extra_body={"target_model_names": "gpt-4.1-openai"}
)
print(finetuning_input_file)

```


**Where is the file written?**:

All gpt-4.1-openai deployments will be written to. This enables loadbalancing across all gpt-4.1-openai deployments in Step 3, when a job is created. Once the job is created, any retrieve/list/cancel operations will be routed to that deployment.

### 3. Create the Finetuning Job

```python showLineNumbers title="create_finetuning_job.py"
... # Step 2

file_id = finetuning_input_file.id

# Create Finetuning Job
ft_job = client.fine_tuning.jobs.create(
    model="gpt-4.1-openai",  # litellm public model name you want to finetune                  
    training_file=file_id,
)
```

### 4. Retrieve Finetuning Job

```python showLineNumbers title="create_finetuning_job.py"
... # Step 3

response = client.fine_tuning.jobs.retrieve(ft_job.id)
print(response)
```

### 5. List Finetuning Jobs

```python showLineNumbers title="create_finetuning_job.py"
...

client.fine_tuning.jobs.list(extra_body={"target_model_names": "gpt-4.1-openai"})
```

### 6. Cancel a Finetuning Job

```python showLineNumbers title="create_finetuning_job.py"
...

cancel_ft_job = client.fine_tuning.jobs.cancel(
    fine_tuning_job_id=ft_job.id,                          # fine tuning job id
)
```



## E2E Example

```python showLineNumbers title="create_finetuning_job.py"
from openai import OpenAI

client = OpenAI(
    base_url="http://0.0.0.0:4000",
    api_key="sk-...",
    max_retries=0
)


# Upload file
finetuning_input_file = client.files.create(
    file=open("./fine_tuning.jsonl", "rb"), # {"model": "azure-gpt-4o"} <-> {"model": "gpt-4o-my-special-deployment"}
    purpose="fine-tune",
    extra_body={"target_model_names": "gpt-4.1-openai"} # 👈 Tells litellm which regions/projects to write the file in. 
)
print(finetuning_input_file) # file.id = "litellm_proxy/..." = {"model_name": {"deployment_id": "deployment_file_id"}}

file_id = finetuning_input_file.id
# # file_id = "bGl0ZWxs..."

# ## create fine-tuning job 
ft_job = client.fine_tuning.jobs.create(
    model="gpt-4.1-openai",  # litellm model name you want to finetune                  
    training_file=file_id,
)

print(f"ft_job: {ft_job}")

ft_job_id = ft_job.id
## cancel fine-tuning job 
cancel_ft_job = client.fine_tuning.jobs.cancel(
    fine_tuning_job_id=ft_job_id,                          # fine tuning job id
)

print("response from cancel ft job={}".format(cancel_ft_job))
# list fine-tuning jobs 
list_ft_jobs = client.fine_tuning.jobs.list(
    extra_query={"target_model_names": "gpt-4.1-openai"}   # tell litellm proxy which provider to use
)

print("list of ft jobs={}".format(list_ft_jobs))

# get fine-tuning job 
response = client.fine_tuning.jobs.retrieve(ft_job.id)
print(response)
```

## FAQ

### Where are my files written?

When a `target_model_names` is specified, the file is written to all deployments that match the `target_model_names`.

No additional infrastructure is required.