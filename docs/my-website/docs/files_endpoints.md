
import TabItem from '@theme/TabItem';
import Tabs from '@theme/Tabs';

# /files

Files are used to upload documents that can be used with features like Assistants, Fine-tuning, and Batch API.

## Quick Start

- Upload a File
- List Files
- Retrieve File Information
- Delete File
- Get File Content



<Tabs>
<TabItem value="managed_files" label="[BETA] LiteLLM Managed Files">

Use this to reuse the same 'file id' across different providers.

Limitations of LiteLLM Managed Files:
- Only works for `/chat/completions` requests. Follow [here](https://github.com/BerriAI/litellm/discussions/9632) for batches support.

### 1. Setup config.yaml

```
model_list:
    - model_name: "gemini-2.0-flash"
      litellm_params:
        model: vertex_ai/gemini-2.0-flash
        vertex_project: my-project-id
        vertex_location: us-central1
    - model_name: "gpt-4o-mini-openai"
      litellm_params:
        model: gpt-4o-mini
        api_key: os.environ/OPENAI_API_KEY
```

### 2. Start proxy

```bash
litellm --config /path/to/config.yaml
```

### 3. Test it!

Specify `target_model_names` to use the same file id across different providers. This is the list of model_names set via config.yaml (or 'public_model_names' on UI). 

```python
target_model_names="gpt-4o-mini-openai, gemini-2.0-flash" # 👈 Specify model_names
```

Check `/v1/models` to see the list of available model names for a key.

#### **Store a PDF file**

```python
from openai import OpenAI

client = OpenAI(base_url="http://0.0.0.0:4000", api_key="sk-1234", max_retries=0)


# Download and save the PDF locally 
url = (
    "https://storage.googleapis.com/cloud-samples-data/generative-ai/pdf/2403.05530.pdf"
)
response = requests.get(url)
response.raise_for_status()

# Save the PDF locally
with open("2403.05530.pdf", "wb") as f:
    f.write(response.content)

file = client.files.create(
    file=open("2403.05530.pdf", "rb"),
    purpose="user_data", # can be any openai 'purpose' value
    extra_body={"target_model_names": "gpt-4o-mini-openai, gemini-2.0-flash"}, # 👈 Specify model_names
)

print(f"file id={file.id}")
```

#### **Use the same file id across different providers**

<Tabs>
<TabItem value="openai" label="OpenAI">

```python
completion = client.chat.completions.create(
    model="gpt-4o-mini-openai",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this recording?"},
                {
                    "type": "file",
                    "file": {
                        "file_id": file.id,
                    },
                },
            ],
        },
    ]
)

print(completion.choices[0].message)
```


</TabItem>
<TabItem value="vertex" label="Vertex AI">

```python
completion = client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this recording?"},
                {
                    "type": "file",
                    "file": {
                        "file_id": file.id,
                    },
                },
            ],
        },
    ]
)

print(completion.choices[0].message)

```

</TabItem>
</Tabs>



#### E2E Example

```python   
import base64
import requests
from openai import OpenAI

client = OpenAI(base_url="http://0.0.0.0:4000", api_key="sk-1234", max_retries=0)


# Download and save the PDF locally
url = (
    "https://storage.googleapis.com/cloud-samples-data/generative-ai/pdf/2403.05530.pdf"
)
response = requests.get(url)
response.raise_for_status()

# Save the PDF locally
with open("2403.05530.pdf", "wb") as f:
    f.write(response.content)

# Read the local PDF file
file = client.files.create(
    file=open("2403.05530.pdf", "rb"),
    purpose="user_data", # can be any openai 'purpose' value
    extra_body={"target_model_names": "gpt-4o-mini-openai, vertex_ai/gemini-2.0-flash"},
)

print(f"file.id: {file.id}") # 👈 Unified file id

## GEMINI CALL ### 
completion = client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this recording?"},
                {
                    "type": "file",
                    "file": {
                        "file_id": file.id,
                    },
                },
            ],
        },
    ]
)

print(completion.choices[0].message)


### OPENAI CALL ### 
completion = client.chat.completions.create(
    model="gpt-4o-mini-openai",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this recording?"},
                {
                    "type": "file",
                    "file": {
                        "file_id": file.id,
                    },
                },
            ],
        },
    ],
)

print(completion.choices[0].message)

```


</TabItem>
<TabItem value="proxy" label="LiteLLM PROXY Server">

1. Setup config.yaml

```
# for /files endpoints
files_settings:
  - custom_llm_provider: azure
    api_base: https://exampleopenaiendpoint-production.up.railway.app
    api_key: fake-key
    api_version: "2023-03-15-preview"
  - custom_llm_provider: openai
    api_key: os.environ/OPENAI_API_KEY
```

2. Start LiteLLM PROXY Server

```bash
litellm --config /path/to/config.yaml

## RUNNING on http://0.0.0.0:4000
```

3. Use OpenAI's /files endpoints

Upload a File

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-...",
    base_url="http://0.0.0.0:4000/v1"
)

client.files.create(
    file=wav_data,
    purpose="user_data",
    extra_body={"custom_llm_provider": "openai"}
)
```

List Files

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-...",
    base_url="http://0.0.0.0:4000/v1"
)

files = client.files.list(extra_body={"custom_llm_provider": "openai"})
print("files=", files)
```

Retrieve File Information

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-...",
    base_url="http://0.0.0.0:4000/v1"
)

file = client.files.retrieve(file_id="file-abc123", extra_body={"custom_llm_provider": "openai"})
print("file=", file)
```

Delete File

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-...",
    base_url="http://0.0.0.0:4000/v1"
)

response = client.files.delete(file_id="file-abc123", extra_body={"custom_llm_provider": "openai"})
print("delete response=", response)
```

Get File Content

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-...",
    base_url="http://0.0.0.0:4000/v1"
)

content = client.files.content(file_id="file-abc123", extra_body={"custom_llm_provider": "openai"})
print("content=", content)
```

</TabItem>
<TabItem value="sdk" label="SDK">

**Upload a File**
```python
from litellm
import os 

os.environ["OPENAI_API_KEY"] = "sk-.."

file_obj = await litellm.acreate_file(
    file=open("mydata.jsonl", "rb"),
    purpose="fine-tune",
    custom_llm_provider="openai",
)
print("Response from creating file=", file_obj)
```

**List Files**
```python
files = await litellm.alist_files(
    custom_llm_provider="openai",
    limit=10
)
print("files=", files)
```

**Retrieve File Information**
```python
file = await litellm.aretrieve_file(
    file_id="file-abc123",
    custom_llm_provider="openai"
)
print("file=", file)
```

**Delete File**
```python
response = await litellm.adelete_file(
    file_id="file-abc123",
    custom_llm_provider="openai"
)
print("delete response=", response)
```

**Get File Content**
```python
content = await litellm.afile_content(
    file_id="file-abc123",
    custom_llm_provider="openai"
)
print("file content=", content)
```

</TabItem>
</Tabs>


## **Supported Providers**:

### [OpenAI](#quick-start)

### [Azure OpenAI](./providers/azure#azure-batches-api)

### [Vertex AI](./providers/vertex#batch-apis)

## [Swagger API Reference](https://litellm-api.up.railway.app/#/files)
