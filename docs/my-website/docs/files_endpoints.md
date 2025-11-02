
import TabItem from '@theme/TabItem';
import Tabs from '@theme/Tabs';

# Provider Files Endpoints

Files are used to upload documents that can be used with features like Assistants, Fine-tuning, and Batch API.

Use this to call the provider's `/files` endpoints directly, in the OpenAI format. 

## Quick Start

- Upload a File
- List Files
- Retrieve File Information
- Delete File
- Get File Content



<Tabs>
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
    extra_headers={"custom-llm-provider": "openai"}
)
```

List Files

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-...",
    base_url="http://0.0.0.0:4000/v1"
)

files = client.files.list(extra_headers={"custom-llm-provider": "openai"})
print("files=", files)
```

Retrieve File Information

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-...",
    base_url="http://0.0.0.0:4000/v1"
)

file = client.files.retrieve(file_id="file-abc123", extra_headers={"custom-llm-provider": "openai"})
print("file=", file)
```

Delete File

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-...",
    base_url="http://0.0.0.0:4000/v1"
)

response = client.files.delete(file_id="file-abc123", extra_headers={"custom-llm-provider": "openai"})
print("delete response=", response)
```

Get File Content

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-...",
    base_url="http://0.0.0.0:4000/v1"
)

content = client.files.content(file_id="file-abc123", extra_headers={"custom-llm-provider": "openai"})
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
