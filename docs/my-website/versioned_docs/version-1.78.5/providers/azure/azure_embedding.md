import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Azure OpenAI Embeddings

### API keys
This can be set as env variables or passed as **params to litellm.embedding()**
```python
import os
os.environ['AZURE_API_KEY'] = 
os.environ['AZURE_API_BASE'] = 
os.environ['AZURE_API_VERSION'] = 
```

### Usage
```python
from litellm import embedding
response = embedding(
    model="azure/<your deployment name>",
    input=["good morning from litellm"],
    api_key=api_key,
    api_base=api_base,
    api_version=api_version,
)
print(response)
```

| Model Name           | Function Call                               |
|----------------------|---------------------------------------------|
| text-embedding-ada-002 | `embedding(model="azure/<your deployment name>", input=input)` |

h/t to [Mikko](https://www.linkedin.com/in/mikkolehtimaki/) for this integration


## **Usage - LiteLLM Proxy Server**

Here's how to call Azure OpenAI models with the LiteLLM Proxy Server

### 1. Save key in your environment

```bash
export AZURE_API_KEY=""
```

### 2. Start the proxy 

```yaml
model_list:
  - model_name: text-embedding-ada-002
    litellm_params:
      model: azure/my-deployment-name
      api_base: https://openai-gpt-4-test-v-1.openai.azure.com/
      api_version: "2023-05-15"
      api_key: os.environ/AZURE_API_KEY # The `os.environ/` prefix tells litellm to read this from the env.
```

### 3. Test it

<Tabs>
<TabItem value="Curl" label="Curl Request">

```shell
curl --location 'http://0.0.0.0:4000/embeddings' \
  --header 'Content-Type: application/json' \
  --data ' {
  "model": "text-embedding-ada-002",
  "input": ["write a litellm poem"]
  }'
```
</TabItem>
<TabItem value="openai" label="OpenAI v1.0.0+">

```python
import openai
from openai import OpenAI

# set base_url to your proxy server
# set api_key to send to proxy server
client = OpenAI(api_key="<proxy-api-key>", base_url="http://0.0.0.0:4000")

response = client.embeddings.create(
    input=["hello from litellm"],
    model="text-embedding-ada-002"
)

print(response)

```
</TabItem>
</Tabs>


