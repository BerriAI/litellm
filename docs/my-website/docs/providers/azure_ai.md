# Azure AI Studio

## Sample Usage
Call Azure Command R Plus in the OpenAI Format
- model=`azure/command-r-plus`. The `azure/` prefix sends this to Azure. The `command-r-plus` indicates the base model being called
- `model_id` = This is your deployment name on Azure AI studio

```python
import litellm
response = litellm.completion(
    model="azure/command-r-plus",
    model_id="Cohere-command-r-plus-gylpd",
    api_base="https://Cohere-command-r-plus-gylpd-serverless.eastus2.inference.ai.azure.com/v1/"
    api_key="eskk******"
    messages=[{"role": "user", "content": "What is the meaning of life?"}],
)
```

## Using Mistral models deployed on Azure AI Studio

### Sample Usage - setting env vars 

Set `MISTRAL_AZURE_API_KEY` and `MISTRAL_AZURE_API_BASE` in your env

```shell
MISTRAL_AZURE_API_KEY = "zE************""
MISTRAL_AZURE_API_BASE = "https://Mistral-large-nmefg-serverless.eastus2.inference.ai.azure.com/v1"
```

```python
from litellm import completion
import os

response = completion(
    model="mistral/Mistral-large-dfgfj", 
    messages=[
       {"role": "user", "content": "hello from litellm"}
   ],
)
print(response)
```

### Sample Usage - passing `api_base` and `api_key` to `litellm.completion`
```python
from litellm import completion
import os

response = completion(
    model="mistral/Mistral-large-dfgfj", 
    api_base="https://Mistral-large-dfgfj-serverless.eastus2.inference.ai.azure.com",
    api_key = "JGbKodRcTp****"
    messages=[
       {"role": "user", "content": "hello from litellm"}
   ],
)
print(response)
```

### [LiteLLM Proxy] Using Mistral Models 

Set this on your litellm proxy config.yaml
```yaml
model_list:
  - model_name: mistral
    litellm_params:
      model: mistral/Mistral-large-dfgfj
      api_base: https://Mistral-large-dfgfj-serverless.eastus2.inference.ai.azure.com
      api_key: JGbKodRcTp****
```


