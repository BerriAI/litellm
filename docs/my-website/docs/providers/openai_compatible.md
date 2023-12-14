# OpenAI-Compatible Endpoints

To call models hosted behind an openai proxy, make 2 changes:

1. Put `openai/` in front of your model name, so litellm knows you're trying to call an openai-compatible endpoint. 

2. **Do NOT** add anything additional to the base url e.g. `/v1/embedding`. LiteLLM uses the openai-client to make these calls, and that automatically adds the relevant endpoints. 

## Usage

```python
import litellm
from litellm import embedding
litellm.set_verbose = True
import os

 
litellm_proxy_endpoint = "http://0.0.0.0:8000"
bearer_token = "sk-1234"

CHOSEN_LITE_LLM_EMBEDDING_MODEL = "openai/GPT-J 6B - Sagemaker Text Embedding (Internal)"

litellm.set_verbose = False

print(litellm_proxy_endpoint)

 

response = embedding(

    model = CHOSEN_LITE_LLM_EMBEDDING_MODEL,     # add `openai/` prefix to model so litellm knows to route to OpenAI

    api_key=bearer_token,

    api_base=litellm_proxy_endpoint,       # set API Base of your Custom OpenAI Endpoint

    input=["good morning from litellm"],

    api_version='2023-07-01-preview'

)

print('================================================')

print(len(response.data[0]['embedding']))

```