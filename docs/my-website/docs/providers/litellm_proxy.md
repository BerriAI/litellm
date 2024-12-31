import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# LiteLLM Proxy (LLM Gateway)

:::tip

[LiteLLM Providers a **self hosted** proxy server (AI Gateway)](../simple_proxy) to call all the LLMs in the OpenAI format

:::

**[LiteLLM Proxy](../simple_proxy) is OpenAI compatible**, you just need the `litellm_proxy/` prefix before the model

## Required Variables

```python
os.environ["LITELLM_PROXY_API_KEY"] = "" # "sk-1234" your litellm proxy api key 
os.environ["LITELLM_PROXY_API_BASE"] = "" # "http://localhost:4000" your litellm proxy api base
```


## Usage (Non Streaming)
```python
import os 
import litellm
from litellm import completion

os.environ["LITELLM_PROXY_API_KEY"] = ""

# set custom api base to your proxy
# either set .env or litellm.api_base
# os.environ["LITELLM_PROXY_API_BASE"] = ""
litellm.api_base = "your-openai-proxy-url"


messages = [{ "content": "Hello, how are you?","role": "user"}]

# litellm proxy call
response = completion(model="litellm_proxy/your-model-name", messages)
```

## Usage - passing `api_base`, `api_key` per request

If you need to set api_base dynamically, just pass it in completions instead - completions(...,api_base="your-proxy-api-base")

```python
import os 
import litellm
from litellm import completion

os.environ["LITELLM_PROXY_API_KEY"] = ""

messages = [{ "content": "Hello, how are you?","role": "user"}]

# litellm proxy call
response = completion(
    model="litellm_proxy/your-model-name", 
    messages, 
    api_base = "your-litellm-proxy-url",
    api_key = "your-litellm-proxy-api-key"
)
```
## Usage - Streaming

```python
import os 
import litellm
from litellm import completion

os.environ["LITELLM_PROXY_API_KEY"] = ""

messages = [{ "content": "Hello, how are you?","role": "user"}]

# openai call
response = completion(
    model="litellm_proxy/your-model-name", 
    messages, 
    api_base = "your-litellm-proxy-url", 
    stream=True
)

for chunk in response:
    print(chunk)
```


## **Usage with Langchain, LLamaindex, OpenAI Js, Anthropic SDK, Instructor**

#### [Follow this doc to see how to use litellm proxy with langchain, llamaindex, anthropic etc](../proxy/user_keys)