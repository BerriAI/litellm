import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Perplexity AI (pplx-api)
https://www.perplexity.ai

## API Key
```python
# env variable
os.environ['PERPLEXITYAI_API_KEY']
```

## Sample Usage
```python
from litellm import completion
import os

os.environ['PERPLEXITYAI_API_KEY'] = ""
response = completion(
    model="perplexity/mistral-7b-instruct", 
    messages=messages
)
print(response)
```

## Sample Usage - Streaming
```python
from litellm import completion
import os

os.environ['PERPLEXITYAI_API_KEY'] = ""
response = completion(
    model="perplexity/mistral-7b-instruct", 
    messages=messages,
    stream=True
)

for chunk in response:
    print(chunk)
```


## Supported Models
All models listed here https://docs.perplexity.ai/docs/model-cards are supported.  Just do `model=perplexity/<model-name>`.

| Model Name               | Function Call                                                                                                                                                      |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| pplx-7b-chat | `completion(model="perplexity/pplx-7b-chat", messages)` | 
| pplx-70b-chat | `completion(model="perplexity/pplx-70b-chat", messages)` | 
| pplx-7b-online | `completion(model="perplexity/pplx-7b-online", messages)` | 
| pplx-70b-online | `completion(model="perplexity/pplx-70b-online", messages)` | 
| codellama-34b-instruct | `completion(model="perplexity/codellama-34b-instruct", messages)` | 
| llama-2-13b-chat | `completion(model="perplexity/llama-2-13b-chat", messages)` | 
| llama-2-70b-chat | `completion(model="perplexity/llama-2-70b-chat", messages)` | 
| mistral-7b-instruct | `completion(model="perplexity/mistral-7b-instruct", messages)` | 
| openhermes-2-mistral-7b | `completion(model="perplexity/openhermes-2-mistral-7b", messages)` | 
| openhermes-2.5-mistral-7b | `completion(model="perplexity/openhermes-2.5-mistral-7b", messages)` | 
| pplx-7b-chat-alpha | `completion(model="perplexity/pplx-7b-chat-alpha", messages)` | 
| pplx-70b-chat-alpha | `completion(model="perplexity/pplx-70b-chat-alpha", messages)` | 







## Return citations 

Perplexity supports returning citations via `return_citations=True`. [Perplexity Docs](https://docs.perplexity.ai/reference/post_chat_completions). Note: Perplexity has this feature in **closed beta**, so you need them to grant you access to get citations from their API. 

If perplexity returns citations, LiteLLM will pass it straight through. 

:::info

For passing more provider-specific, [go here](../completion/provider_specific_params.md)
:::

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
import os

os.environ['PERPLEXITYAI_API_KEY'] = ""
response = completion(
    model="perplexity/mistral-7b-instruct", 
    messages=messages,
    return_citations=True
)
print(response)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Add perplexity to config.yaml

```yaml
model_list:
  - model_name: "perplexity-model"
    litellm_params:
      model: "llama-3.1-sonar-small-128k-online"
      api_key: os.environ/PERPLEXITY_API_KEY
```

2. Start proxy 

```bash
litellm --config /path/to/config.yaml
```

3. Test it! 

```bash
curl -L -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "model": "perplexity-model",
    "messages": [
      {
        "role": "user",
        "content": "Who won the world cup in 2022?"
      }
    ],
    "return_citations": true
}'
```

[**Call w/ OpenAI SDK, Langchain, Instructor, etc.**](../proxy/user_keys.md#chatcompletions)

</TabItem>
</Tabs>
