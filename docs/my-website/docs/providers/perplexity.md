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
    model="perplexity/llama-3.1-sonar-small-128k-online	", 
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
    model="perplexity/llama-3.1-sonar-small-128k-online	", 
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
| llama-3.1-sonar-small-128k-online	 | `completion(model="perplexity/llama-3.1-sonar-small-128k-online", messages)` | 
| llama-3.1-sonar-large-128k-online	 | `completion(model="perplexity/llama-3.1-sonar-large-128k-online", messages)` | 
| llama-3.1-sonar-huge-128k-online	 | `completion(model="perplexity/llama-3.1-sonar-huge-128k-online", messages)` | 


## Return citations 

Perplexity returns citations for the generated answer. Prior to November 2024, this required setting `return_citations=True`, however it now returns citations by default and that parameter has no effect. [Perplexity Docs](https://docs.perplexity.ai/api-reference/chat-completions).

If Perplexity returns citations, LiteLLM will pass it straight through. 

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
    model="perplexity/llama-3.1-sonar-small-128k-online", 
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
