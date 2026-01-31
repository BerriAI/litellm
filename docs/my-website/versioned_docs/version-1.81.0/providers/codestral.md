import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Codestral API [Mistral AI]

Codestral is available in select code-completion plugins but can also be queried directly. See the documentation for more details.

## API Key
```python
# env variable
os.environ['CODESTRAL_API_KEY']
```

## FIM / Completions

:::info

Official Mistral API Docs: https://docs.mistral.ai/api/#operation/createFIMCompletion

:::


<Tabs>
<TabItem value="no-streaming" label="No Streaming">

#### Sample Usage

```python
import os
import litellm

os.environ['CODESTRAL_API_KEY']

response = await litellm.atext_completion(
    model="text-completion-codestral/codestral-2405",
    prompt="def is_odd(n): \n return n % 2 == 1 \ndef test_is_odd():", 
    suffix="return True",                                              # optional
    temperature=0,                                                     # optional
    top_p=1,                                                           # optional
    max_tokens=10,                                                     # optional
    min_tokens=10,                                                     # optional
    seed=10,                                                           # optional
    stop=["return"],                                                   # optional
)
```

#### Expected Response

```json
{
  "id": "b41e0df599f94bc1a46ea9fcdbc2aabe",
  "object": "text_completion",
  "created": 1589478378,
  "model": "codestral-latest",
  "choices": [
    {
      "text": "\n assert is_odd(1)\n assert",
      "index": 0,
      "logprobs": null,
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 7,
    "total_tokens": 12
  }
}

```


</TabItem>
<TabItem value="stream" label="Streaming">

#### Sample Usage - Streaming

```python
import os
import litellm

os.environ['CODESTRAL_API_KEY']

response = await litellm.atext_completion(
    model="text-completion-codestral/codestral-2405",
    prompt="def is_odd(n): \n return n % 2 == 1 \ndef test_is_odd():",
    suffix="return True",    # optional
    temperature=0,           # optional
    top_p=1,                 # optional
    stream=True,                
    seed=10,                 # optional
    stop=["return"],         # optional
)

async for chunk in response:
    print(chunk)
```

#### Expected Response

```json
{
  "id": "726025d3e2d645d09d475bb0d29e3640",
  "object": "text_completion",
  "created": 1718659669,
  "choices": [
    {
      "text": "This",
      "index": 0,
      "logprobs": null,
      "finish_reason": null
    }
  ],
  "model": "codestral-2405", 
}

```
</TabItem>
</Tabs>

### Supported Models
All models listed here https://docs.mistral.ai/platform/endpoints are supported. We actively maintain the list of models, pricing, token window, etc. [here](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json).

| Model Name     | Function Call                                                |
|----------------|--------------------------------------------------------------|
| Codestral Latest  | `completion(model="text-completion-codestral/codestral-latest", messages)` |
| Codestral 2405 | `completion(model="text-completion-codestral/codestral-2405", messages)`|




## Chat Completions

:::info

Official Mistral API Docs: https://docs.mistral.ai/api/#operation/createChatCompletion
:::


<Tabs>
<TabItem value="no-streaming" label="No Streaming">

#### Sample Usage

```python
import os
import litellm

os.environ['CODESTRAL_API_KEY']

response = await litellm.acompletion(
    model="codestral/codestral-latest",
    messages=[
        {
            "role": "user",
            "content": "Hey, how's it going?",
        }
    ],
    temperature=0.0,       # optional
    top_p=1,               # optional
    max_tokens=10,         # optional
    safe_prompt=False,     # optional
    seed=12,               # optional
)
```

#### Expected Response

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "codestral/codestral-latest",
  "system_fingerprint": None,
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "\n\nHello there, how may I assist you today?",
    },
    "logprobs": null,
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 9,
    "completion_tokens": 12,
    "total_tokens": 21
  }
}


```


</TabItem>
<TabItem value="stream" label="Streaming">

#### Sample Usage - Streaming

```python
import os
import litellm

os.environ['CODESTRAL_API_KEY']

response = await litellm.acompletion(
    model="codestral/codestral-latest",
    messages=[
        {
            "role": "user",
            "content": "Hey, how's it going?",
        }
    ],
    stream=True,           # optional
    temperature=0.0,       # optional
    top_p=1,               # optional
    max_tokens=10,         # optional
    safe_prompt=False,     # optional
    seed=12,               # optional
)
async for chunk in response:
    print(chunk)
```

#### Expected Response

```json
{
    "id":"chatcmpl-123",
    "object":"chat.completion.chunk",
    "created":1694268190,
    "model": "codestral/codestral-latest",
    "system_fingerprint": None, 
    "choices":[
        {
            "index":0,
            "delta":{"role":"assistant","content":"gm"},
            "logprobs":null,
        "   finish_reason":null
        }
    ]
}

```
</TabItem>
</Tabs>

### Supported Models
All models listed here https://docs.mistral.ai/platform/endpoints are supported. We actively maintain the list of models, pricing, token window, etc. [here](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json).

| Model Name     | Function Call                                                |
|----------------|--------------------------------------------------------------|
| Codestral Latest  | `completion(model="codestral/codestral-latest", messages)` |
| Codestral 2405 | `completion(model="codestral/codestral-2405", messages)`|