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


### Sample Usage

```python
import os
import litellm

os.environ['CODESTRAL_API_KEY']

response = await litellm.atext_completion(
    model="text-completion-codestral/codestral-2405",
    prompt="def is_odd(n): \n return n % 2 == 1 \ndef test_is_odd():",
    suffix="return True",
    temperature=0,
    top_p=1,
    max_tokens=10,
    min_tokens=10,
    seed=10,
    stop=["return"],
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



### Sample Usage - Streaming

```python
import os
import litellm

os.environ['CODESTRAL_API_KEY']

response = await litellm.atext_completion(
    model="text-completion-codestral/codestral-2405",
    prompt="def is_odd(n): \n return n % 2 == 1 \ndef test_is_odd():",
    suffix="return True",
    temperature=0,
    top_p=1,
    stream=True,
    seed=10,
    stop=["return"],
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

### Supported Models
All models listed here https://docs.mistral.ai/platform/endpoints are supported. We actively maintain the list of models, pricing, token window, etc. [here](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json).

| Model Name     | Function Call                                                |
|----------------|--------------------------------------------------------------|
| Codestral Latest  | `completion(model="text-completion-codestral/codestral-latest", messages)` |
| Codestral 2405 | `completion(model="text-completion-codestral/codestral-2405", messages)`|