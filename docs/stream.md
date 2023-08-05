# Streaming Responses & Async Completion

- [Streaming Responses](#streaming-responses)
- [Async Completion](#async-completion)

## Streaming Responses
LiteLLM supports streaming the model response back by passing `stream=True` as an argument to the completion function
### Usage
```python
response = completion(model="gpt-3.5-turbo", messages=messages, stream=True)
for chunk in response:
    print(chunk['choices'][0]['delta'])

```

## Async Completion
Asynchronous Completion with LiteLLM
LiteLLM provides an asynchronous version of the completion function called `acompletion`
### Usage
```
from litellm import acompletion
import asyncio

async def test_get_response():
    user_message = "Hello, how are you?"
    messages = [{"content": user_message, "role": "user"}]
    response = await acompletion(model="gpt-3.5-turbo", messages=messages)
    return response

response = asyncio.run(test_get_response())
print(response)

```