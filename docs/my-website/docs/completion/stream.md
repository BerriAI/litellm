# Streaming + Async

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

## Async Streaming
We've implemented an `__anext__()` function in the streaming object returned. This enables async iteration over the streaming object. 
### Usage
```
from litellm import acompletion
import asyncio

def logger_fn(model_call_object: dict):
    print(f"LOGGER FUNCTION: {model_call_object}")


user_message = "Hello, how are you?"
messages = [{"content": user_message, "role": "user"}]

# # test on ai21 completion call
async def ai21_async_completion_call():
    try:
        response = completion(
            model="j2-ultra", messages=messages, stream=True, logger_fn=logger_fn
        )
        print(f"response: {response}")
        complete_response = ""
        start_time = time.time()
        # Change for loop to async for loop
        async for chunk in response:
            chunk_time = time.time()
            print(f"time since initial request: {chunk_time - start_time:.5f}")
            print(chunk["choices"][0]["delta"])
            complete_response += chunk["choices"][0]["delta"]["content"]
        if complete_response == "": 
            raise Exception("Empty response received")
    except:
        print(f"error occurred: {traceback.format_exc()}")
        pass

asyncio.run(ai21_async_completion_call())
```