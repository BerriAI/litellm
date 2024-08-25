import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Streaming + Async

- [Streaming Responses](#streaming-responses)
- [Async Completion](#async-completion)
- [Async + Streaming Completion](#async-streaming)

## Streaming Responses
LiteLLM supports streaming the model response back by passing `stream=True` as an argument to the completion function
### Usage
```python
from litellm import completion
messages = [{"role": "user", "content": "Hey, how's it going?"}]
response = completion(model="gpt-3.5-turbo", messages=messages, stream=True)
for part in response:
    print(part.choices[0].delta.content or "")
```

### Helper function

LiteLLM also exposes a helper function to rebuild the complete streaming response from the list of chunks. 

```python
from litellm import completion
messages = [{"role": "user", "content": "Hey, how's it going?"}]
response = completion(model="gpt-3.5-turbo", messages=messages, stream=True)

for chunk in response: 
    chunks.append(chunk)

print(litellm.stream_chunk_builder(chunks, messages=messages))
```

## Async Completion
Asynchronous Completion with LiteLLM. LiteLLM provides an asynchronous version of the completion function called `acompletion`
### Usage
```python
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
Here's an example of using it with openai.
```python
from litellm import acompletion
import asyncio, os, traceback

async def completion_call():
    try:
        print("test acompletion + streaming")
        response = await acompletion(
            model="gpt-3.5-turbo", 
            messages=[{"content": "Hello, how are you?", "role": "user"}], 
            stream=True
        )
        print(f"response: {response}")
        async for chunk in response:
            print(chunk)
    except:
        print(f"error occurred: {traceback.format_exc()}")
        pass

asyncio.run(completion_call())
```

## Error Handling - Infinite Loops

Sometimes a model might enter an infinite loop, and keep repeating the same chunks - [e.g. issue](https://github.com/BerriAI/litellm/issues/5158)

Break out of it with: 

```python
litellm.REPEATED_STREAMING_CHUNK_LIMIT = 100 # # catch if model starts looping the same chunk while streaming. Uses high default to prevent false positives.
```

LiteLLM provides error handling for this, by checking if a chunk is repeated 'n' times (Default is 100). If it exceeds that limit, it will raise a `litellm.InternalServerError`, to allow retry logic to happen. 

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import litellm 
import os 

litellm.set_verbose = False
loop_amount = litellm.REPEATED_STREAMING_CHUNK_LIMIT + 1
chunks = [
    litellm.ModelResponse(**{
    "id": "chatcmpl-123",
    "object": "chat.completion.chunk",
    "created": 1694268190,
    "model": "gpt-3.5-turbo-0125",
    "system_fingerprint": "fp_44709d6fcb",
    "choices": [
        {"index": 0, "delta": {"content": "How are you?"}, "finish_reason": "stop"}
    ],
}, stream=True)
] * loop_amount
completion_stream = litellm.ModelResponseListIterator(model_responses=chunks)

response = litellm.CustomStreamWrapper(
    completion_stream=completion_stream,
    model="gpt-3.5-turbo",
    custom_llm_provider="cached_response",
    logging_obj=litellm.Logging(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hey"}],
        stream=True,
        call_type="completion",
        start_time=time.time(),
        litellm_call_id="12345",
        function_id="1245",
    ),
)

for chunk in response:
    continue # expect to raise InternalServerError 
```

</TabItem>
<TabItem value="proxy" label="PROXY">

Define this on your config.yaml on the proxy. 

```yaml
litellm_settings:
    REPEATED_STREAMING_CHUNK_LIMIT: 100 # this overrides the litellm default
```

The proxy uses the litellm SDK. To validate this works, try the 'SDK' code snippet. 

</TabItem>
</Tabs>