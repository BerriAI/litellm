# Mock Requests

For testing purposes, you can use `mock_completion()` to mock calling the completion endpoint. 

This will return a response object with a default response (works for streaming as well), without calling the LLM APIs. 

## quick start
```python
from litellm import mock_completion 

model = "gpt-3.5-turbo"
messages = [{"role":"user", "content":"This is a test request"}]

mock_completion(model=model, messages=messages)
```

## streaming

```python
model = "gpt-3.5-turbo"
messages = [{"role": "user", "content": "Hey, I'm a mock request"}]
response = litellm.mock_completion(model=model, messages=messages, stream=True)
complete_response = "" 
for chunk in response: 
    print(chunk) # {'choices': [{'delta': {'role': 'assistant', 'content': 'Thi'}, 'finish_reason': None}]}
    complete_response += chunk["choices"][0]["delta"]["content"]
if complete_response == "": 
    raise Exception("Empty response received")
```

## set mock response
You can also customize the mock response text returned. By default it's set to - `This is a mock request`. But you can override this with `mock_response`. 

```python
model = "gpt-3.5-turbo"
messages = [{"role": "user", "content": "Hey, I'm a mock request"}]
response = litellm.mock_completion(model=model, messages=messages, mock_response="My custom mock response", stream=True)
complete_response = "" 
for chunk in response: 
    print(chunk) # {'choices': [{'delta': {'role': 'assistant', 'content': 'My '}, 'finish_reason': None}]}
    complete_response += chunk["choices"][0]["delta"]["content"]
if complete_response == "": 
    raise Exception("Empty response received")
```

## (Non-streaming) Mock Response Object 

```
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "This is a mock request",
        "role": "assistant",
        "logprobs": null
      }
    }
  ],
  "created": 1694459929.4496052,
  "model": "MockResponse",
  "usage": {
    "prompt_tokens": null,
    "completion_tokens": null,
    "total_tokens": null
  }
}
```