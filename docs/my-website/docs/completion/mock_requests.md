# Mock Completion() Responses - Save Testing Costs 💰

For testing purposes, you can use `completion()` with `mock_response` to mock calling the completion endpoint. 

This will return a response object with a default response (works for streaming as well), without calling the LLM APIs. 

## quick start
```python
from litellm import completion 

model = "gpt-3.5-turbo"
messages = [{"role":"user", "content":"This is a test request"}]

completion(model=model, messages=messages, mock_response="It's simple to use and easy to get started")
```

## streaming

```python
from litellm import completion 
model = "gpt-3.5-turbo"
messages = [{"role": "user", "content": "Hey, I'm a mock request"}]
response = completion(model=model, messages=messages, stream=True, mock_response="It's simple to use and easy to get started")
for chunk in response: 
    print(chunk) # {'choices': [{'delta': {'role': 'assistant', 'content': 'Thi'}, 'finish_reason': None}]}
    complete_response += chunk["choices"][0]["delta"]["content"]
```

## (Non-streaming) Mock Response Object 

```json
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

## Building a pytest function using `completion` with `mock_response`

```python
from litellm import completion
import pytest

def test_completion_openai():
    try:
        response = completion(
            model="gpt-3.5-turbo",
            messages=[{"role":"user", "content":"Why is LiteLLM amazing?"}],
            mock_response="LiteLLM is awesome"
        )
        # Add any assertions here to check the response
        print(response)
        assert(response['choices'][0]['message']['content'] == "LiteLLM is awesome")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
```