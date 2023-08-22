Using completion() with Fallbacks for Reliability

This tutorial demonstrates how to employ the `completion()` function with model fallbacks to ensure reliability. LLM APIs can be unstable, completion() with fallbacks ensures you'll always get a response from your calls

## Usage 
To use fallback models with `completion()`, specify a list of models in the `fallbacks` parameter. 

Example
```python
try:
    response = completion(
        model="bad-model",
        messages=messages,
        fallbacks=["gpt-3.5-turbo", "command-nightly"]
    )
```

Output
```
Completion with 'bad-model': got exception Unable to map your input to a model. Check your input - {'model': 'bad-model'



completion call gpt-3.5-turbo
{
  "id": "chatcmpl-7qTmVRuO3m3gIBg4aTmAumV1TmQhB",
  "object": "chat.completion",
  "created": 1692741891,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "I apologize, but as an AI, I do not have the capability to provide real-time weather updates. However, you can easily check the current weather in San Francisco by using a search engine or checking a weather website or app."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 16,
    "completion_tokens": 46,
    "total_tokens": 62
  }
}

```
