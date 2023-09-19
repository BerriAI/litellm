# Batching Completion() Calls 
LiteLLM allows you to:
* Send multiple completion calls to 1 model
* Send 1 completion call to N models

## Send multiple completion calls to 1 model

In the batch_completion method, you provide a list of `messages` where each sub-list of messages is passed to `litellm.completion()`, allowing you to process multiple prompts efficiently in a single API call.

<a target="_blank" href="https://colab.research.google.com/github/BerriAI/litellm/blob/main/cookbook/LiteLLM_batch_completion.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Example Code
```python
import litellm
import os
from litellm import batch_completion

os.environ['ANTHROPIC_API_KEY'] = ""


responses = batch_completion(
    model="claude-2",
    messages = [
        [
            {
                "role": "user",
                "content": "good morning? "
            }
        ],
        [
            {
                "role": "user",
                "content": "what's the time? "
            }
        ]
    ]
)
```

## Send 1 completion call to N models
This makes parallel calls to the specified `models` and returns the first response 

Use this to reduce latency

### Example Code
```python
import litellm
import os
from litellm import batch_completion_models

os.environ['ANTHROPIC_API_KEY'] = ""
os.environ['OPENAI_API_KEY'] = ""
os.environ['COHERE_API_KEY'] = ""

response = batch_completion_models(
    models=["gpt-3.5-turbo", "claude-instant-1.2", "command-nightly"], 
    messages=[{"role": "user", "content": "Hey, how's it going"}]
)
print(result)
```

### Output
Returns the first response
```json
{
  "object": "chat.completion",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": " I'm doing well, thanks for asking! I'm an AI assistant created by Anthropic to be helpful, harmless, and honest.",
        "role": "assistant",
        "logprobs": null
      }
    }
  ],
  "id": "chatcmpl-23273eed-e351-41be-a492-bafcf5cf3274",
  "created": 1695154628.2076092,
  "model": "command-nightly",
  "usage": {
    "prompt_tokens": 6,
    "completion_tokens": 14,
    "total_tokens": 20
  }
}
```