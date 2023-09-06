# Batching Completion() Calls 

In the batch_completion method, you provide a list of `messages` where each sub-list of messages is passed to `litellm.completion()`, allowing you to process multiple prompts efficiently in a single API call.

<a target="_blank" href="https://colab.research.google.com/github/BerriAI/litellm/blob/main/cookbook/LiteLLM_batch_completion.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Example Code
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