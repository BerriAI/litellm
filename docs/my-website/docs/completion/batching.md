# Batching Completion() Calls 

In the batch_completion method, you provide a list of `messages` where each sub-list of messages is passed to `litellm.completion()`, allowing you to process multiple prompts efficiently in a single API call.

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