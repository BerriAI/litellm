# Using Fine-Tuned gpt-3.5-turbo
LiteLLM allows you to call `completion` with your fine-tuned gpt-3.5-turbo models
If you're trying to create your custom finetuned gpt-3.5-turbo model following along on this tutorial: https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset

Once you've created your fine tuned model, you can call it with `completion()` 

## Usage
```python
import os
from litellm import completion
# set your OPENAI key in your .env as "OPENAI_API_KEY"

response = completion(
  model="ft:gpt-3.5-turbo:my-org:custom_suffix:id",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)

print(response.choices[0].message)
```