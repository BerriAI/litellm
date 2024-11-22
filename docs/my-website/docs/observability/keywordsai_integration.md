# 🗝️ Keywords AI - LLM monitoring platform

:::tip

This is community maintained. Please make an issue if you run into a bug:
https://github.com/BerriAI/litellm

:::

[Keywords AI](https://keywordsai.co/) makes it easy for developers to build LLM applications. With 2 lines of code, developers get a complete monitoring platform that speeds up deploying & monitoring AI apps in production.

## Using Keywords AI with LiteLLM

LiteLLM provides two methods to integrate with Keywords AI:
1. Using callbacks (recommended)
2. Using Keywords AI as a proxy

### Approach 1: Using Callbacks (Recommended)

This method works across all LiteLLM-supported models. Simply set Keywords AI as a success callback:

```python
import litellm
from litellm import completion
import os
# Set up Keywords AI callback
os.environ["KEYWORDSAI_API_KEY"]="YOUR_KEYWORDSAI_API_KEY"
litellm.success_callback = ["keywordsai"]

# Optional: Add additional logging parameters
extra_params = {
    "keywordsai_params": {
        "customer_params": {
            "customer_identifier": "your_customer_id",
            "email": "user@example.com",
            "name": "user name"
        },
        "thread_identifier": "thread_123",
        "metadata": {"key": "value"},
        "evaluation_identifier": "eval_123",
        "prompt_id": "prompt_123",
    }
}

# Make completion call with any LiteLLM-supported model
response = completion(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}],
    metadata=extra_params  # Pass additional logging parameters
)

print(response)
```

### Approach 2: Using Keywords AI as a proxy

Alternatively, you can route requests through Keywords AI's API:

```python
import os
import litellm

# Set Keywords AI as the API base
litellm.api_base = "https://api.keywordsai.co/api/"
KEYWORDSAI_API_KEY = os.getenv("KEYWORDSAI_API_KEY")

response = litellm.completion(
    api_key=KEYWORDSAI_API_KEY,  # Use Keywords AI API key
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}]
)

# View logs at https://platform.keywordsai.co/
```
