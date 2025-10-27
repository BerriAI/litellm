# OpenRouter
LiteLLM supports all the text / chat / vision models from [OpenRouter](https://openrouter.ai/docs)

<a target="_blank" href="https://colab.research.google.com/github/BerriAI/litellm/blob/main/cookbook/LiteLLM_OpenRouter.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Usage
```python
import os
from litellm import completion

os.environ["OPENROUTER_API_KEY"] = ""
os.environ["OPENROUTER_API_BASE"] = "" # [OPTIONAL] defaults to https://openrouter.ai/api/v1
os.environ["OR_SITE_URL"] = "" # [OPTIONAL]
os.environ["OR_APP_NAME"] = "" # [OPTIONAL]

response = completion(
            model="openrouter/google/palm-2-chat-bison",
            messages=messages,
        )
```

## Configuration with Environment Variables

For production environments, you can dynamically configure the base_url using environment variables:

```python
import os
from litellm import completion

# Configure with environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")

# Set environment for LiteLLM
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
os.environ["OPENROUTER_API_BASE"] = OPENROUTER_BASE_URL

response = completion(
    model="openrouter/google/palm-2-chat-bison",
    messages=messages,
    base_url=OPENROUTER_BASE_URL  # Explicitly pass base_url for clarity
)
```

This approach provides better flexibility for managing configurations across different environments (dev, staging, production) and makes it easier to switch between self-hosted and cloud endpoints.

## OpenRouter Completion Models
🚨 LiteLLM supports ALL OpenRouter models, send `model=openrouter/<your-openrouter-model>` to send it to open router. See all openrouter models [here](https://openrouter.ai/models)

| Model Name                | Function Call                                       |
|---------------------------|-----------------------------------------------------|
| openrouter/openai/gpt-3.5-turbo | `completion('openrouter/openai/gpt-3.5-turbo', messages)` | `os.environ['OR_SITE_URL']`,`os.environ['OR_APP_NAME']`,`os.environ['OPENROUTER_API_KEY']` |
| openrouter/openai/gpt-3.5-turbo-16k | `completion('openrouter/openai/gpt-3.5-turbo-16k', messages)` | `os.environ['OR_SITE_URL']`,`os.environ['OR_APP_NAME']`,`os.environ['OPENROUTER_API_KEY']` |
| openrouter/openai/gpt-4    | `completion('openrouter/openai/gpt-4', messages)`       | `os.environ['OR_SITE_URL']`,`os.environ['OR_APP_NAME']`,`os.environ['OPENROUTER_API_KEY']` |
| openrouter/openai/gpt-4-32k | `completion('openrouter/openai/gpt-4-32k', messages)` | `os.environ['OR_SITE_URL']`,`os.environ['OR_APP_NAME']`,`os.environ['OPENROUTER_API_KEY']` |
| openrouter/anthropic/claude-2 | `completion('openrouter/anthropic/claude-2', messages)` | `os.environ['OR_SITE_URL']`,`os.environ['OR_APP_NAME']`,`os.environ['OPENROUTER_API_KEY']` |
| openrouter/anthropic/claude-instant-v1 | `completion('openrouter/anthropic/claude-instant-v1', messages)` | `os.environ['OR_SITE_URL']`,`os.environ['OR_APP_NAME']`,`os.environ['OPENROUTER_API_KEY']` |
| openrouter/google/palm-2-chat-bison | `completion('openrouter/google/palm-2-chat-bison', messages)` | `os.environ['OR_SITE_URL']`,`os.environ['OR_APP_NAME']`,`os.environ['OPENROUTER_API_KEY']` |
| openrouter/google/palm-2-codechat-bison | `completion('openrouter/google/palm-2-codechat-bison', messages)` | `os.environ['OR_SITE_URL']`,`os.environ['OR_APP_NAME']`,`os.environ['OPENROUTER_API_KEY']` |
| openrouter/meta-llama/llama-2-13b-chat | `completion('openrouter/meta-llama/llama-2-13b-chat', messages)` | `os.environ['OR_SITE_URL']`,`os.environ['OR_APP_NAME']`,`os.environ['OPENROUTER_API_KEY']` |
| openrouter/meta-llama/llama-2-70b-chat | `completion('openrouter/meta-llama/llama-2-70b-chat', messages)` | `os.environ['OR_SITE_URL']`,`os.environ['OR_APP_NAME']`,`os.environ['OPENROUTER_API_KEY']` |

## Passing OpenRouter Params - transforms, models, route
Pass `transforms`, `models`, `route`as arguments to `litellm.completion()`

```python
import os
from litellm import completion

os.environ["OPENROUTER_API_KEY"] = ""

response = completion(
            model="openrouter/google/palm-2-chat-bison",
            messages=messages,
            transforms = [""],
            route= ""
        )
```
