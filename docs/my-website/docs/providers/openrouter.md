# OpenRouter
LiteLLM supports all the text models from [OpenRouter](https://openrouter.ai/docs)

## Usage
```python
import os
from litellm import completion
os.environ["OPENROUTER_API_KEY"] = ""

response = completion(
            model="openrouter/google/palm-2-chat-bison",
            messages=messages,
        )
```

### OpenRouter Completion Models

| Model Name                | Function Call                                       | Required OS Variables                                        |
|---------------------------|-----------------------------------------------------|--------------------------------------------------------------|
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

