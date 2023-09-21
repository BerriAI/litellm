# OpenRouter
LiteLLM supports all the text models from [OpenRouter](https://openrouter.ai/docs)

## Usage
```python
import os
from litellm import completion
os.environ["OPENROUTER_API_KEY"] = ""

response = completion(
            model="google/palm-2-chat-bison",
            messages=messages,
        )
```

### OpenRouter Completion Models

| Model Name       | Function Call                              | Required OS Variables                |
|------------------|--------------------------------------------|--------------------------------------|
| openai/gpt-3.5-turbo | `completion('openai/gpt-3.5-turbo', messages)` | `os.environ['OR_SITE_URL']`,`os.environ['OR_APP_NAME']`,`os.environ['OPENROUTER_API_KEY']`       |
| openai/gpt-3.5-turbo-16k | `completion('openai/gpt-3.5-turbo-16k', messages)` | `os.environ['OR_SITE_URL']`,`os.environ['OR_APP_NAME']`,`os.environ['OPENROUTER_API_KEY']`       |
| openai/gpt-4 | `completion('openai/gpt-4', messages)` | `os.environ['OR_SITE_URL']`,`os.environ['OR_APP_NAME']`,`os.environ['OPENROUTER_API_KEY']`       |
| openai/gpt-4-32k | `completion('openai/gpt-4-32k', messages)` | `os.environ['OR_SITE_URL']`,`os.environ['OR_APP_NAME']`,`os.environ['OPENROUTER_API_KEY']`       |
| anthropic/claude-2 | `completion('anthropic/claude-2', messages)` | `os.environ['OR_SITE_URL']`,`os.environ['OR_APP_NAME']`,`os.environ['OPENROUTER_API_KEY']`       |
| anthropic/claude-instant-v1 | `completion('anthropic/claude-instant-v1', messages)` | `os.environ['OR_SITE_URL']`,`os.environ['OR_APP_NAME']`,`os.environ['OPENROUTER_API_KEY']`       |
| google/palm-2-chat-bison | `completion('google/palm-2-chat-bison', messages)` | `os.environ['OR_SITE_URL']`,`os.environ['OR_APP_NAME']`,`os.environ['OPENROUTER_API_KEY']`       |
| google/palm-2-codechat-bison | `completion('google/palm-2-codechat-bison', messages)` | `os.environ['OR_SITE_URL']`,`os.environ['OR_APP_NAME']`,`os.environ['OPENROUTER_API_KEY']`       |
| meta-llama/llama-2-13b-chat | `completion('meta-llama/llama-2-13b-chat', messages)` | `os.environ['OR_SITE_URL']`,`os.environ['OR_APP_NAME']`,`os.environ['OPENROUTER_API_KEY']`       |
| meta-llama/llama-2-70b-chat | `completion('meta-llama/llama-2-70b-chat', messages)` | `os.environ['OR_SITE_URL']`,`os.environ['OR_APP_NAME']`,`os.environ['OPENROUTER_API_KEY']`       |
