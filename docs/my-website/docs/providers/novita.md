# Novita AI
LiteLLM supports all models from [Novita AI](https://novita.ai/models/llm?utm_source=github_litellm&utm_medium=github_readme&utm_campaign=github_link)

## Usage
```python
import os
from litellm import completion
os.environ["NOVITA_API_KEY"] = ""

response = completion(
            model="meta-llama/llama-3.3-70b-instruct",
            messages=messages,
        )
```

## Novita AI Completion Models

🚨 LiteLLM supports ALL Novita AI models, send `model=novita/<your-novita-model>` to send it to Novita AI. See all Novita AI models [here](https://novita.ai/models/llm?utm_source=github_litellm&utm_medium=github_readme&utm_campaign=github_link)

| Model Name                | Function Call                                       |
|---------------------------|-----------------------------------------------------|
| novita/deepseek/deepseek-r1 | `completion('novita/deepseek/deepseek-r1', messages)` | `os.environ['NOVITA_API_KEY']` |
| novita/deepseek/deepseek_v3 | `completion('novita/deepseek/deepseek_v3', messages)` | `os.environ['NOVITA_API_KEY']` |
| novita/meta-llama/llama-3.3-70b-instruct | `completion('novita/meta-llama/llama-3.3-70b-instruct', messages)` | `os.environ['NOVITA_API_KEY']` |
| novita/meta-llama/llama-3.1-8b-instruct | `completion('novita/meta-llama/llama-3.1-8b-instruct', messages)` | `os.environ['NOVITA_API_KEY']` |
| novita/meta-llama/llama-3.1-8b-instruct-max | `completion('novita/meta-llama/llama-3.1-8b-instruct-max', messages)` | `os.environ['NOVITA_API_KEY']` |
| novita/meta-llama/llama-3.1-70b-instruct | `completion('novita/meta-llama/llama-3.1-70b-instruct', messages)` | `os.environ['NOVITA_API_KEY']` |
| novita/meta-llama/llama-3-8b-instruct | `completion('novita/meta-llama/llama-3-8b-instruct', messages)` | `os.environ['NOVITA_API_KEY']` |
| novita/meta-llama/llama-3-70b-instruct | `completion('novita/meta-llama/llama-3-70b-instruct', messages)` | `os.environ['NOVITA_API_KEY']` |
| novita/meta-llama/llama-3.2-1b-instruct | `completion('novita/meta-llama/llama-3.2-1b-instruct', messages)` | `os.environ['NOVITA_API_KEY']` |
| novita/meta-llama/llama-3.2-11b-vision-instruct | `completion('novita/meta-llama/llama-3.2-11b-vision-instruct', messages)` | `os.environ['NOVITA_API_KEY']` |
| novita/meta-llama/llama-3.2-3b-instruct | `completion('novita/meta-llama/llama-3.2-3b-instruct', messages)` | `os.environ['NOVITA_API_KEY']` |
| novita/gryphe/mythomax-l2-13b | `completion('novita/gryphe/mythomax-l2-13b', messages)` | `os.environ['NOVITA_API_KEY']` |
| novita/google/gemma-2-9b-it | `completion('novita/google/gemma-2-9b-it', messages)` | `os.environ['NOVITA_API_KEY']` |
| novita/mistralai/mistral-nemo | `completion('novita/mistralai/mistral-nemo', messages)` | `os.environ['NOVITA_API_KEY']` |
| novita/mistralai/mistral-7b-instruct | `completion('novita/mistralai/mistral-7b-instruct', messages)` | `os.environ['NOVITA_API_KEY']` |
| novita/qwen/qwen-2.5-72b-instruct | `completion('novita/qwen/qwen-2.5-72b-instruct', messages)` | `os.environ['NOVITA_API_KEY']` |
| novita/qwen/qwen-2-vl-72b-instruct | `completion('novita/qwen/qwen-2-vl-72b-instruct', messages)` | `os.environ['NOVITA_API_KEY']` |

