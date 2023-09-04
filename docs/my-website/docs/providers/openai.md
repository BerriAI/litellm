# OpenAI
LiteLLM supports OpenAI Chat + Text completion and embedding calls.

### API KEYS
```
import os 

os.environ["OPENAI_API_KEY"] = ""
```

### OpenAI Chat Completion Models

| Model Name       | Function Call                          | Required OS Variables                |
|------------------|----------------------------------------|--------------------------------------|
| gpt-3.5-turbo    | `completion('gpt-3.5-turbo', messages)` | `os.environ['OPENAI_API_KEY']`       |
| gpt-3.5-turbo-0301    | `completion('gpt-3.5-turbo-0301', messages)` | `os.environ['OPENAI_API_KEY']`       |
| gpt-3.5-turbo-0613    | `completion('gpt-3.5-turbo-0613', messages)` | `os.environ['OPENAI_API_KEY']`       |
| gpt-3.5-turbo-16k    | `completion('gpt-3.5-turbo-16k', messages)` | `os.environ['OPENAI_API_KEY']`       |
| gpt-3.5-turbo-16k-0613    | `completion('gpt-3.5-turbo-16k-0613', messages)` | `os.environ['OPENAI_API_KEY']`       |
| gpt-4            | `completion('gpt-4', messages)`         | `os.environ['OPENAI_API_KEY']`       |
| gpt-4-0314            | `completion('gpt-4-0314', messages)`         | `os.environ['OPENAI_API_KEY']`       |
| gpt-4-0613            | `completion('gpt-4-0613', messages)`         | `os.environ['OPENAI_API_KEY']`       |
| gpt-4-32k            | `completion('gpt-4-32k', messages)`         | `os.environ['OPENAI_API_KEY']`       |
| gpt-4-32k-0314            | `completion('gpt-4-32k-0314', messages)`         | `os.environ['OPENAI_API_KEY']`       |
| gpt-4-32k-0613            | `completion('gpt-4-32k-0613', messages)`         | `os.environ['OPENAI_API_KEY']`       |

These also support the `OPENAI_API_BASE` environment variable, which can be used to specify a custom API endpoint.

### OpenAI Text Completion Models

| Model Name       | Function Call                              | Required OS Variables                |
|------------------|--------------------------------------------|--------------------------------------|
| text-davinci-003 | `completion('text-davinci-003', messages)` | `os.environ['OPENAI_API_KEY']`       |
| ada-001 | `completion('ada-001', messages)` | `os.environ['OPENAI_API_KEY']`       |
| curie-001 | `completion('curie-001', messages)` | `os.environ['OPENAI_API_KEY']`       |
| babbage-001 | `completion('babbage-001', messages)` | `os.environ['OPENAI_API_KEY']`       |
| babbage-002 | `completion('ada-001', messages)` | `os.environ['OPENAI_API_KEY']`       |
| davinci-002 | `completion('davinci-002', messages)` | `os.environ['OPENAI_API_KEY']`       |