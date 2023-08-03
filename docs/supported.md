## Generation/Completion/Chat Completion Models

### OpenAI Chat Completion Models

| Model Name       | Function Call                          | Required OS Variables                |
|------------------|----------------------------------------|--------------------------------------|
| gpt-3.5-turbo    | `completion('gpt-3.5-turbo', messages)` | `os.environ['OPENAI_API_KEY']`       |
| gpt-3.5-turbo-16k    | `completion('gpt-3.5-turbo-16k', messages)` | `os.environ['OPENAI_API_KEY']`       |
| gpt-3.5-turbo-16k-0613    | `completion('gpt-3.5-turbo-16k-0613', messages)` | `os.environ['OPENAI_API_KEY']`       |
| gpt-4            | `completion('gpt-4', messages)`         | `os.environ['OPENAI_API_KEY']`       |

## Azure OpenAI Chat Completion Models

| Model Name       | Function Call                           | Required OS Variables                     |
|------------------|-----------------------------------------|-------------------------------------------|
| gpt-3.5-turbo    | `completion('gpt-3.5-turbo', messages, azure=True)` | `os.environ['AZURE_API_KEY']`,<br>`os.environ['AZURE_API_BASE']`,<br>`os.environ['AZURE_API_VERSION']` |
| gpt-4            | `completion('gpt-4', messages, azure=True)`         | `os.environ['AZURE_API_KEY']`,<br>`os.environ['AZURE_API_BASE']`,<br>`os.environ['AZURE_API_VERSION']` |

### OpenAI Text Completion Models

| Model Name       | Function Call                              | Required OS Variables                |
|------------------|--------------------------------------------|--------------------------------------|
| text-davinci-003 | `completion('text-davinci-003', messages)` | `os.environ['OPENAI_API_KEY']`       |

### Cohere Models

| Model Name       | Function Call                              | Required OS Variables                |
|------------------|--------------------------------------------|--------------------------------------|
| command-nightly  | `completion('command-nightly', messages)` | `os.environ['COHERE_API_KEY']`       |


### Anthropic Models

| Model Name       | Function Call                              | Required OS Variables                |
|------------------|--------------------------------------------|--------------------------------------|
| claude-instant-1  | `completion('claude-instant-1', messages)` | `os.environ['ANTHROPIC_API_KEY']`       |
| claude-v2  | `completion('claude-v2', messages)` | `os.environ['ANTHROPIC_API_KEY']`       |

