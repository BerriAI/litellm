## Generation/Completion/Chat Completion Models

### OpenAI Chat Completion Models

| Model Name       | Function Call                          | Required OS Variables                |
|------------------|----------------------------------------|--------------------------------------|
| gpt-3.5-turbo    | `completion('gpt-3.5-turbo', messages)` | `os.environ['OPENAI_API_KEY']`       |
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

### OpenRouter Models

| Model Name                       | Function Call                                                        | Required OS Variables                                                     |
|----------------------------------|----------------------------------------------------------------------|---------------------------------------------------------------------------|
| google/palm-2-codechat-bison     | `completion('google/palm-2-codechat-bison', messages)`                | `os.environ['OPENROUTER_API_KEY']`,<br>`os.environ['OR_SITE_URL']`,<br>`os.environ['OR_APP_NAME']` |
| google/palm-2-chat-bison         | `completion('google/palm-2-chat-bison', messages)`                    | `os.environ['OPENROUTER_API_KEY']`,<br>`os.environ['OR_SITE_URL']`,<br>`os.environ['OR_APP_NAME']` |
| openai/gpt-3.5-turbo             | `completion('openai/gpt-3.5-turbo', messages)`                        | `os.environ['OPENROUTER_API_KEY']`,<br>`os.environ['OR_SITE_URL']`,<br>`os.environ['OR_APP_NAME']` |
| openai/gpt-3.5-turbo-16k         | `completion('openai/gpt-3.5-turbo-16k', messages)`                    | `os.environ['OPENROUTER_API_KEY']`,<br>`os.environ['OR_SITE_URL']`,<br>`os.environ['OR_APP_NAME']` |
| openai/gpt-4-32k                 | `completion('openai/gpt-4-32k', messages)`                            | `os.environ['OPENROUTER_API_KEY']`,<br>`os.environ['OR_SITE_URL']`,<br>`os.environ['OR_APP_NAME']` |
| anthropic/claude-2               | `completion('anthropic/claude-2', messages)`                          | `os.environ['OPENROUTER_API_KEY']`,<br>`os.environ['OR_SITE_URL']`,<br>`os.environ['OR_APP_NAME']` |
| anthropic/claude-instant-v1      | `completion('anthropic/claude-instant-v1', messages)`                 | `os.environ['OPENROUTER_API_KEY']`,<br>`os.environ['OR_SITE_URL']`,<br>`os.environ['OR_APP_NAME']` |
| meta-llama/llama-2-13b-chat      | `completion('meta-llama/llama-2-13b-chat', messages)`                 | `os.environ['OPENROUTER_API_KEY']`,<br>`os.environ['OR_SITE_URL']`,<br>`os.environ['OR_APP_NAME']` |
| meta-llama/llama-2-70b-chat      | `completion('meta-llama/llama-2-70b-chat', messages)`                 | `os.environ['OPENROUTER_API_KEY']`,<br>`os.environ['OR_SITE_URL']`,<br>`os.environ['OR_APP_NAME']` |
