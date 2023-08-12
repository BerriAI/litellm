# Secret Managers
liteLLM reads secrets from yoour secret manager, .env file 

- [Infisical Secret Manager](#infisical-secret-manager)
- [.env Files](#env-files)

For expected format of secrets see [supported LLM models](https://litellm.readthedocs.io/en/latest/supported)

## Infisical Secret Manager
Integrates with [Infisical's Secret Manager](https://infisical.com/) for secure storage and retrieval of API keys and sensitive data.

### Usage
liteLLM manages reading in your LLM API secrets/env variables from Infisical for you

```
import litellm
from infisical import InfisicalClient

litellm.secret_manager = InfisicalClient(token="your-token")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather like today?"},
]

response = litellm.completion(model="gpt-3.5-turbo", messages=messages)

print(response)
```


## .env Files
If no secret manager client is specified, Litellm automatically uses the `.env` file to manage sensitive data.
