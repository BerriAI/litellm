# Custom Auth 

You can now override the default api key auth.

## Usage

#### 1. Create a custom auth file. 

Make sure the response type follows the `UserAPIKeyAuth` pydantic object. This is used by for logging usage specific to that user key.

```python
from litellm.proxy._types import UserAPIKeyAuth

async def user_api_key_auth(request: Request, api_key: str) -> UserAPIKeyAuth: 
    try: 
        modified_master_key = "sk-my-master-key"
        if api_key == modified_master_key:
            return UserAPIKeyAuth(api_key=api_key)
        raise Exception
    except: 
        raise Exception
```

#### 2. Pass the filepath (relative to the config.yaml)

Pass the filepath to the config.yaml 

e.g. if they're both in the same dir - `./config.yaml` and `./custom_auth.py`, this is what it looks like:
```yaml 
model_list: 
  - model_name: "openai-model"
    litellm_params: 
      model: "gpt-3.5-turbo"

litellm_settings:
  drop_params: True
  set_verbose: True

general_settings:
  custom_auth: custom_auth.user_api_key_auth
```

[**Implementation Code**](https://github.com/BerriAI/litellm/blob/caf2a6b279ddbe89ebd1d8f4499f65715d684851/litellm/proxy/utils.py#L122)

#### 3. Start the proxy
```shell
$ litellm --config /path/to/config.yaml 
```

## ✨ Support LiteLLM Virtual Keys + Custom Auth

Supported from v1.72.2+

:::info 

✨ Supporting Custom Auth + LiteLLM Virtual Keys is on LiteLLM Enterprise

[Enterprise Pricing](https://www.litellm.ai/#pricing)

[Get free 7-day trial key](https://www.litellm.ai/enterprise#trial)
:::

### Usage

1. Setup custom auth file

```python
"""
Example custom auth function.

This will allow all keys starting with "my-custom-key" to pass through.
"""
from typing import Union

from fastapi import Request

from litellm.proxy._types import UserAPIKeyAuth


async def user_api_key_auth(
    request: Request, api_key: str
) -> Union[UserAPIKeyAuth, str]:
    try:
        if api_key.startswith("my-custom-key"):
            return "sk-P1zJMdsqCPNN54alZd_ETw"
        else:
            raise Exception("Invalid API key")
    except Exception:
        raise Exception("Invalid API key")

```

2. Setup config.yaml

Key change set `mode: auto`. This will check both litellm api key auth + custom auth.

```yaml
model_list: 
  - model_name: "openai-model"
    litellm_params: 
      model: "gpt-3.5-turbo"
      api_key: os.environ/OPENAI_API_KEY

general_settings:
  custom_auth: custom_auth_auto.user_api_key_auth
  custom_auth_settings:
    mode: "auto" # can be 'on', 'off', 'auto' - 'auto' checks both litellm api key auth + custom auth
```

Flow:
1. Checks custom auth first
2. If custom auth fails, checks litellm api key auth
3. If both fail, returns 401


3. Test it! 

```bash
curl -L -X POST 'http://0.0.0.0:4000/v1/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-P1zJMdsqCPNN54alZd_ETw' \
-d '{
    "model": "openai-model",
    "messages": [
          {
            "role": "user",
            "content": "Hey! My name is John"
          }
        ]
}'
```




#### Bubble up custom exceptions

If you want to bubble up custom exceptions, you can do so by raising a `ProxyException`.

```python
"""
Example custom auth function.

This will allow all keys starting with "my-custom-key" to pass through.
"""

from typing import Union

from fastapi import Request

from litellm.proxy._types import UserAPIKeyAuth, ProxyException


async def user_api_key_auth(
    request: Request, api_key: str
) -> Union[UserAPIKeyAuth, str]:
    try:
        if api_key.startswith("my-custom-key"):
            return "sk-P1zJMdsqCPNN54alZd_ETw"
        if api_key == "invalid-api-key":
            # raise a custom exception back to the client
            raise ProxyException(
                message="Invalid API key",
                type="invalid_request_error",
                param="api_key",
                code=401,
            )
        else:
            raise Exception("Invalid API key")
    except Exception:
        raise Exception("Invalid API key")

```