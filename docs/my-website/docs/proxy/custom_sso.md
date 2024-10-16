# Event Hook for SSO Login (Custom Handler)

Use this if you want to run your own code after a user signs on to the LiteLLM UI using SSO

## How it works
- User lands on Admin UI
- LiteLLM redirects user to your SSO provider
- Your SSO provider redirects user back to LiteLLM
- LiteLLM has retrieved user information from your IDP
- **Your custom SSO handler is called and returns an object of type SSOUserDefinedValues**
- User signed in to UI

## Usage

#### 1. Create a custom sso handler file. 

Make sure the response type follows the `SSOUserDefinedValues` pydantic object. This is used for logging the user into the Admin UI

```python
from fastapi import Request
from fastapi_sso.sso.base import OpenID

from litellm.proxy._types import LitellmUserRoles, SSOUserDefinedValues
from litellm.proxy.management_endpoints.internal_user_endpoints import (
    new_user,
    user_info,
)
from litellm.proxy.management_endpoints.team_endpoints import add_new_member


async def custom_sso_handler(userIDPInfo: OpenID) -> SSOUserDefinedValues:
    try:
        print("inside custom sso handler")  # noqa
        print(f"userIDPInfo: {userIDPInfo}")  # noqa

        if userIDPInfo.id is None:
            raise ValueError(
                f"No ID found for user. userIDPInfo.id is None {userIDPInfo}"
            )
        

        #################################################
        # Run you custom code / logic here
        # check if user exists in litellm proxy DB
        _user_info = await user_info(user_id=userIDPInfo.id)
        print("_user_info from litellm DB ", _user_info)  # noqa
        #################################################

        return SSOUserDefinedValues(
            models=[],                                      # models user has access to
            user_id=userIDPInfo.id,                         # user id to use in the LiteLLM DB
            user_email=userIDPInfo.email,                   # user email to use in the LiteLLM DB
            user_role=LitellmUserRoles.INTERNAL_USER.value, # role to use for the user 
            max_budget=0.01,                                # Max budget for this UI login Session
            budget_duration="1d",                           # Duration of the budget for this UI login Session, 1d, 2d, 30d ...
        )
    except Exception as e:
        raise Exception("Failed custom auth")
```

#### 2. Pass the filepath (relative to the config.yaml)

Pass the filepath to the config.yaml 

e.g. if they're both in the same dir - `./config.yaml` and `./custom_sso.py`, this is what it looks like:
```yaml 
model_list: 
  - model_name: "openai-model"
    litellm_params: 
      model: "gpt-3.5-turbo"

litellm_settings:
  drop_params: True
  set_verbose: True

general_settings:
  custom_sso: custom_sso.custom_sso_handler
```

#### 3. Start the proxy
```shell
$ litellm --config /path/to/config.yaml 
```
