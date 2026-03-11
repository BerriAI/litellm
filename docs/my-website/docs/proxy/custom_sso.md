# ✨ Event Hooks for SSO Login

:::info
✨ SSO is free for up to 5 users. After that, an enterprise license is required. [Get Started with Enterprise here](https://www.litellm.ai/enterprise)
:::

## Overview

LiteLLM provides two different SSO hooks depending on your authentication setup:

| Hook Type | When to Use | What It Does |
|-----------|-------------|--------------|
| **Custom UI SSO Sign-in Handler** | You have an OAuth proxy (oauth2-proxy, Gatekeeper, Vouch, etc.) in front of LiteLLM | Parses user info from request headers and signs user into UI |
| **Custom SSO Handler** | You use direct SSO providers (Google, Microsoft, SAML) and want custom post-auth logic | Runs custom code after standard OAuth flow to set user permissions/teams |

**Quick Decision Guide:**
- ✅ **Use Custom UI SSO Sign-in Handler** if user authentication happens outside LiteLLM (via headers)
- ✅ **Use Custom SSO Handler** if you want LiteLLM to handle OAuth flow + run custom logic afterward

---

## Option 1: Custom UI SSO Sign-in Handler

Use this when you have an **OAuth proxy in front of LiteLLM** that has already authenticated the user and passes user information via request headers.

### How it works
- User lands on Admin UI  
- 👉 **Your custom SSO sign-in handler is called to parse request headers and return user info**
- LiteLLM has retrieved user information from your custom handler
- User signed in to UI

### Usage

#### 1. Create a custom UI SSO handler file

This handler parses request headers and returns user information as an OpenID object:

```python
from fastapi import Request
from fastapi_sso.sso.base import OpenID
from litellm.integrations.custom_sso_handler import CustomSSOLoginHandler


class MyCustomSSOLoginHandler(CustomSSOLoginHandler):
    """
    Custom handler for parsing OAuth proxy headers
    
    Use this when you have an OAuth proxy (like oauth2-proxy, Vouch, etc.) 
    in front of LiteLLM that adds user info to request headers
    """
    async def handle_custom_ui_sso_sign_in(
        self,
        request: Request,
    ) -> OpenID:
        # Parse headers from your OAuth proxy
        request_headers = dict(request.headers)
        
        # Extract user info from headers (adjust header names for your proxy)
        user_id = request_headers.get("x-forwarded-user") or request_headers.get("x-user")
        user_email = request_headers.get("x-forwarded-email") or request_headers.get("x-email")
        user_name = request_headers.get("x-forwarded-preferred-username") or request_headers.get("x-preferred-username")
        
        # Return OpenID object with user information
        return OpenID(
            id=user_id or "unknown",
            email=user_email or "unknown@example.com", 
            first_name=user_name or "Unknown",
            last_name="User",
            display_name=user_name or "Unknown User",
            picture=None,
            provider="oauth-proxy",
        )

# Create an instance to be used by LiteLLM
custom_ui_sso_sign_in_handler = MyCustomSSOLoginHandler()
```

#### 2. Configure in config.yaml

```yaml
model_list: 
  - model_name: "openai-model"
    litellm_params: 
      model: "gpt-3.5-turbo"

general_settings:
  custom_ui_sso_sign_in_handler: custom_sso_handler.custom_ui_sso_sign_in_handler

litellm_settings:
  drop_params: True
  set_verbose: True
```

#### 3. Start the proxy
```shell
$ litellm --config /path/to/config.yaml 
```

#### 4. Navigate to the Admin UI

When a user attempts navigating to the LiteLLM Admin UI, the request will be routed to your custom UI SSO sign-in handler. 

---

## Option 2: Custom SSO Handler (Post-Authentication)

Use this if you want to run your own code **after** a user signs on to the LiteLLM UI using standard SSO providers (Google, Microsoft, etc.)

### How it works
- User lands on Admin UI
- LiteLLM redirects user to your SSO provider (Google, Microsoft, etc.)
- Your SSO provider redirects user back to LiteLLM  
- LiteLLM has retrieved user information from your IDP
- 👉 **Your custom SSO handler is called and returns an object of type SSOUserDefinedValues**
- User signed in to UI

### Usage

#### 1. Create a custom SSO handler file

Make sure the response type follows the `SSOUserDefinedValues` pydantic object. This is used for logging the user into the Admin UI:

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
        # Access extra fields from SSO provider (requires GENERIC_USER_EXTRA_ATTRIBUTES env var)
        # Example: Set GENERIC_USER_EXTRA_ATTRIBUTES="department,employee_id,groups"
        extra_fields = getattr(userIDPInfo, 'extra_fields', None) or {}
        user_department = extra_fields.get("department")
        employee_id = extra_fields.get("employee_id")
        user_groups = extra_fields.get("groups", [])
        
        print(f"User department: {user_department}")  # noqa
        print(f"Employee ID: {employee_id}")  # noqa
        print(f"User groups: {user_groups}")  # noqa
        #################################################

        #################################################
        # Run your custom code / logic here
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

#### 2. Configure in config.yaml

Pass the filepath to the config.yaml. 

e.g. if they're both in the same dir - `./config.yaml` and `./custom_sso.py`, this is what it looks like:

```yaml 
model_list: 
  - model_name: "openai-model"
    litellm_params: 
      model: "gpt-3.5-turbo"

general_settings:
  custom_sso: custom_sso.custom_sso_handler

litellm_settings:
  drop_params: True
  set_verbose: True
```

#### 3. Start the proxy
```shell
$ litellm --config /path/to/config.yaml 
```
