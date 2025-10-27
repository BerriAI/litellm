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

## UserAPIKeyAuth Fields Reference

The `UserAPIKeyAuth` object supports the following fields for comprehensive auth configuration:

### Core Authentication Fields
```python
UserAPIKeyAuth(
    # Basic auth fields
    api_key: Optional[str] = None,                    # The API key (will be hashed automatically)
    token: Optional[str] = None,                      # Hashed token for internal use
    key_name: Optional[str] = None,                   # Human-readable key name
    key_alias: Optional[str] = None,                  # Key alias for identification
    
    # User identification
    user_id: Optional[str] = None,                    # Unique user identifier
    user_email: Optional[str] = None,                 # User email address
    user_role: Optional[LitellmUserRoles] = None,     # User role (PROXY_ADMIN, INTERNAL_USER, etc.)
    
    # Team/Organization
    team_id: Optional[str] = None,                    # Team identifier
    team_alias: Optional[str] = None,                 # Team display name
    org_id: Optional[str] = None,                     # Organization identifier
)
```

### Budget and Spend Tracking
```python
UserAPIKeyAuth(
    # User budgets
    max_budget: Optional[float] = None,               # Maximum budget for the key
    spend: float = 0.0,                              # Current spend amount
    soft_budget: Optional[float] = None,              # Soft budget limit (warnings)
    model_max_budget: Dict = {},                      # Per-model budget limits
    model_spend: Dict = {},                           # Per-model spend tracking
    
    # Team budgets
    team_max_budget: Optional[float] = None,          # Team's maximum budget
    team_spend: Optional[float] = None,               # Team's current spend
    team_member_spend: Optional[float] = None,        # This user's spend within the team
    
    # Budget timing
    budget_duration: Optional[str] = None,            # Budget reset period
    budget_reset_at: Optional[datetime] = None,       # When budget resets
)
```

### Rate Limiting
```python
UserAPIKeyAuth(
    # User limits
    tpm_limit: Optional[int] = None,                  # Tokens per minute limit
    rpm_limit: Optional[int] = None,                  # Requests per minute limit
    user_tpm_limit: Optional[int] = None,             # User-specific TPM limit
    user_rpm_limit: Optional[int] = None,             # User-specific RPM limit
    
    # Team limits
    team_tpm_limit: Optional[int] = None,             # Team TPM limit
    team_rpm_limit: Optional[int] = None,             # Team RPM limit
    team_member_tpm_limit: Optional[int] = None,      # Per-member TPM limit
    team_member_rpm_limit: Optional[int] = None,      # Per-member RPM limit
    
    # Per-model limits
    rpm_limit_per_model: Optional[Dict[str, int]] = None,  # RPM limits by model
    tpm_limit_per_model: Optional[Dict[str, int]] = None,  # TPM limits by model
)
```

### End User Tracking
```python
UserAPIKeyAuth(
    # End user identification and limits
    end_user_id: Optional[str] = None,                # End user identifier
    end_user_tpm_limit: Optional[int] = None,         # End user TPM limit
    end_user_rpm_limit: Optional[int] = None,         # End user RPM limit
    end_user_max_budget: Optional[float] = None,      # End user budget limit
)
```

### Model and Route Access
```python
UserAPIKeyAuth(
    # Model access control
    models: List = [],                                # Allowed models list
    team_models: List = [],                           # Team's allowed models
    aliases: Dict = {},                               # Model aliases
    
    # Route permissions
    allowed_routes: Optional[list] = [],              # Allowed API routes
    allowed_cache_controls: Optional[list] = [],      # Cache control permissions
    permissions: Dict = {},                           # General permissions
)
```

### Advanced Configuration
```python
UserAPIKeyAuth(
    # Request handling
    max_parallel_requests: Optional[int] = None,      # Concurrent request limit
    allowed_model_region: Optional[AllowedModelRegion] = None,  # Geographic restrictions
    
    # Expiration and status
    expires: Optional[Union[str, datetime]] = None,   # Key expiration
    blocked: Optional[bool] = None,                   # Whether key is blocked
    
    # Metadata and configuration
    metadata: Dict = {},                              # Custom metadata
    config: Dict = {},                               # Configuration settings
    team_metadata: Optional[Dict] = None,             # Team metadata
    
    # Internal tracking
    request_route: Optional[str] = None,              # Current request route
    last_refreshed_at: Optional[float] = None,        # Cache refresh timestamp
)
```

### Complete Example

```python
from datetime import datetime, timedelta
from litellm.proxy._types import UserAPIKeyAuth, LitellmUserRoles

async def user_api_key_auth(request: Request, api_key: str) -> UserAPIKeyAuth:
    try:
        # Example: Comprehensive auth configuration
        if api_key.startswith("sk-admin-"):
            return UserAPIKeyAuth(
                api_key=api_key,
                user_id="admin_user_123",
                user_email="admin@company.com",
                user_role=LitellmUserRoles.PROXY_ADMIN,
                team_id="admin_team",
                team_alias="Administrative Team",
                max_budget=1000.0,
                soft_budget=800.0,
                tpm_limit=10000,
                rpm_limit=100,
                models=["gpt-4", "claude-3-sonnet", "gpt-3.5-turbo"],
                allowed_routes=["/chat/completions", "/embeddings"],
                expires=datetime.now() + timedelta(days=30),
                metadata={"department": "engineering", "cost_center": "ai_ops"}
            )
        elif api_key.startswith("sk-team-"):
            return UserAPIKeyAuth(
                api_key=api_key,
                user_id="team_user_456",
                user_email="user@company.com",
                user_role=LitellmUserRoles.INTERNAL_USER,
                team_id="dev_team",
                team_alias="Development Team",
                max_budget=100.0,
                tpm_limit=1000,
                rpm_limit=20,
                models=["gpt-3.5-turbo", "claude-3-haiku"],
                team_member_tpm_limit=500,  # Limit within team
                end_user_tpm_limit=100,     # Per end-user limit
                metadata={"project": "chatbot_v2"}
            )
        else:
            raise Exception("Invalid API key")
    except Exception:
        raise Exception("Authentication failed")
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