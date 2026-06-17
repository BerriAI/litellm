# Custom Auth 

You can now override the default api key auth.

:::warning Enforcement with custom auth

By default, custom auth enforces only the rate limits you set on the returned object. Budgets and model-access require a flag. The table below shows, for each control, where to configure it and which flags it needs.

:::

## What gets enforced

| Goal | Set it on | Flags required |
| --- | --- | --- |
| Key / user / team / end-user rate limits | returned object (`rpm_limit`, `team_tpm_limit`, …) | none |
| Per-model rate limits, key / team scoped | `metadata` / `team_metadata` on the returned object | none |
| Per-model rate limits, project scoped | the project record (`model_tpm_limit` / `model_rpm_limit`) | `custom_auth_run_common_checks` |
| Team / user / project budget | the team / user / project record | `custom_auth_run_common_checks` |
| Team / user / project model allowlist | the team / user / project record | `custom_auth_run_common_checks` |
| End-user budget | the end-user record | `custom_auth_run_common_checks` or `enable_post_custom_auth_checks` |
| Key model allowlist (`models`) | returned object | both flags |
| Key per-model budget (`model_max_budget`) | returned object | `enable_post_custom_auth_checks` |
| Key expiry (`expires`) | returned object | `enable_post_custom_auth_checks` |
| Key scalar budget (`max_budget` / `soft_budget`) | not supported; use a per-scope budget | n/a |

**Note:** Project per-model limits go on the object's `project_metadata` with the flag off, but the DB project record overrides it once the flag is on, so set them there. (Team per-model via `team_metadata` always stays on the object.)

See [Enforce budgets and model access](#enforce-budgets-and-model-access) and [Key-level enforcement](#key-level-enforcement) for examples.

## Usage

#### 1. Create a custom auth file. 

Make sure the response type follows the `UserAPIKeyAuth` pydantic object. This is used by for logging usage specific to that user key.

```python
from fastapi import Request
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

## UserAPIKeyAuth Fields Reference

These fields are read straight off the returned object and enforced with no flag. Budgets and model-access are enforced behind flags (see below).

### Identity

Who the request belongs to. The `*_id` fields also tell LiteLLM which DB records to load when `custom_auth_run_common_checks: true`.

```python
UserAPIKeyAuth(
    api_key: Optional[str] = None,                    # The API key (will be hashed automatically)
    token: Optional[str] = None,                      # Hashed token for internal use
    key_alias: Optional[str] = None,                  # Key alias for identification
    user_id: Optional[str] = None,                    # User identifier (also used to load the user record)
    user_email: Optional[str] = None,                 # User email address
    user_role: Optional[LitellmUserRoles] = None,     # User role (PROXY_ADMIN, INTERNAL_USER, etc.)
    team_id: Optional[str] = None,                    # Team identifier (also used to load the team record)
    org_id: Optional[str] = None,                     # Organization identifier (also used to load the org record)
    end_user_id: Optional[str] = None,                # End-user identifier (also used to load the end-user record)
)
```

### Rate limits

All scopes below are enforced directly off the returned object, with no flag.

```python
UserAPIKeyAuth(
    # Key
    tpm_limit: Optional[int] = None,
    rpm_limit: Optional[int] = None,
    # User
    user_tpm_limit: Optional[int] = None,
    user_rpm_limit: Optional[int] = None,
    # Team
    team_tpm_limit: Optional[int] = None,
    team_rpm_limit: Optional[int] = None,
    # Per team-member
    team_member_tpm_limit: Optional[int] = None,
    team_member_rpm_limit: Optional[int] = None,
    # Per end-user
    end_user_tpm_limit: Optional[int] = None,
    end_user_rpm_limit: Optional[int] = None,
    # Per-model (key / team scoped)
    metadata: Dict = {},          # e.g. {"model_tpm_limit": {...}, "model_rpm_limit": {...}}
    team_metadata: Optional[Dict] = None,  # same keys, team scoped
)
```

:::note

Per-model rate limits are read from `metadata` (key) and `team_metadata` (team), keyed by model name. The model key must equal the request's `model` string exactly, or the limit is skipped silently.

`rpm_limit_per_model` / `tpm_limit_per_model` exist on the object but are inert; use `metadata` / `team_metadata` instead, or the project record (see below).

:::

### Advanced

```python
UserAPIKeyAuth(
    max_parallel_requests: Optional[int] = None,      # Concurrent request limit
    allowed_model_region: Optional[AllowedModelRegion] = None,  # Geographic restrictions
    blocked: Optional[bool] = None,                   # Whether the key is blocked
    config: Dict = {},                                # Configuration settings
)
```

### Object permission (MCP, agents, etc.)

```python
from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
    global_mcp_server_manager,
)

def _server_id(name: str) -> str:
    server = global_mcp_server_manager.get_mcp_server_by_name(name)
    if not server:
        raise ValueError(f"Unknown MCP server '{name}'")
    return server.server_id

object_permission = LiteLLM_ObjectPermissionTable(
    mcp_servers=[_server_id("deepwiki"), _server_id("everything")], # MCP servers this key is allowed to use
    mcp_tool_permissions={"deepwiki": ["search", "read_doc"]},      # optional per-server tool allow-list
)

UserAPIKeyAuth(
    object_permission=object_permission,
)
```

## Enforce budgets and model access

Set `custom_auth_run_common_checks: true` to enforce budgets and model-access alongside custom auth:

```yaml
general_settings:
  custom_auth: custom_auth.user_api_key_auth
  custom_auth_run_common_checks: true
```

Your handler returns the IDs; the budgets and allowlists live on the matching DB records (`/team/new`, `/user/new`, `/project/new`, `/customer/new`, or the UI), which LiteLLM loads and enforces.

For example, a team with a budget and model allowlist:

```bash
curl -X POST 'http://0.0.0.0:4000/team/new' \
  -H 'Authorization: Bearer sk-master-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "team_id": "eng-team",
    "max_budget": 100,
    "models": ["gpt-4o-mini", "claude-3-haiku"]
  }'
```

```python
# ...then return that team_id from custom auth:
return UserAPIKeyAuth(api_key=api_key, team_id="eng-team")
```

For project per-model rate limits, set `model_tpm_limit` / `model_rpm_limit` on the project record (keyed by model name) and return that `project_id`:

```python
# On the project record (via /project/new or the UI):
#   model_tpm_limit = {"gpt-4o": 100000, "claude-3-haiku": 50000}
#   model_rpm_limit = {"gpt-4o": 100,    "claude-3-haiku": 200}
```

:::note

- The project record's metadata replaces any `project_metadata` you set on the returned object, so configure project per-model limits on the project record, not on the object.
- For per-model rate limits, the model key must equal the request's `model` string exactly, or the limit is skipped silently. This was the actual Expedia failure mode.

:::

### Key `models` vs project `models`

These are separate controls:

| Field | Where it is enforced | Source of truth |
| --- | --- | --- |
| `models` on `UserAPIKeyAuth` | Key-level allowlist | Value you return from custom auth |
| `project_id` on `UserAPIKeyAuth` | Project-level allowlist | `models` on the **project record in LiteLLM's DB** |

An empty `models` list (`[]`) means no restriction. Names must match the model group in your config (wildcards supported). See [Project Management](./project_management) and [Config Settings](./config_settings#all-settings).

## Key-level enforcement

The following are enforced from the returned object, but only when `litellm.enable_post_custom_auth_checks: true` is also set:

```yaml
general_settings:
  custom_auth: custom_auth.user_api_key_auth
  custom_auth_run_common_checks: true   # required for the key models allowlist

litellm_settings:
  enable_post_custom_auth_checks: true
```

```python
from datetime import datetime, timedelta, timezone

return UserAPIKeyAuth(
    api_key=api_key,
    models=["gpt-4o-mini"],                                   # key model allowlist (needs both flags)
    model_max_budget={"gpt-4o": {"budget_limit": 100, "time_period": "30d"}},  # key per-model budget
    expires=datetime.now(timezone.utc) + timedelta(days=30),  # key expiry
)
```

This path also enforces end-user budgets and per-model end-user budgets when `end_user_id` is set.

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
