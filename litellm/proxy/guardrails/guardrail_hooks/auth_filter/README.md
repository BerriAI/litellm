# Auth Filter Guardrail

The auth_filter guardrail enables custom authentication enrichment and validation logic that runs **after** LiteLLM's standard authentication completes.

## When to Use

Use auth_filter when you need to:
- Validate authenticated users against external systems
- Enrich the auth object with additional metadata
- Apply organization-specific access rules
- Call external APIs for compliance checks
- Block requests based on custom logic

## Execution Flow

```
1. Standard LiteLLM Auth (DB lookup) → UserAPIKeyAuth
2. Auth Filter Guardrails Execute → Enrich/Validate/Block
3. Continue to LLM Request
```

The auth_filter receives the **validated** UserAPIKeyAuth object from standard authentication, along with the original request and API key.

## Configuration

### Basic Example

```yaml
guardrails:
  - guardrail_name: "org-validator"
    litellm_params:
      guardrail: "auth_filter"
      mode: "post_auth_check"
      custom_code: |
        def auth_filter(request, api_key, user_api_key_auth):
            # Access organization ID from authenticated user
            org_id = user_api_key_auth.organization_id

            if org_id == "restricted-org":
                return block("Organization access restricted")

            return allow()
```

### Async Example with External API

```yaml
guardrails:
  - guardrail_name: "compliance-check"
    litellm_params:
      guardrail: "auth_filter"
      mode: "post_auth_check"
      custom_code: |
        async def auth_filter(request, api_key, user_api_key_auth):
            # Call external compliance API
            org_id = user_api_key_auth.organization_id

            response = await http_post(
                "https://compliance.internal/validate",
                body={"org_id": org_id, "endpoint": request.url.path}
            )

            if not response["success"]:
                return block("Compliance validation failed")

            # Enrich with compliance session
            user_api_key_auth.metadata["compliance_session"] = response["body"]["session_id"]
            return modify(user_api_key_auth=user_api_key_auth)
```

### Reading Request Headers

```yaml
custom_code: |
    def auth_filter(request, api_key, user_api_key_auth):
        # Extract custom headers
        department = request.headers.get("X-Department")
        environment = request.headers.get("X-Environment", "prod")

        # Apply department-specific rules
        if department == "healthcare" and environment == "prod":
            # Additional validation for healthcare in production
            if not user_api_key_auth.metadata.get("hipaa_certified"):
                return block("HIPAA certification required for healthcare")

        return allow()
```

## Function Signature

```python
def auth_filter(request, api_key, user_api_key_auth):
    """
    Args:
        request (Request): FastAPI Request object with headers, query params, body
        api_key (str): The API key used for authentication
        user_api_key_auth (UserAPIKeyAuth): Validated auth object from standard auth

    Returns:
        - allow() - Continue without modification
        - block(reason) - Reject with 403 error
        - modify(user_api_key_auth=obj) - Return enriched auth object
    """
```

### Async Support

Use `async def` when making external HTTP requests:

```python
async def auth_filter(request, api_key, user_api_key_auth):
    response = await http_get("https://api.example.com/validate")
    # ...
```

## Available Primitives

Auth filters run in the same sandbox as custom_code guardrails, with access to:

### HTTP Requests
- `http_get(url, headers=None, timeout=30)`
- `http_post(url, body=None, headers=None, timeout=30)`
- `http_request(method, url, body=None, headers=None, timeout=30)`

### Result Actions
- `allow()` - Continue without changes
- `block(reason)` - Reject with error message
- `modify(user_api_key_auth=obj)` - Return modified auth object

### Regex
- `regex_match(text, pattern)`
- `regex_replace(text, pattern, replacement)`
- `regex_find_all(text, pattern)`

### JSON
- `json_parse(text)`
- `json_stringify(obj)`
- `json_schema_valid(data, schema)`

### Text Utilities
- `contains(text, substring)`
- `lower(text)`, `upper(text)`, `trim(text)`
- `word_count(text)`, `char_count(text)`

### Safe Builtins
- `len()`, `str()`, `int()`, `float()`, `bool()`
- `list()`, `dict()`, `True`, `False`, `None`

## UserAPIKeyAuth Fields

The `user_api_key_auth` object contains:

### Core Fields
- `api_key`: The API key
- `token`: Token identifier
- `key_name`, `key_alias`: Key identifiers

### User/Team/Org
- `user_id`, `user_email`, `user_role`
- `team_id`, `team_alias`
- `org_id`, `organization_id`

### Access Control
- `models`: Allowed models
- `team_models`: Team-level models
- `permissions`: User permissions
- `allowed_routes`: Allowed API routes

### Budgets & Limits
- `max_budget`, `soft_budget`
- `tpm_limit`, `rpm_limit` (tokens/requests per minute)
- `team_max_budget`, `user_max_budget`

### Metadata
- `metadata`: Dict for custom data

## Hot-Reload Support

Auth filters support hot-reload without proxy restart:

1. **Update via API:**
   ```bash
   curl -X PUT http://localhost:4000/guardrails/{guardrail_id} \
     -H "Authorization: Bearer $ADMIN_KEY" \
     -d '{"custom_code": "..."}'
   ```

2. **Update via Database:**
   Update the `custom_code` field in `LiteLLM_GuardrailsTable`

Changes take effect immediately on the next request.

## Error Handling

### Compilation Errors

If custom code has syntax errors, they're caught at initialization:

```python
# Bad syntax - will fail at startup
def auth_filter(request, api_key, user_api_key_auth)  # Missing colon
    return allow()
```

Error message: `"Syntax error in custom code: ..."`

### Runtime Errors

Uncaught exceptions are logged and converted to 500 errors:

```python
# Runtime error example
def auth_filter(request, api_key, user_api_key_auth):
    x = 1 / 0  # ZeroDivisionError
```

**Best Practice:** Use try/except in your code:

```python
def auth_filter(request, api_key, user_api_key_auth):
    try:
        # Your logic
        result = some_operation()
    except Exception as e:
        return block(f"Validation failed: {e}")
    return allow()
```

## Use Cases

### 1. Organization-Based Access Control

```python
def auth_filter(request, api_key, user_api_key_auth):
    org_id = user_api_key_auth.organization_id

    # Different rules per org
    if org_id == "free-tier":
        allowed_models = ["gpt-3.5-turbo"]
        if request.url.path not in ["/chat/completions"]:
            return block("Free tier: only chat completions allowed")
    elif org_id == "enterprise":
        # Enterprise has full access
        pass
    else:
        return block("Unknown organization")

    return allow()
```

### 2. External Compliance Validation

```python
async def auth_filter(request, api_key, user_api_key_auth):
    response = await http_post(
        "https://compliance.internal/validate",
        body={
            "user_id": user_api_key_auth.user_id,
            "org_id": user_api_key_auth.organization_id,
            "endpoint": request.url.path
        }
    )

    if response["status_code"] != 200:
        return block("Compliance check failed")

    compliance_data = response["body"]
    if not compliance_data.get("approved"):
        return block(f"Access denied: {compliance_data.get('reason')}")

    # Enrich with compliance session
    user_api_key_auth.metadata["compliance_session_id"] = compliance_data["session_id"]
    user_api_key_auth.metadata["compliance_level"] = compliance_data["level"]

    return modify(user_api_key_auth=user_api_key_auth)
```

### 3. Header-Based Department Routing

```python
def auth_filter(request, api_key, user_api_key_auth):
    department = request.headers.get("X-Department")

    if not department:
        return block("X-Department header required")

    # Store department in metadata for downstream use
    user_api_key_auth.metadata["department"] = department

    # Department-specific model restrictions
    if department == "research":
        # Research can use expensive models
        pass
    elif department == "customer-service":
        # Customer service limited to fast models
        user_api_key_auth.models = ["gpt-3.5-turbo", "gpt-4o-mini"]
    else:
        return block(f"Unknown department: {department}")

    return modify(user_api_key_auth=user_api_key_auth)
```

## Security Notes

- Auth filters run in a sandboxed environment
- No file I/O access
- No arbitrary imports
- HTTP requests have timeout limits (30s default, 60s max)
- Exceptions are caught and logged, not exposed to clients

## Debugging

Enable debug logging to see auth filter execution:

```python
# In custom code
def auth_filter(request, api_key, user_api_key_auth):
    # Use print() for debugging (goes to proxy logs)
    print(f"Auth filter: checking user {user_api_key_auth.user_id}")
    print(f"Organization: {user_api_key_auth.organization_id}")

    return allow()
```

View logs with:
```bash
tail -f litellm_proxy.log | grep "Auth filter"
```
