import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# SSO Configuration API

:::info

✨ SSO Configuration API is on LiteLLM Enterprise

[Enterprise Pricing](https://www.litellm.ai/#pricing)

[Get free 7-day trial key](https://www.litellm.ai/#trial)

:::

The SSO Configuration API provides endpoints to programmatically manage SSO provider settings for the LiteLLM Proxy. These endpoints allow you to get, update, and delete SSO configurations without manually editing environment variables or configuration files.

## Authentication

All SSO Configuration API endpoints require authentication using your proxy admin key:

```bash
curl -X GET "https://your-proxy-url/get/sso_provider_config" \
  -H "Authorization: Bearer sk-your-admin-key"
```

## Endpoints

### Get SSO Provider Configuration

Retrieve the current SSO provider configuration including client IDs and endpoints.

**Endpoint:** `GET /get/sso_provider_config`

**Response:**
```json
{
  "config": {
    "sso_provider": "google",
    "google": {
      "google_client_id": "your-client-id",
      "google_client_secret": "***"
    },
    "microsoft": {
      "microsoft_client_id": null,
      "microsoft_client_secret": null,
      "microsoft_tenant": null
    },
    "generic": {
      "generic_client_id": null,
      "generic_client_secret": null,
      "generic_authorization_endpoint": null,
      "generic_token_endpoint": null,
      "generic_userinfo_endpoint": null,
      "generic_scope": "openid email profile"
    },
    "proxy_base_url": "https://your-proxy.com",
    "user_email": "admin@yourcompany.com"
  },
  "status": "success"
}
```

**Example:**
```bash
curl -X GET "https://your-proxy-url/get/sso_provider_config" \
  -H "Authorization: Bearer sk-your-admin-key"
```

### Update SSO Provider Configuration

Update SSO provider configuration programmatically. This endpoint updates both environment variables and configuration files.

**Endpoint:** `POST /update/sso_provider_config`

**Request Body:**

<Tabs>
<TabItem value="google" label="Google SSO">

```json
{
  "sso_provider": "google",
  "google_client_id": "your-google-client-id",
  "google_client_secret": "your-google-client-secret",
  "proxy_base_url": "https://your-proxy.com",
  "user_email": "admin@yourcompany.com"
}
```

</TabItem>

<TabItem value="microsoft" label="Microsoft SSO">

```json
{
  "sso_provider": "microsoft",
  "microsoft_client_id": "your-microsoft-client-id",
  "microsoft_client_secret": "your-microsoft-client-secret",
  "microsoft_tenant": "your-tenant-id",
  "proxy_base_url": "https://your-proxy.com",
  "user_email": "admin@yourcompany.com"
}
```

</TabItem>

<TabItem value="generic" label="Generic/Okta SSO">

```json
{
  "sso_provider": "generic",
  "generic_client_id": "your-client-id",
  "generic_client_secret": "your-client-secret",
  "generic_authorization_endpoint": "https://your-provider.com/oauth/authorize",
  "generic_token_endpoint": "https://your-provider.com/oauth/token",
  "generic_userinfo_endpoint": "https://your-provider.com/oauth/userinfo",
  "generic_scope": "openid email profile",
  "proxy_base_url": "https://your-proxy.com",
  "user_email": "admin@yourcompany.com"
}
```

</TabItem>
</Tabs>

**Response:**
```json
{
  "message": "SSO provider configuration updated successfully",
  "status": "success",
  "provider": "google",
  "note": "Some changes may require a server restart to take full effect"
}
```

**Example:**
```bash
curl -X POST "https://your-proxy-url/update/sso_provider_config" \
  -H "Authorization: Bearer sk-your-admin-key" \
  -H "Content-Type: application/json" \
  -d '{
    "sso_provider": "google",
    "google_client_id": "your-google-client-id",
    "google_client_secret": "your-google-client-secret",
    "proxy_base_url": "https://your-proxy.com",
    "user_email": "admin@yourcompany.com"
  }'
```

### Delete SSO Provider Configuration

Remove SSO provider configuration and reset to defaults.

**Endpoint:** `DELETE /delete/sso_provider_config`

**Response:**
```json
{
  "message": "SSO provider configuration deleted successfully",
  "status": "success",
  "note": "Server restart may be required for changes to take full effect"
}
```

**Example:**
```bash
curl -X DELETE "https://your-proxy-url/delete/sso_provider_config" \
  -H "Authorization: Bearer sk-your-admin-key"
```

## User and Team Default Settings

### Get Internal User Settings

Retrieve default settings applied to new SSO users.

**Endpoint:** `GET /get/internal_user_settings`

**Response:**
```json
{
  "values": {
    "max_budget": 100.0,
    "budget_duration": "1mo",
    "allowed_models": ["gpt-4", "gpt-3.5-turbo"],
    "user_role": "internal_user"
  },
  "schema": {
    "description": "Default settings for internal users created via SSO",
    "properties": {
      "max_budget": {
        "description": "Maximum budget for the user",
        "type": "number"
      },
      "budget_duration": {
        "description": "Budget duration (e.g., '1d', '1w', '1mo')",
        "type": "string"
      }
    }
  }
}
```

### Update Internal User Settings

Update default settings for new SSO users.

**Endpoint:** `PATCH /update/internal_user_settings`

**Request Body:**
```json
{
  "max_budget": 200.0,
  "budget_duration": "1mo",
  "allowed_models": ["gpt-4", "gpt-3.5-turbo", "claude-3"],
  "user_role": "internal_user"
}
```

**Response:**
```json
{
  "message": "Internal user settings updated successfully",
  "status": "success",
  "settings": {
    "max_budget": 200.0,
    "budget_duration": "1mo",
    "allowed_models": ["gpt-4", "gpt-3.5-turbo", "claude-3"],
    "user_role": "internal_user"
  }
}
```

### Get Default Team Settings

Retrieve default settings applied to new teams created from SSO.

**Endpoint:** `GET /get/default_team_settings`

### Update Default Team Settings

Update default settings for new teams created from SSO.

**Endpoint:** `PATCH /update/default_team_settings`

**Request Body:**
```json
{
  "max_budget": 500.0,
  "budget_duration": "1mo",
  "models": ["gpt-4", "gpt-3.5-turbo"],
  "spend": 0.0
}
```

## IP Address Management

### Get Allowed IPs

**Endpoint:** `GET /get/allowed_ips`

### Add Allowed IP

**Endpoint:** `POST /add/allowed_ip`

**Request Body:**
```json
{
  "ip": "192.168.1.100"
}
```

### Delete Allowed IP

**Endpoint:** `POST /delete/allowed_ip`

**Request Body:**
```json
{
  "ip": "192.168.1.100"
}
```

## Error Handling

All endpoints return appropriate HTTP status codes and error messages:

- `200 OK`: Successful operation
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Missing or invalid authentication
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

**Example Error Response:**
```json
{
  "detail": "Failed to update SSO provider configuration: Invalid client ID"
}
```

## Notes

- Some configuration changes may require a server restart to take full effect
- Secret values (like client secrets) are masked in GET responses for security
- All configuration changes are persisted to both memory and configuration files
- Changes are immediately available but may need a restart for SSO flow to work properly

## Python SDK Example

```python
import requests

# Get current SSO config
response = requests.get(
    "https://your-proxy-url/get/sso_provider_config",
    headers={"Authorization": "Bearer sk-your-admin-key"}
)
config = response.json()

# Update SSO config
update_data = {
    "sso_provider": "google",
    "google_client_id": "new-client-id",
    "google_client_secret": "new-client-secret",
    "proxy_base_url": "https://your-proxy.com",
    "user_email": "admin@yourcompany.com"
}

response = requests.post(
    "https://your-proxy-url/update/sso_provider_config",
    headers={
        "Authorization": "Bearer sk-your-admin-key",
        "Content-Type": "application/json"
    },
    json=update_data
)
result = response.json()
print(result["message"])
``` 