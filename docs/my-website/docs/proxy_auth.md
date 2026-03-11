import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# SDK Proxy Authentication (OAuth2/JWT Auto-Refresh)

Automatically obtain and refresh OAuth2/JWT tokens when using the LiteLLM Python SDK with a LiteLLM Proxy that requires JWT authentication.

## Overview

When your LiteLLM Proxy is protected by an OAuth2/OIDC provider (Azure AD, Keycloak, Okta, Auth0, etc.), your SDK clients need valid JWT tokens for every request. Instead of manually managing token lifecycle, `litellm.proxy_auth` handles this automatically:

- Obtains tokens from your identity provider
- Caches tokens to avoid unnecessary requests
- Refreshes tokens before they expire (60-second buffer)
- Injects `Authorization: Bearer <token>` headers into every request

## Quick Start

### Azure AD

<Tabs>
<TabItem value="default" label="DefaultAzureCredential">

Uses the [DefaultAzureCredential](https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential) chain (environment variables, managed identity, Azure CLI, etc.):

```python
import litellm
from litellm.proxy_auth import AzureADCredential, ProxyAuthHandler

# One-time setup
litellm.proxy_auth = ProxyAuthHandler(
    credential=AzureADCredential(),  # uses DefaultAzureCredential
    scope="api://my-litellm-proxy/.default"
)
litellm.api_base = "https://my-proxy.example.com"

# All requests now include Authorization headers automatically
response = litellm.completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

</TabItem>
<TabItem value="client-secret" label="ClientSecretCredential">

Use a specific Azure AD app registration:

```python
import litellm
from azure.identity import ClientSecretCredential
from litellm.proxy_auth import AzureADCredential, ProxyAuthHandler

azure_cred = ClientSecretCredential(
    tenant_id="your-tenant-id",
    client_id="your-client-id",
    client_secret="your-client-secret"
)

litellm.proxy_auth = ProxyAuthHandler(
    credential=AzureADCredential(credential=azure_cred),
    scope="api://my-litellm-proxy/.default"
)
litellm.api_base = "https://my-proxy.example.com"

response = litellm.completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

</TabItem>
</Tabs>

**Required package:** `pip install azure-identity`

### Generic OAuth2 (Okta, Auth0, Keycloak, etc.)

Works with any OAuth2 provider that supports the `client_credentials` grant type:

```python
import litellm
from litellm.proxy_auth import GenericOAuth2Credential, ProxyAuthHandler

litellm.proxy_auth = ProxyAuthHandler(
    credential=GenericOAuth2Credential(
        client_id="your-client-id",
        client_secret="your-client-secret",
        token_url="https://your-idp.example.com/oauth2/token"
    ),
    scope="litellm_proxy_api"
)
litellm.api_base = "https://my-proxy.example.com"

response = litellm.completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Custom Credential Provider

Implement the `TokenCredential` protocol to use any authentication mechanism:

```python
import time
import litellm
from litellm.proxy_auth import AccessToken, ProxyAuthHandler

class MyCustomCredential:
    """Any class with a get_token(scope) -> AccessToken method works."""

    def get_token(self, scope: str) -> AccessToken:
        # Your custom logic to obtain a token
        token = my_auth_system.get_jwt(scope=scope)
        return AccessToken(
            token=token,
            expires_on=int(time.time()) + 3600
        )

litellm.proxy_auth = ProxyAuthHandler(
    credential=MyCustomCredential(),
    scope="my-scope"
)
```

## Supported Endpoints

Auth headers are automatically injected for:

| Endpoint | Function |
|----------|----------|
| Chat Completions | `litellm.completion()` / `litellm.acompletion()` |
| Embeddings | `litellm.embedding()` / `litellm.aembedding()` |

## How It Works

```
┌──────────┐     ┌──────────────────┐     ┌──────────────┐     ┌──────────────┐
│  Your    │     │  ProxyAuthHandler │     │   Identity   │     │  LiteLLM     │
│  Code    │────▶│  (token cache)   │────▶│   Provider   │     │  Proxy       │
│          │     │                  │◀────│  (Azure AD,  │     │              │
│          │     │                  │     │   Okta, etc) │     │              │
│          │     └────────┬─────────┘     └──────────────┘     │              │
│          │              │ Authorization: Bearer <token>      │              │
│          │──────────────┼───────────────────────────────────▶│              │
│          │◀─────────────┼────────────────────────────────────│              │
└──────────┘              │                                    └──────────────┘
```

1. You set `litellm.proxy_auth` once at startup
2. On each SDK call (`completion()`, `embedding()`), the handler checks its cached token
3. If the token is missing or expires within 60 seconds, it requests a new one from your identity provider
4. The `Authorization: Bearer <token>` header is injected into the request
5. If token retrieval fails, a warning is logged and the request proceeds without auth headers

## API Reference

### ProxyAuthHandler

The main handler that manages the token lifecycle.

```python
from litellm.proxy_auth import ProxyAuthHandler

handler = ProxyAuthHandler(
    credential=<TokenCredential>,  # required - credential provider
    scope="<oauth2-scope>"         # required - OAuth2 scope to request
)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `credential` | `TokenCredential` | Yes | A credential provider (AzureADCredential, GenericOAuth2Credential, or custom) |
| `scope` | `str` | Yes | The OAuth2 scope to request tokens for |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `get_token()` | `AccessToken` | Get a valid token, refreshing if needed |
| `get_auth_headers()` | `dict` | Get `{"Authorization": "Bearer <token>"}` headers |

### AzureADCredential

Wraps any `azure-identity` credential with lazy initialization.

```python
from litellm.proxy_auth import AzureADCredential

# Uses DefaultAzureCredential (recommended)
cred = AzureADCredential()

# Or wrap a specific azure-identity credential
from azure.identity import ManagedIdentityCredential
cred = AzureADCredential(credential=ManagedIdentityCredential())
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `credential` | Azure `TokenCredential` | No | An azure-identity credential. If `None`, uses `DefaultAzureCredential` |

### GenericOAuth2Credential

Standard OAuth2 client credentials flow for any provider.

```python
from litellm.proxy_auth import GenericOAuth2Credential

cred = GenericOAuth2Credential(
    client_id="your-client-id",
    client_secret="your-client-secret",
    token_url="https://your-idp.com/oauth2/token"
)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `client_id` | `str` | Yes | OAuth2 client ID |
| `client_secret` | `str` | Yes | OAuth2 client secret |
| `token_url` | `str` | Yes | Token endpoint URL |

### AccessToken

Dataclass representing an OAuth2 access token.

```python
from litellm.proxy_auth import AccessToken

token = AccessToken(
    token="eyJhbG...",     # JWT string
    expires_on=1234567890  # Unix timestamp
)
```

### TokenCredential Protocol

Any class implementing this protocol can be used as a credential provider:

```python
from litellm.proxy_auth import AccessToken

class MyCredential:
    def get_token(self, scope: str) -> AccessToken:
        ...
```

## Provider-Specific Examples

### Keycloak

```python
from litellm.proxy_auth import GenericOAuth2Credential, ProxyAuthHandler

litellm.proxy_auth = ProxyAuthHandler(
    credential=GenericOAuth2Credential(
        client_id="litellm-client",
        client_secret="your-keycloak-client-secret",
        token_url="https://keycloak.example.com/realms/your-realm/protocol/openid-connect/token"
    ),
    scope="openid"
)
```

### Okta

```python
from litellm.proxy_auth import GenericOAuth2Credential, ProxyAuthHandler

litellm.proxy_auth = ProxyAuthHandler(
    credential=GenericOAuth2Credential(
        client_id="your-okta-client-id",
        client_secret="your-okta-client-secret",
        token_url="https://your-org.okta.com/oauth2/default/v1/token"
    ),
    scope="litellm_api"
)
```

### Auth0

```python
from litellm.proxy_auth import GenericOAuth2Credential, ProxyAuthHandler

litellm.proxy_auth = ProxyAuthHandler(
    credential=GenericOAuth2Credential(
        client_id="your-auth0-client-id",
        client_secret="your-auth0-client-secret",
        token_url="https://your-tenant.auth0.com/oauth/token"
    ),
    scope="https://my-proxy.example.com/api"
)
```

### Azure AD with Managed Identity

```python
from azure.identity import ManagedIdentityCredential
from litellm.proxy_auth import AzureADCredential, ProxyAuthHandler

litellm.proxy_auth = ProxyAuthHandler(
    credential=AzureADCredential(
        credential=ManagedIdentityCredential()
    ),
    scope="api://my-litellm-proxy/.default"
)
```

## Combining with `use_litellm_proxy`

You can use `proxy_auth` together with [`use_litellm_proxy`](./providers/litellm_proxy#send-all-sdk-requests-to-litellm-proxy) to route all SDK requests through an authenticated proxy:

```python
import os
import litellm
from litellm.proxy_auth import AzureADCredential, ProxyAuthHandler

# Route all requests through the proxy
os.environ["LITELLM_PROXY_API_BASE"] = "https://my-proxy.example.com"
litellm.use_litellm_proxy = True

# Authenticate with OAuth2/JWT
litellm.proxy_auth = ProxyAuthHandler(
    credential=AzureADCredential(),
    scope="api://my-litellm-proxy/.default"
)

# This request goes through the proxy with automatic JWT auth
response = litellm.completion(
    model="vertex_ai/gemini-2.0-flash-001",
    messages=[{"role": "user", "content": "Hello!"}]
)
```
