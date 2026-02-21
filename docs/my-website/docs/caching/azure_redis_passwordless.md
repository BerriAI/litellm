---
title: Azure Managed Redis Passwordless (IAM)
---

# Azure Managed Redis Passwordless Authentication

LiteLLM supports [passwordless authentication to Azure Managed Redis](https://learn.microsoft.com/en-us/azure/azure-cache-for-redis/cache-azure-active-directory-for-authentication) using Azure Active Directory (Microsoft Entra ID). This allows you to securely connect to your Redis cache using Azure Managed Identities or Service Principals, avoiding the need to store static connection passwords.

## Prerequisites
1. **Azure Cache for Redis** instance with **Azure AD Authentication enabled**.
2. **`azure-identity` package** installed in your LiteLLM environment:
    ```bash
    pip install azure-identity
    ```
3. Your Azure Identity must have a role assignment (e.g. `Data Owner` or `Data Contributor`) to the Redis cache.

## Configuration

Enable passwordless authentication in your LiteLLM configuration using `azure_redis_ad_token: true`.

### 1. Using System-Assigned Managed Identity

If running on an Azure service with a System-Assigned Managed Identity (e.g., Azure Container Apps, App Service, AKS), you don't need additional credentials. The `DefaultAzureCredential` will automatically discover the identity.

**Config (`config.yaml`)**:
```yaml
litellm_settings:
  cache: true
  cache_params:
    type: redis
    host: <my-redis-cache>.redis.cache.windows.net
    port: 6380
    ssl: true
    azure_redis_ad_token: true
```

*Note: Azure Managed Redis mandates SSL, so `port: 6380` and `ssl: true` are required.*

### 2. Using User-Assigned Managed Identity

If using a User-Assigned Managed Identity, provide your `AZURE_CLIENT_ID` via environment variables.

**Environment variables**:
```bash
export AZURE_CLIENT_ID="<your-managed-identity-client-id>"
# Optional. The Object ID of your identity. Defaults to empty string.
export REDIS_USERNAME="<your-principal-object-id>"
```

### 3. Using Service Principal

If authenticating via an Azure Service Principal, set the standard Azure identity environment variables:

**Environment variables**:
```bash
export AZURE_CLIENT_ID="<your-sp-client-id>"
export AZURE_TENANT_ID="<your-sp-tenant-id>"
export AZURE_CLIENT_SECRET="<your-sp-client-secret>"
# Optional. The Object ID of your Service Principal. Defaults to empty string.
export REDIS_USERNAME="<your-principal-object-id>"
```

Alternatively, you can provide these directly in the `config.yaml`:

```yaml
litellm_settings:
  cache: true
  cache_params:
    type: redis
    host: <my-redis-cache>.redis.cache.windows.net
    port: 6380
    ssl: true
    azure_redis_ad_token: true
    azure_client_id: os.environ/AZURE_CLIENT_ID
    azure_tenant_id: os.environ/AZURE_TENANT_ID
    azure_client_secret: os.environ/AZURE_CLIENT_SECRET
```

## How It Works

1. LiteLLM uses `azure-identity` to request short-lived access tokens explicitly scoped for Redis (`https://redis.azure.com/.default`).
2. LiteLLM establishes a secure TLS connection with Redis and sends an `AUTH` command using the generated token.
3. Every time the underlying `redis-py` connection disconnects or reconnects, LiteLLM intercepts the connection attempt to **generate a fresh token**, seamlessly handling token expiration.
