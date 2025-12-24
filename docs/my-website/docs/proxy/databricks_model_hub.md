# Databricks Model Hub Integration

LiteLLM's model hub can dynamically fetch and display your actual Databricks serving endpoints instead of relying on the static LiteLLM model cost map.

## Overview

By default, the LiteLLM model hub shows models from the LiteLLM cost map. However, for Databricks, you can configure it to fetch your actual deployed serving endpoints directly from the Databricks API. This provides:

- **Real-time endpoint visibility**: See only the models you've actually deployed
- **Accurate endpoint names**: Use the exact endpoint names from your Databricks workspace
- **Better discovery**: Easily find and use your custom fine-tuned models

## How It Works

When Databricks credentials are configured, the model hub can query the Databricks API (`/api/2.0/serving-endpoints`) to retrieve your deployed endpoints and display them in the model hub interface.

## Configuration

### Step 1: Set Databricks Credentials

Configure Databricks authentication using one of these methods:

#### Option A: Personal Access Token (Development)

```bash
export DATABRICKS_API_KEY="dapi..."
export DATABRICKS_API_BASE="https://adb-xxx.azuredatabricks.net/serving-endpoints"
```

#### Option B: OAuth M2M (Production - Recommended)

```bash
export DATABRICKS_CLIENT_ID="your-service-principal-application-id"
export DATABRICKS_CLIENT_SECRET="your-service-principal-secret"
export DATABRICKS_API_BASE="https://adb-xxx.azuredatabricks.net/serving-endpoints"
```

#### Option C: Databricks SDK (Automatic)

If you have the Databricks SDK configured with unified authentication, no additional environment variables are needed:

```bash
pip install databricks-sdk
# SDK will use your configured authentication automatically
```

### Step 2: Access the Model Hub

#### Via API

Fetch Databricks endpoints directly:

```bash
curl http://localhost:4000/public/databricks/serving_endpoints \
  -H "Authorization: Bearer YOUR_LITELLM_API_KEY"
```

Get models for the model hub filtered by provider:

```bash
curl http://localhost:4000/public/model_hub?provider=databricks \
  -H "Authorization: Bearer YOUR_LITELLM_API_KEY"
```

#### Via UI

Navigate to the Model Hub in the LiteLLM UI Dashboard. When Databricks credentials are configured, you'll see your actual deployed endpoints listed with the `databricks/` prefix.

## API Endpoints

### List Databricks Serving Endpoints

```
GET /public/databricks/serving_endpoints
```

**Authentication**: Requires LiteLLM API key

**Query Parameters**:
- `api_key` (optional): Databricks API key (PAT)
- `api_base` (optional): Databricks workspace URL
- `client_id` (optional): OAuth client ID
- `client_secret` (optional): OAuth client secret

**Response**:
```json
{
  "endpoints": [
    {
      "name": "databricks-dbrx-instruct",
      "creator": "user@company.com",
      "creation_timestamp": 1234567890,
      "last_updated_timestamp": 1234567890,
      "state": "READY",
      "config": {
        "served_models": [...],
        "served_entities": [...]
      },
      "endpoint_url": "https://adb-xxx.azuredatabricks.net/serving-endpoints/databricks-dbrx-instruct"
    }
  ],
  "workspace_url": "https://adb-xxx.azuredatabricks.net"
}
```

### Get Model Hub (with Databricks filter)

```
GET /public/model_hub?provider=databricks
```

**Authentication**: Requires LiteLLM API key

**Query Parameters**:
- `provider` (optional): Filter by provider, e.g., "databricks"

**Response** (when provider=databricks):
```json
[
  {
    "model_group": "databricks/databricks-dbrx-instruct",
    "providers": ["databricks"],
    "mode": "chat",
    "supports_function_calling": true,
    "supports_vision": false,
    "supports_parallel_function_calling": false,
    "is_public_model_group": true
  }
]
```

## Example Usage

### Python SDK

```python
import requests

# Fetch Databricks endpoints
response = requests.get(
    "http://localhost:4000/public/databricks/serving_endpoints",
    headers={"Authorization": "Bearer YOUR_LITELLM_API_KEY"}
)

endpoints = response.json()["endpoints"]
for endpoint in endpoints:
    print(f"Endpoint: {endpoint['name']}, State: {endpoint['state']}")
```

### Using in LiteLLM Proxy

Once endpoints are fetched, you can use them directly in your completions:

```python
import openai

client = openai.OpenAI(
    api_key="YOUR_LITELLM_API_KEY",
    base_url="http://localhost:4000"
)

# Use your Databricks endpoint
response = client.chat.completions.create(
    model="databricks/your-custom-endpoint",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Security

- All sensitive credentials (API keys, tokens, secrets) are automatically redacted from logs
- OAuth M2M authentication is recommended for production deployments
- Credentials are never stored; they're only used for API calls

## Troubleshooting

### "Missing Databricks credentials" error

Ensure you've set one of the following:
- `DATABRICKS_API_KEY` + `DATABRICKS_API_BASE`, or
- `DATABRICKS_CLIENT_ID` + `DATABRICKS_CLIENT_SECRET` + `DATABRICKS_API_BASE`, or
- Have the Databricks SDK installed and configured

### No endpoints returned

- Verify your Databricks credentials are correct
- Check that you have serving endpoints deployed in your Databricks workspace
- Ensure your credentials have permission to list serving endpoints

### Authentication fails

- For PAT: Verify your token hasn't expired
- For OAuth M2M: Check that your service principal has the necessary permissions
- Test your credentials directly with the Databricks API:
  ```bash
  curl https://your-workspace.databricks.net/api/2.0/serving-endpoints \
    -H "Authorization: Bearer YOUR_TOKEN"
  ```

## Benefits

1. **Dynamic Discovery**: Automatically see all your deployed models without manual configuration
2. **No Manual Updates**: As you deploy new endpoints, they appear in the model hub automatically
3. **Better Organization**: Filter and view only your Databricks models
4. **Accurate Metadata**: See endpoint state, creation time, and configuration details

## Limitations

- Requires Databricks credentials to be configured
- Only shows serving endpoints (not all available models in Databricks)
- Pricing information is not available from the Databricks API (falls back to LiteLLM cost map if needed)

## Related Documentation

- [Databricks Provider Setup](./databricks.md)
- [Model Hub Overview](../features/model_hub.md)
- [Authentication Methods](../proxy/virtual_keys.md)
