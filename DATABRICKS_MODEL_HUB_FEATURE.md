# Databricks Model Hub Integration

## Overview

This feature enables LiteLLM's model hub to dynamically fetch and display actual Databricks serving endpoints instead of relying on the static LiteLLM model cost map. This provides users with:

- **Real-time visibility** of their deployed Databricks models
- **Accurate endpoint names** matching their Databricks workspace
- **Better discovery** of custom and fine-tuned models
- **Dynamic updates** as endpoints are deployed/removed

## Changes Made

### Backend Implementation

#### 1. New Public Endpoint: `/public/databricks/serving_endpoints`

**File**: `litellm/proxy/public_endpoints/public_endpoints.py`

A new endpoint that queries the Databricks API (`/api/2.0/serving-endpoints`) to retrieve deployed serving endpoints.

**Features**:
- Supports multiple authentication methods (PAT, OAuth M2M, Databricks SDK)
- Returns formatted endpoint data including:
  - Endpoint name, state, and configuration
  - Served models and entities
  - Creation/update timestamps
  - Direct endpoint URLs
- Includes proper error handling and credential validation

**Authentication Support**:
1. **Personal Access Token (PAT)**: Via `DATABRICKS_API_KEY` + `DATABRICKS_API_BASE`
2. **OAuth M2M** (Recommended for production): Via `DATABRICKS_CLIENT_ID` + `DATABRICKS_CLIENT_SECRET` + `DATABRICKS_API_BASE`
3. **Databricks SDK**: Automatic authentication using configured credentials

#### 2. Enhanced Model Hub Endpoint

**File**: `litellm/proxy/public_endpoints/public_endpoints.py`

Updated the `/public/model_hub` endpoint to support a `provider` query parameter.

**Features**:
- When `provider=databricks` is specified and Databricks credentials are available, fetches actual endpoints from Databricks
- Converts Databricks endpoint data to `ModelGroupInfoProxy` format
- Falls back to default behavior (LiteLLM cost map) if credentials are missing or for other providers
- Graceful error handling with appropriate fallbacks

### Frontend Integration

#### Updated Networking Functions

**File**: `ui/litellm-dashboard/src/components/networking.tsx`

Added two new functions:

1. **`getDatabricksServingEndpoints()`**: Fetches Databricks endpoints with optional credential parameters
2. **Enhanced `modelHubPublicModelsCall()`**: Now accepts optional `provider` parameter to filter by provider

### Tests

**File**: `tests/test_litellm/proxy/test_databricks_model_hub.py`

Comprehensive test suite covering:
- API key authentication
- OAuth M2M authentication
- Missing credentials error handling
- Model hub integration with Databricks provider
- Proper endpoint formatting
- API error handling

### Documentation

#### 1. User Documentation

**File**: `docs/my-website/docs/proxy/databricks_model_hub.md`

Complete user-facing documentation including:
- Configuration instructions for all authentication methods
- API endpoint reference with examples
- Python SDK usage examples
- Troubleshooting guide
- Security considerations
- Benefits and limitations

#### 2. Example Code

**File**: `examples/databricks_model_hub_example.py`

Runnable Python example demonstrating:
- Fetching Databricks endpoints
- Getting models from model hub
- Using discovered models for completions
- Error handling and validation

## Usage

### Quick Start

1. **Set Databricks Credentials** (choose one method):

   ```bash
   # Method A: Personal Access Token
   export DATABRICKS_API_KEY="dapi..."
   export DATABRICKS_API_BASE="https://adb-xxx.azuredatabricks.net/serving-endpoints"
   
   # Method B: OAuth M2M (Recommended for production)
   export DATABRICKS_CLIENT_ID="your-service-principal-id"
   export DATABRICKS_CLIENT_SECRET="your-secret"
   export DATABRICKS_API_BASE="https://adb-xxx.azuredatabricks.net/serving-endpoints"
   ```

2. **Fetch Databricks Endpoints**:

   ```bash
   curl http://localhost:4000/public/databricks/serving_endpoints \
     -H "Authorization: Bearer YOUR_LITELLM_API_KEY"
   ```

3. **Get Models from Model Hub**:

   ```bash
   curl http://localhost:4000/public/model_hub?provider=databricks \
     -H "Authorization: Bearer YOUR_LITELLM_API_KEY"
   ```

### API Endpoints

#### GET `/public/databricks/serving_endpoints`

Fetches Databricks serving endpoints from the Databricks API.

**Query Parameters** (optional):
- `api_key`: Databricks API key
- `api_base`: Databricks workspace URL
- `client_id`: OAuth client ID
- `client_secret`: OAuth client secret

**Response**:
```json
{
  "endpoints": [
    {
      "name": "databricks-dbrx-instruct",
      "creator": "user@company.com",
      "creation_timestamp": 1234567890,
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

#### GET `/public/model_hub?provider=databricks`

Gets model hub information, optionally filtered by provider.

**Query Parameters**:
- `provider`: Filter by provider (e.g., "databricks")

**Response**:
```json
[
  {
    "model_group": "databricks/databricks-dbrx-instruct",
    "providers": ["databricks"],
    "mode": "chat",
    "supports_function_calling": true,
    "supports_vision": false,
    "is_public_model_group": true
  }
]
```

## Architecture

```
┌─────────────────┐
│   UI Dashboard  │
│  (Model Hub)    │
└────────┬────────┘
         │
         │ GET /public/model_hub?provider=databricks
         ▼
┌─────────────────────────┐
│  LiteLLM Proxy Server   │
│  (public_endpoints.py)  │
└────────┬────────────────┘
         │
         │ Check Databricks credentials
         │
         ├─── If credentials available ───┐
         │                                 │
         │                                 ▼
         │                    ┌────────────────────────┐
         │                    │ GET /api/2.0/          │
         │                    │ serving-endpoints      │
         │                    │                        │
         │                    │ Databricks API         │
         │                    └────────────────────────┘
         │                                 │
         │                                 │ Returns endpoints
         │                                 ▼
         │                    ┌────────────────────────┐
         │                    │ Format as              │
         │                    │ ModelGroupInfoProxy    │
         │                    └────────────────────────┘
         │
         └─── If no credentials ─────────────┐
                                             │
                                             ▼
                                ┌────────────────────────┐
                                │ Return models from     │
                                │ LiteLLM cost map       │
                                └────────────────────────┘
```

## Security

- All sensitive credentials (API keys, tokens, secrets) are automatically redacted from logs using `DatabricksBase.redact_sensitive_data()`
- OAuth M2M authentication is recommended for production deployments
- Credentials are never stored; they're only used for API calls
- Headers are redacted in debug logs showing only first 8 characters

## Benefits

1. **Dynamic Model Discovery**: Automatically see all deployed models without manual configuration
2. **No Manual Updates**: New endpoints appear automatically
3. **Better Organization**: Filter and view only Databricks models
4. **Accurate Metadata**: See endpoint state, creation time, and configuration
5. **Custom Models**: Discover and use fine-tuned or custom deployed models

## Limitations

1. Requires Databricks credentials to be configured
2. Only shows serving endpoints (not all available models in Databricks)
3. Pricing information not available from Databricks API (falls back to cost map if needed)
4. Endpoint must be deployed and in "READY" state to be usable

## Future Enhancements

Potential improvements for future iterations:

1. **Caching**: Cache Databricks endpoint list to reduce API calls
2. **Model Metadata**: Extract more metadata from endpoint config (model size, workload type, etc.)
3. **Auto-refresh**: Periodically refresh endpoint list in background
4. **Endpoint Status**: Show detailed status and health metrics
5. **Cost Estimation**: Integrate with Databricks pricing API if/when available
6. **UI Filters**: Add UI controls to filter by endpoint state, creator, etc.

## Testing

Run the test suite:

```bash
pytest tests/test_litellm/proxy/test_databricks_model_hub.py -v
```

Run the example:

```bash
export DATABRICKS_API_KEY="your-key"
export DATABRICKS_API_BASE="https://adb-xxx.azuredatabricks.net/serving-endpoints"
export LITELLM_API_KEY="your-litellm-key"

python examples/databricks_model_hub_example.py
```

## Troubleshooting

### "Missing Databricks credentials" error

Ensure you've set one of the following:
- `DATABRICKS_API_KEY` + `DATABRICKS_API_BASE`, or
- `DATABRICKS_CLIENT_ID` + `DATABRICKS_CLIENT_SECRET` + `DATABRICKS_API_BASE`, or
- Have the Databricks SDK installed and configured

### No endpoints returned

- Verify your Databricks credentials are correct
- Check that you have serving endpoints deployed in your workspace
- Ensure your credentials have permission to list serving endpoints
- Test directly with Databricks API:
  ```bash
  curl https://your-workspace.databricks.net/api/2.0/serving-endpoints \
    -H "Authorization: Bearer YOUR_TOKEN"
  ```

### Authentication fails

- For PAT: Verify your token hasn't expired
- For OAuth M2M: Check that your service principal has necessary permissions
- Check the LiteLLM proxy logs for detailed error messages (credentials will be redacted)

## Related Documentation

- [Databricks Provider Setup](docs/my-website/docs/providers/databricks.md)
- [Databricks Model Hub Documentation](docs/my-website/docs/proxy/databricks_model_hub.md)
- [LiteLLM Model Hub](docs/my-website/docs/features/model_hub.md)

## Implementation Details

### Key Files Modified

1. **Backend**:
   - `litellm/proxy/public_endpoints/public_endpoints.py`: Core implementation
   - `litellm/llms/databricks/common_utils.py`: Authentication utilities (already existed)

2. **Frontend**:
   - `ui/litellm-dashboard/src/components/networking.tsx`: API client functions

3. **Tests**:
   - `tests/test_litellm/proxy/test_databricks_model_hub.py`: Comprehensive test suite

4. **Documentation**:
   - `docs/my-website/docs/proxy/databricks_model_hub.md`: User documentation
   - `examples/databricks_model_hub_example.py`: Usage example

### Code Quality

- ✅ Type hints throughout
- ✅ Comprehensive error handling
- ✅ Security: Credential redaction in logs
- ✅ Async support using httpx
- ✅ Falls back gracefully on errors
- ✅ Comprehensive test coverage
- ✅ Documentation and examples

## Contributing

When adding support for other providers (AWS Bedrock, Azure, etc.), follow this pattern:

1. Create a new endpoint: `/public/{provider}/endpoints`
2. Implement authentication using provider-specific utilities
3. Transform provider response to `ModelGroupInfoProxy` format
4. Update `/public/model_hub` to handle new provider filter
5. Add tests, documentation, and examples

## License

This feature follows the same license as LiteLLM (MIT License).
