import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Azure Sentinel

LiteLLM supports logging to Azure Sentinel via the Azure Monitor Logs Ingestion API. Azure Sentinel uses Log Analytics workspaces for data storage, so logs sent to the workspace will be available in Sentinel for security monitoring and analysis.

## Azure Sentinel Integration

| Feature | Details |
|---------|---------|
| **What is logged** | [StandardLoggingPayload](../proxy/logging_spec) |
| **Events** | Success + Failure |
| **Product Link** | [Azure Sentinel](https://learn.microsoft.com/en-us/azure/sentinel/overview) |
| **API Reference** | [Logs Ingestion API](https://learn.microsoft.com/en-us/azure/azure-monitor/logs/logs-ingestion-api-overview) |

We will use the `--config` to set `litellm.callbacks = ["azure_sentinel"]` this will log all successful and failed LLM calls to Azure Sentinel.

**Step 1**: Create a `config.yaml` file and set `litellm_settings`: `callbacks`

```yaml
model_list:
 - model_name: gpt-3.5-turbo
    litellm_params:
      model: gpt-3.5-turbo
litellm_settings:
  callbacks: ["azure_sentinel"] # logs llm success + failure logs to Azure Sentinel
```

**Step 2**: Set Up Azure Resources

Before using the Logs Ingestion API, you need to set up the following in Azure:

1. **Create a Log Analytics Workspace** (if you don't have one)
2. **Create a Custom Table** in your Log Analytics workspace (e.g., `LiteLLM_CL`)
3. **Create a Data Collection Rule (DCR)** with:
   - Stream declaration matching your data structure
   - Transformation to map data to your custom table
   - Access granted to your app registration
4. **Register an Application** in Microsoft Entra ID (Azure AD) with:
   - Client ID
   - Client Secret
   - Permissions to write to the DCR

For detailed setup instructions, see the [Microsoft documentation on Logs Ingestion API](https://learn.microsoft.com/en-us/azure/azure-monitor/logs/logs-ingestion-api-overview).

**Step 3**: Set Required Environment Variables

Set the following environment variables with your Azure credentials:

```shell
# Required: Data Collection Rule (DCR) configuration
AZURE_SENTINEL_DCR_IMMUTABLE_ID="dcr-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # DCR Immutable ID from Azure portal
AZURE_SENTINEL_STREAM_NAME="Custom-LiteLLM"                          # Stream name from your DCR (default: "Custom-LiteLLM")
AZURE_SENTINEL_ENDPOINT="https://your-dce.eastus-1.ingest.monitor.azure.com"  # DCE endpoint or DCR ingestion endpoint

# Required: OAuth2 Authentication (App Registration)
AZURE_SENTINEL_TENANT_ID="your-tenant-id"                            # Azure Tenant ID
AZURE_SENTINEL_CLIENT_ID="your-client-id"                            # Application (client) ID
AZURE_SENTINEL_CLIENT_SECRET="your-client-secret"                    # Client secret value

# Alternative: You can also use generic Azure env vars
# AZURE_TENANT_ID="your-tenant-id"
# AZURE_CLIENT_ID="your-client-id"
# AZURE_CLIENT_SECRET="your-client-secret"
```

**Note**: The `AZURE_SENTINEL_ENDPOINT` can be either:
- A Data Collection Endpoint (DCE) URL: `https://your-dce.eastus-1.ingest.monitor.azure.com`
- A DCR ingestion endpoint (if your DCR has one configured)

**Step 4**: Start the proxy and make a test request

Start proxy

```shell
litellm --config config.yaml --debug
```

Test Request

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
    --header 'Content-Type: application/json' \
    --data '{
    "model": "gpt-3.5-turbo",
    "messages": [
        {
        "role": "user",
        "content": "what llm are you"
        }
    ],
    "metadata": {
        "your-custom-metadata": "custom-field",
    }
}'
```

**Step 5**: View logs in Azure Sentinel

1. Navigate to your Azure Sentinel workspace in the Azure portal
2. Go to "Logs" and query your custom table (e.g., `LiteLLM_CL`)
3. Run a query like:

```kusto
LiteLLM_CL
| where TimeGenerated > ago(1h)
| project TimeGenerated, model, status, total_tokens, response_cost
| order by TimeGenerated desc
```

## Environment Variables

| Environment Variable | Description | Default Value | Required |
|---------------------|-------------|---------------|----------|
| `AZURE_SENTINEL_DCR_IMMUTABLE_ID` | Data Collection Rule (DCR) Immutable ID | None | ✅ Yes |
| `AZURE_SENTINEL_ENDPOINT` | Data Collection Endpoint (DCE) or DCR ingestion endpoint URL | None | ✅ Yes |
| `AZURE_SENTINEL_STREAM_NAME` | Stream name from DCR (e.g., "Custom-LiteLLM") | "Custom-LiteLLM" | ❌ No |
| `AZURE_SENTINEL_TENANT_ID` | Azure Tenant ID for OAuth2 authentication | None (falls back to `AZURE_TENANT_ID`) | ✅ Yes |
| `AZURE_SENTINEL_CLIENT_ID` | Application (client) ID for OAuth2 authentication | None (falls back to `AZURE_CLIENT_ID`) | ✅ Yes |
| `AZURE_SENTINEL_CLIENT_SECRET` | Client secret for OAuth2 authentication | None (falls back to `AZURE_CLIENT_SECRET`) | ✅ Yes |

## How It Works

The Azure Sentinel integration uses the [Azure Monitor Logs Ingestion API](https://learn.microsoft.com/en-us/azure/azure-monitor/logs/logs-ingestion-api-overview) to send logs to your Log Analytics workspace. The integration:

- Authenticates using OAuth2 client credentials flow with your app registration
- Sends logs to the Data Collection Rule (DCR) endpoint
- Batches logs for efficient transmission
- Sends logs in the [StandardLoggingPayload](../proxy/logging_spec) format
- Automatically handles both success and failure events
- Caches OAuth2 tokens and refreshes them automatically

Logs sent to the Log Analytics workspace are automatically available in Azure Sentinel for security monitoring, threat detection, and analysis.

## Setup Guide

For detailed instructions on setting up the required Azure resources (DCR, app registration, etc.), refer to the [Microsoft Logs Ingestion API documentation](https://learn.microsoft.com/en-us/azure/azure-monitor/logs/logs-ingestion-api-overview).
