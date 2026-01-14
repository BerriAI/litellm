import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Azure Sentinel

<Image img={require('../../img/sentinel.png')} />

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

```yaml showLineNumbers title="config.yaml"
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

```shell showLineNumbers title="Environment Variables"
# Required: Data Collection Rule (DCR) configuration
AZURE_SENTINEL_DCR_IMMUTABLE_ID="dcr-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # DCR Immutable ID from Azure portal
AZURE_SENTINEL_STREAM_NAME="Custom-LiteLLM_CL_CL"                    # Stream name from your DCR
AZURE_SENTINEL_ENDPOINT="https://your-dcr-endpoint.eastus-1.ingest.monitor.azure.com"  # DCR logs ingestion endpoint (NOT the DCE endpoint)

# Required: OAuth2 Authentication (App Registration)
AZURE_SENTINEL_TENANT_ID="your-tenant-id"                            # Azure Tenant ID
AZURE_SENTINEL_CLIENT_ID="your-client-id"                            # Application (client) ID
AZURE_SENTINEL_CLIENT_SECRET="your-client-secret"                    # Client secret value

```

**Note**: The `AZURE_SENTINEL_ENDPOINT` should be the DCR's logs ingestion endpoint (found in the DCR Overview page), NOT the Data Collection Endpoint (DCE). The DCR endpoint is associated with your specific DCR and looks like: `https://your-dcr-endpoint.{region}-1.ingest.monitor.azure.com`

**Step 4**: Start the proxy and make a test request

Start proxy

```shell showLineNumbers title="Start Proxy"
litellm --config config.yaml --debug
```

Test Request

```shell showLineNumbers title="Test Request"
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

```kusto showLineNumbers title="KQL Query"
LiteLLM_CL
| where TimeGenerated > ago(1h)
| project TimeGenerated, model, status, total_tokens, response_cost
| order by TimeGenerated desc
```

You should see following logs in Azure Workspace.

<Image img={require('../../img/sentinel.png')} />

## Environment Variables

| Environment Variable | Description | Default Value | Required |
|---------------------|-------------|---------------|----------|
| `AZURE_SENTINEL_DCR_IMMUTABLE_ID` | Data Collection Rule (DCR) Immutable ID | None | ✅ Yes |
| `AZURE_SENTINEL_ENDPOINT` | DCR logs ingestion endpoint URL (from DCR Overview page) | None | ✅ Yes |
| `AZURE_SENTINEL_STREAM_NAME` | Stream name from DCR (e.g., "Custom-LiteLLM_CL_CL") | "Custom-LiteLLM" | ❌ No |
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

## Azure Sentinel Setup Guide

Follow this step-by-step guide to set up Azure Sentinel with LiteLLM.

### Step 1: Create a Log Analytics Workspace

1. Navigate to [https://portal.azure.com/#home](https://portal.azure.com/#home)

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-17/5659f6f5-a166-4b26-a991-73352274e3bb/ascreenshot.jpeg?tl_px=0,210&br_px=2618,1673&force_format=jpeg&q=100&width=1120.0)

2. Search for "Log Analytics workspaces" and click "Create"

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-17/a827ba10-a391-486a-a36a-51816c6255de/ascreenshot.jpeg?tl_px=0,0&br_px=2618,1463&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=21,106)

3. Enter a name for your workspace (e.g., "litellm-sentinel-prod")

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-17/943458f1-fd4c-47dd-a273-ea5a04734ed9/ascreenshot.jpeg?tl_px=0,420&br_px=2618,1884&force_format=jpeg&q=100&width=1120.0)

4. Click "Review + Create"

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-17/c54828fb-f895-4eb7-b810-cacf437617bd/ascreenshot.jpeg?tl_px=0,420&br_px=2618,1884&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=40,564)

### Step 2: Create a Custom Table

1. Go to your Log Analytics workspace and click "Tables"

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-17/72d65f70-75c0-471f-95e9-947c72e173cc/ascreenshot.jpeg?tl_px=0,142&br_px=2618,1605&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=330,277)

2. Click "Create" → "New custom log (Direct Ingest)"

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-17/863ad29b-2c3a-4b7c-9a6b-36d3a76c9f32/ascreenshot.jpeg?tl_px=0,0&br_px=2618,1463&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=526,146)

3. Enter a table name (e.g., "LITELLM_PROD_CL")

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-17/ef2f1c52-aa36-46a1-91e6-9bd868891b15/ascreenshot.jpeg?tl_px=0,0&br_px=2618,1463&force_format=jpeg&q=100&width=1120.0)

### Step 3: Create a Data Collection Rule (DCR)

1. Click "Create a new data collection rule"

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-17/f2abc0d3-8be8-4057-9290-946d10cfd183/ascreenshot.jpeg?tl_px=0,420&br_px=2618,1884&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=264,404)

2. Enter a name for the DCR (e.g., "litellm-prod")

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-17/79bbebdc-e4d9-46ff-a270-1930619050a1/ascreenshot.jpeg?tl_px=0,8&br_px=2618,1471&force_format=jpeg&q=100&width=1120.0)

3. Select a Data Collection Endpoint

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-17/f3112e9a-551e-415c-a7f9-55aad801bc8a/ascreenshot.jpeg?tl_px=0,420&br_px=2618,1884&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=332,480)

4. Upload the sample JSON file for schema (use the [example_standard_logging_payload.json](https://github.com/BerriAI/litellm/blob/main/litellm/integrations/azure_sentinel/example_standard_logging_payload.json) file)

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-17/703c0762-840a-4f1f-a60f-876dc24b7a03/ascreenshot.jpeg?tl_px=0,0&br_px=2618,1463&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=518,272)

5. Click "Next" and then "Create"

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-17/0bca0200-5c64-4fbd-8061-9308aa6656b8/ascreenshot.jpeg?tl_px=0,420&br_px=2618,1884&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=128,560)

### Step 4: Get the DCR Immutable ID and Logs Ingestion Endpoint

1. Go to "Data Collection Rules" and select your DCR

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-17/11c06a0d-584f-4d22-b36e-9c338d43812c/ascreenshot.jpeg?tl_px=0,0&br_px=2618,1463&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=94,258)

2. Copy the **DCR Immutable ID** (starts with `dcr-`)

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-17/cd0ad69a-4d95-4b6a-9533-7720908ba809/ascreenshot.jpeg?tl_px=1160,92&br_px=2618,907&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=530,277)

3. Copy the **Logs Ingestion Endpoint** URL

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-17/3d3752ed-08ea-4490-8c98-a97d33947ea7/ascreenshot.jpeg?tl_px=1160,464&br_px=2618,1279&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=532,277)

### Step 5: Get the Stream Name

1. Click "JSON View" in the DCR

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-17/fd8a5504-4769-4f23-983e-520f256ee308/ascreenshot.jpeg?tl_px=1160,0&br_px=2618,814&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=965,257)

2. Find the **Stream Name** in the `streamDeclarations` section (e.g., "Custom-LITELLM_PROD_CL_CL")

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-17/a4052b32-2028-4d12-8930-bfcdf6f47652/ascreenshot.jpeg?tl_px=405,270&br_px=2115,1225&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=523,277)

### Step 6: Register an App and Grant Permissions

1. Go to **Microsoft Entra ID** → **App registrations** → **New registration**
2. Create a new app and note the **Client ID** and **Tenant ID**
3. Go to **Certificates & secrets** → Create a new client secret and copy the **Secret Value**
4. Go back to your DCR → **Access Control (IAM)** → **Add role assignment**
5. Assign the **"Monitoring Metrics Publisher"** role to your app registration

### Summary: Where to Find Each Value

| Environment Variable | Where to Find It |
|---------------------|------------------|
| `AZURE_SENTINEL_DCR_IMMUTABLE_ID` | DCR Overview page → Immutable ID (starts with `dcr-`) |
| `AZURE_SENTINEL_ENDPOINT` | DCR Overview page → Logs Ingestion Endpoint |
| `AZURE_SENTINEL_STREAM_NAME` | DCR JSON View → `streamDeclarations` section |
| `AZURE_SENTINEL_TENANT_ID` | App Registration → Overview → Directory (tenant) ID |
| `AZURE_SENTINEL_CLIENT_ID` | App Registration → Overview → Application (client) ID |
| `AZURE_SENTINEL_CLIENT_SECRET` | App Registration → Certificates & secrets → Secret Value |

For more details, refer to the [Microsoft Logs Ingestion API documentation](https://learn.microsoft.com/en-us/azure/azure-monitor/logs/logs-ingestion-api-overview).
