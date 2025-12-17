import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Azure Sentinel

LiteLLM supports logging to Azure Sentinel via the Azure Log Analytics HTTP Data Collector API. Azure Sentinel uses Log Analytics workspaces for data storage, so logs sent to the workspace will be available in Sentinel for security monitoring and analysis.

## Azure Sentinel Integration

| Feature | Details |
|---------|---------|
| **What is logged** | [StandardLoggingPayload](../proxy/logging_spec) |
| **Events** | Success + Failure |
| **Product Link** | [Azure Sentinel](https://learn.microsoft.com/en-us/azure/sentinel/overview) |

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

**Step 2**: Set Required Environment Variables

You need to obtain your Azure Log Analytics Workspace credentials from the Azure portal:

1. Navigate to your Log Analytics workspace in the Azure portal
2. Under "Settings" > "Agents management" > "Log Analytics agent instructions"
3. Note your **Workspace ID** and **Primary Key** (or Secondary Key)

Set the following environment variables:

```shell
AZURE_SENTINEL_WORKSPACE_ID="your-workspace-id"           # Your Log Analytics Workspace ID
AZURE_SENTINEL_SHARED_KEY="your-primary-or-secondary-key" # Your Primary or Secondary Key (base64 encoded)
AZURE_SENTINEL_LOG_TYPE="LiteLLM"                        # [OPTIONAL] Custom log type name (default: "LiteLLM")
```

**Note**: The `AZURE_SENTINEL_SHARED_KEY` should be the base64-encoded key from your Azure Log Analytics workspace. This is the same key shown in the Azure portal under "Agents management".

**Step 3**: Start the proxy and make a test request

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

**Step 4**: View logs in Azure Sentinel

1. Navigate to your Azure Sentinel workspace in the Azure portal
2. Go to "Logs" and query your custom log type (default: `LiteLLM`)
3. Run a query like:

```kusto
LiteLLM
| where TimeGenerated > ago(1h)
| project TimeGenerated, model, status, total_tokens, response_cost
| order by TimeGenerated desc
```

## Environment Variables

| Environment Variable | Description | Default Value | Required |
|---------------------|-------------|---------------|----------|
| `AZURE_SENTINEL_WORKSPACE_ID` | Your Azure Log Analytics Workspace ID | None | ✅ Yes |
| `AZURE_SENTINEL_SHARED_KEY` | Your Primary or Secondary Key (base64 encoded) | None | ✅ Yes |
| `AZURE_SENTINEL_LOG_TYPE` | Custom log type name (table name in Log Analytics) | "LiteLLM" | ❌ No |

## How It Works

The Azure Sentinel integration uses the [Azure Log Analytics HTTP Data Collector API](https://learn.microsoft.com/en-us/azure/azure-monitor/logs/data-collector-api) to send logs to your Log Analytics workspace. The integration:

- Authenticates using HMAC-SHA256 signature with your workspace ID and shared key
- Batches logs for efficient transmission
- Sends logs in the [StandardLoggingPayload](../proxy/logging_spec) format
- Automatically handles both success and failure events

Logs sent to the Log Analytics workspace are automatically available in Azure Sentinel for security monitoring, threat detection, and analysis.

