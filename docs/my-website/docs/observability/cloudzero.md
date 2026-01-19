import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# CloudZero Integration

LiteLLM provides an integration with CloudZero's AnyCost API, allowing you to export your LLM usage data to CloudZero for cost tracking analysis.

## Overview

| Property | Details |
|----------|---------|
| Description | Export LiteLLM usage data to CloudZero AnyCost API for cost tracking and analysis |
| callback name | `cloudzero`|
| Supported Operations | • Automatic hourly data export<br/>• Manual data export<br/>• Dry run testing<br/>• Cost and token usage tracking |
| Data Format | CloudZero Billing Format (CBF) with proper resource tagging |
| Export Frequency | Hourly (configurable via `CLOUDZERO_EXPORT_INTERVAL_MINUTES`) |

## Environment Variables

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `CLOUDZERO_API_KEY` | Yes | Your CloudZero API key | `cz_api_xxxxxxxxxx` |
| `CLOUDZERO_CONNECTION_ID` | Yes | CloudZero connection ID for data submission | `conn_xxxxxxxxxx` |
| `CLOUDZERO_TIMEZONE` | No | Timezone for date handling (default: UTC) | `America/New_York` |
| `CLOUDZERO_EXPORT_INTERVAL_MINUTES` | No | Export frequency in minutes (default: 60) | `60` |

## Setup

### End to End Video Walkthrough
This video walks through the entire process of setting up LiteLLM with CloudZero integration and viewing LiteLLM exported usage data in CloudZero.

<iframe width="840" height="500" src="https://www.loom.com/embed/59b57593183f4cc3b1c05a2dd3277f92" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>

### Step 1: Configure Environment Variables

Set your CloudZero credentials in your environment:

```bash
export CLOUDZERO_API_KEY="cz_api_xxxxxxxxxx"
export CLOUDZERO_CONNECTION_ID="conn_xxxxxxxxxx"
export CLOUDZERO_TIMEZONE="UTC"  # Optional, defaults to UTC
```

### Step 2: Enable CloudZero Integration

Add the CloudZero callback to your LiteLLM configuration YAML file:


```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: sk-xxxxxxx

litellm_settings:
  callbacks: ["cloudzero"]  # Enable CloudZero integration
```

### Step 3: Start LiteLLM Proxy

Start your LiteLLM proxy with the configuration:

```bash
litellm --config /path/to/config.yaml
```

## Setup on UI

1\. Click "Settings"

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-22/5ac36280-c688-41a3-8d0e-23e19c6a470b/ascreenshot.jpeg?tl_px=0,332&br_px=1308,1064&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=119,444)


2\. Click "Logging & Alerts"

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-22/13f76b09-e0c4-4738-ba05-2d5111c6ad3e/ascreenshot.jpeg?tl_px=0,332&br_px=1308,1064&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=58,507)


3\. Click "CloudZero Cost Tracking"

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-22/f96cc1e5-7bc0-4d7c-9aeb-5cbbec549b12/ascreenshot.jpeg?tl_px=0,0&br_px=1308,731&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=389,56)


4\. Click "Add CloudZero Integration"

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-22/04fbc748-0e6f-43bb-8a57-dd2e83dbfcb5/ascreenshot.jpeg?tl_px=0,90&br_px=1308,821&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=616,277)


5\. Enter your CloudZero API Key.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-22/080e82f1-f94f-4ed7-8014-e495380336f3/ascreenshot.jpeg?tl_px=0,0&br_px=1308,731&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=506,129)


6\. Enter your CloudZero Connection ID.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-22/af417aa2-67a8-4dee-a014-84b1892dc07e/ascreenshot.jpeg?tl_px=0,0&br_px=1308,731&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=488,213)


7\. Click "Create"

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-22/647e672f-9a4a-4754-a7b0-abf1397abad4/ascreenshot.jpeg?tl_px=0,88&br_px=1308,819&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=711,277)


8\. Test your payload with "Run Dry Run Simulation" 

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-22/7447cbe0-3450-4be5-bdc4-37fb8280aa58/ascreenshot.jpeg?tl_px=0,125&br_px=1308,856&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=334,277)


10\. Click "Export Data Now" to export to CLoudZero

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-22/7be9bd48-6e27-4c68-bc75-946f3ab593d9/ascreenshot.jpeg?tl_px=0,130&br_px=1308,861&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=518,277)

## Testing Your Setup

### Dry Run Export

Call the dry run endpoint to test your CloudZero configuration without sending data to CloudZero. This endpoint will not send any data to CloudZero, but will return the data that would be exported.

```bash
curl -X POST "http://localhost:4000/cloudzero/dry-run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "limit": 10
  }' | jq
```

**Expected Response:**
```json
{
  "message": "CloudZero dry run export completed successfully.",
  "status": "success",
  "dry_run_data": {
    "usage_data": [...],
    "cbf_data": [...],
    "summary": {
      "total_cost": 0.05,
      "total_tokens": 1250,
      "total_records": 10
    }
  }
}
```

### Manual Export

Call the export endpoint to send data immediately to CloudZero. We suggest setting a small `limit` to test the export. This will only export the last 10 records to CloudZero. Note: Cloudzero can take up to 15 minutes to process the exported data.

```bash
curl -X POST "http://localhost:4000/cloudzero/export" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "limit": 10
  }' | jq
```

**Expected Response:**
```json
{
  "message": "CloudZero export completed successfully",
  "status": "success"
}
```

## Data Export Details

### Automatic Export Schedule

- **Frequency**: Every 60 minutes (configurable via `CLOUDZERO_EXPORT_INTERVAL_MINUTES`)
- **Data Processing**: LiteLLM automatically processes and exports usage data hourly
- **CloudZero Processing**: CloudZero typically takes 10-15 minutes to process data from LiteLLM

### Data Format

LiteLLM exports data in CloudZero Billing Format (CBF) with the following structure:

```json
{
  "time/usage_start": "2024-01-15T14:00:00Z",
  "cost/cost": 0.002,
  "usage/amount": 150,
  "usage/units": "tokens",
  "resource/id": "czrn:litellm:openai:cross-region:team-123:llm-usage:gpt-4o",
  "resource/service": "litellm",
  "resource/account": "team-123",
  "resource/region": "cross-region",
  "resource/usage_family": "llm-usage",
  "resource/tag:provider": "openai",
  "resource/tag:model": "gpt-4o",
  "resource/tag:prompt_tokens": "100",
  "resource/tag:completion_tokens": "50"
}
```

### Resource Tagging

LiteLLM automatically creates comprehensive resource tags for cost attribution:

- **Provider Tags**: `openai`, `anthropic`, `azure`, etc.
- **Model Tags**: Specific model names like `gpt-4o`, `claude-3-sonnet`
- **Team/User Tags**: Team IDs and user IDs for cost allocation
- **Token Breakdown**: Separate tracking of prompt and completion tokens
- **Usage Metrics**: Total tokens consumed per request

## Advanced Configuration

### Custom Export Frequency

Change the export frequency (not recommended to go below 60 minutes):

```bash
export CLOUDZERO_EXPORT_INTERVAL_MINUTES=120  # Export every 2 hours
```

### Custom Time Range Export

Export data for a specific time range:

```bash
curl -X POST "http://localhost:4000/cloudzero/export" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "start_time_utc": "2024-01-15T00:00:00Z",
    "end_time_utc": "2024-01-15T23:59:59Z",
    "operation": "replace_hourly"
  }' | jq
```

## Troubleshooting

### Common Issues

1. **Missing Credentials Error**
   ```
   CloudZero configuration missing. Please set CLOUDZERO_API_KEY and CLOUDZERO_CONNECTION_ID environment variables.
   ```
   **Solution**: Ensure both environment variables are set with valid values.

2. **Connection Issues**
   - Verify your CloudZero API key is valid
   - Check that the connection ID exists in your CloudZero account
   - Ensure your proxy has internet access to reach CloudZero's API

3. **No Data in CloudZero**
   - CloudZero can take 10-15 minutes to process data
   - Check that your LiteLLM proxy is generating usage data
   - Use the dry-run endpoint to verify data is being formatted correctly

## Related Links

- [CloudZero Documentation](https://docs.cloudzero.com/)
- [CloudZero AnyCost API](https://docs.cloudzero.com/reference/anycost-api)
