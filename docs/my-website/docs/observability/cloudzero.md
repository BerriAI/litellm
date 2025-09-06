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

## Features

- **Automatic Data Export**: LiteLLM automatically exports usage data to CloudZero every hour
- **Cost Tracking**: Track costs by model, team, user, and other dimensions
- **Token Usage Analytics**: Monitor prompt tokens, completion tokens, and total token consumption
- **Resource Tagging**: Comprehensive tagging with CloudZero Resource Names (CZRNs) for detailed cost attribution
- **Dry Run Testing**: Test your configuration without sending data to CloudZero
- **Manual Export**: On-demand data export with customizable time ranges

## Prerequisites

1. **CloudZero Account**: Active CloudZero account with API access
2. **CloudZero API Key**: API key from your CloudZero account settings
3. **CloudZero Connection ID**: Connection ID for data submission (created in CloudZero dashboard)
4. **LiteLLM Proxy**: Running LiteLLM proxy with database enabled

## Configuration

### Environment Variables

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `CLOUDZERO_API_KEY` | Yes | Your CloudZero API key | `cz_api_xxxxxxxxxx` |
| `CLOUDZERO_CONNECTION_ID` | Yes | CloudZero connection ID for data submission | `conn_xxxxxxxxxx` |
| `CLOUDZERO_TIMEZONE` | No | Timezone for date handling (default: UTC) | `America/New_York` |
| `CLOUDZERO_EXPORT_INTERVAL_MINUTES` | No | Export frequency in minutes (default: 60) | `60` |

## Setup

### End to End Video Walkthrough


<iframe width="840" height="500" src="https://www.loom.com/embed/59b57593183f4cc3b1c05a2dd3277f92" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>

### Step 1: Get CloudZero Credentials

This video walks through:
1. Creating a CloudZero connection ID
2. Setting up LiteLLM with CloudZero integration
3. Running a dry run to test the configuration

### Step 2: Configure Environment Variables

Set your CloudZero credentials in your environment:

```bash
export CLOUDZERO_API_KEY="cz_api_xxxxxxxxxx"
export CLOUDZERO_CONNECTION_ID="conn_xxxxxxxxxx"
export CLOUDZERO_TIMEZONE="UTC"  # Optional, defaults to UTC
```

### Step 3: Enable CloudZero Integration

Add the CloudZero callback to your LiteLLM configuration:

<Tabs>
<TabItem value="config" label="config.yaml">

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: sk-xxxxxxx

litellm_settings:
  callbacks: ["cloudzero"]  # Enable CloudZero integration
```

</TabItem>
<TabItem value="env" label="Environment Variables">

```bash
export LITELLM_CALLBACKS="cloudzero"
```

</TabItem>
<TabItem value="python" label="Python SDK">

```python
import litellm
import os

# Set your CloudZero credentials
os.environ["CLOUDZERO_API_KEY"] = "cz_api_xxxxxxxxxx"
os.environ["CLOUDZERO_CONNECTION_ID"] = "conn_xxxxxxxxxx"

# Enable CloudZero callback
litellm.callbacks = ["cloudzero"]

# Make LLM requests as usual
response = litellm.completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

</TabItem>
</Tabs>

### Step 4: Start LiteLLM Proxy

Start your LiteLLM proxy with the configuration:

```bash
litellm --config /path/to/config.yaml
```

## Testing Your Setup

### Dry Run Export

Test your CloudZero configuration without sending data to CloudZero:

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

Send data immediately to CloudZero:

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

### Debug Mode

Enable verbose logging to see detailed CloudZero export information:

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import litellm
litellm.set_verbose = True
```

</TabItem>
<TabItem value="proxy" label="PROXY">

```bash
export LITELLM_LOG="DEBUG"
```

</TabItem>
</Tabs>

This will show:
- CloudZero export job scheduling
- Data transformation details
- API request/response information
- Error details if exports fail

## Data Processing Timeline

1. **LiteLLM Request**: User makes an LLM API request
2. **Usage Logging**: LiteLLM logs usage data to database
3. **Hourly Export**: CloudZero logger exports data every hour
4. **Data Transformation**: Usage data is converted to CBF format
5. **CloudZero Submission**: Data is sent to CloudZero AnyCost API
6. **CloudZero Processing**: CloudZero processes data (10-15 minutes)
7. **Dashboard Availability**: Data appears in CloudZero dashboard

## Related Links

- [CloudZero Documentation](https://docs.cloudzero.com/)
- [CloudZero AnyCost API](https://docs.cloudzero.com/reference/anycost-api)
- [LiteLLM Observability](https://docs.litellm.ai/docs/observability/)
