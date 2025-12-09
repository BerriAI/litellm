import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# FOCUS Format Export

LiteLLM supports exporting cost and usage data in [FOCUS (FinOps Open Cost & Usage Specification)](https://focus.finops.org/) format, an open specification for consistent cost and usage datasets from the FinOps Foundation.

## Overview

| Property | Details |
|----------|---------|
| Description | Export LiteLLM usage data in FOCUS format for FinOps tools |
| Supported Operations | • JSON export<br/>• CSV export<br/>• Dry run testing |
| Data Format | FOCUS 1.0 specification compliant |
| Compatible Tools | APTIO, CloudHealth, Apptio Cloudability, and other FinOps tools |

## What is FOCUS?

FOCUS (FinOps Open Cost & Usage Specification) is an open specification developed by the FinOps Foundation to standardize cloud cost and usage data. It provides a common schema that enables:

- Consistent cost reporting across multiple cloud providers and services
- Easy integration with FinOps tools and platforms
- Simplified cost allocation and chargeback processes
- Better visibility into AI/LLM spending

Learn more at [focus.finops.org](https://focus.finops.org/)

## Setup

### Prerequisites

- LiteLLM Proxy with a connected database
- Admin API key for authentication

No additional environment variables are required for FOCUS export.

### Optional Environment Variables

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `FOCUS_EXPORT_TIMEZONE` | No | Timezone for date handling | `UTC` |

## API Endpoints

### Export as JSON

Export usage data in FOCUS format as JSON:

```bash
curl -X POST "http://localhost:4000/focus/export" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "limit": 1000,
    "include_tags": true,
    "include_token_breakdown": true
  }' | jq
```

**Request Parameters:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `limit` | integer | Maximum records to export | No limit |
| `start_time_utc` | datetime | Start time filter (ISO format) | None |
| `end_time_utc` | datetime | End time filter (ISO format) | None |
| `include_tags` | boolean | Include resource tags | `true` |
| `include_token_breakdown` | boolean | Include token breakdown in tags | `true` |

**Example Response:**

```json
{
  "message": "FOCUS export completed successfully",
  "status": "success",
  "format": "json",
  "data": {
    "focus_version": "1.0",
    "export_timestamp": "2024-01-15T14:30:00Z",
    "record_count": 100,
    "records": [
      {
        "BilledCost": 0.002,
        "BillingPeriodStart": "2024-01-15T00:00:00",
        "BillingPeriodEnd": "2024-01-16T00:00:00",
        "ChargeCategory": "Usage",
        "ChargeClass": "Standard",
        "ChargeDescription": "LLM inference using gpt-4o via openai",
        "ConsumedQuantity": 150,
        "ConsumedUnit": "Tokens",
        "EffectiveCost": 0.002,
        "ListCost": 0.002,
        "ProviderName": "OpenAI",
        "PublisherName": "LiteLLM",
        "ResourceId": "litellm/openai/team-123/gpt-4o",
        "ResourceName": "gpt-4o",
        "ResourceType": "LLM",
        "ServiceCategory": "AI and Machine Learning",
        "ServiceName": "LLM Inference",
        "SubAccountId": "team-123",
        "SubAccountName": "Engineering Team",
        "Tags": {
          "litellm:provider": "openai",
          "litellm:model": "gpt-4o",
          "litellm:prompt_tokens": "100",
          "litellm:completion_tokens": "50"
        }
      }
    ]
  },
  "summary": {
    "total_records": 100,
    "total_billed_cost": 0.50,
    "total_consumed_quantity": 50000,
    "unique_providers": 3,
    "unique_sub_accounts": 5
  }
}
```

### Export as CSV

Export usage data in FOCUS format as CSV:

```bash
curl -X POST "http://localhost:4000/focus/export/csv" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "limit": 1000
  }' -o focus_export.csv
```

The CSV export is ideal for importing into spreadsheets or FinOps tools that accept CSV data.

### Dry Run Export

Test the export without committing any changes:

```bash
curl -X POST "http://localhost:4000/focus/dry-run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "limit": 100
  }' | jq
```

**Example Response:**

```json
{
  "message": "FOCUS dry run export completed successfully",
  "status": "success",
  "raw_data_sample": [...],
  "focus_data": [...],
  "summary": {
    "total_records": 100,
    "total_billed_cost": 0.50,
    "total_consumed_quantity": 50000,
    "unique_providers": 3,
    "unique_sub_accounts": 5
  }
}
```

### Get Schema Information

Get documentation about the FOCUS columns:

```bash
curl "http://localhost:4000/focus/schema" \
  -H "Authorization: Bearer sk-1234" | jq
```

## FOCUS Columns

### Required Columns

| Column | Description | Example |
|--------|-------------|---------|
| `BilledCost` | The cost that is invoiced/billed | `0.002` |
| `BillingPeriodStart` | Start of billing period | `2024-01-15T00:00:00` |
| `BillingPeriodEnd` | End of billing period | `2024-01-16T00:00:00` |

### Recommended Columns

| Column | Description | Example |
|--------|-------------|---------|
| `ChargeCategory` | Type of charge | `Usage` |
| `ChargeClass` | Classification of charge | `Standard` |
| `ChargeDescription` | Human-readable description | `LLM inference using gpt-4o` |
| `ChargePeriodStart` | Start of charge period | `2024-01-15T00:00:00` |
| `ChargePeriodEnd` | End of charge period | `2024-01-16T00:00:00` |
| `ConsumedQuantity` | Amount consumed (tokens) | `150` |
| `ConsumedUnit` | Unit of consumption | `Tokens` |
| `EffectiveCost` | Cost after discounts | `0.002` |
| `ListCost` | Cost at list prices | `0.002` |
| `ProviderName` | LLM provider name | `OpenAI` |
| `PublisherName` | Publisher name | `LiteLLM` |
| `Region` | Geographic region (if applicable) | `us-east-1` |
| `ResourceId` | Unique resource identifier | `litellm/openai/team-123/gpt-4o` |
| `ResourceName` | Model name | `gpt-4o` |
| `ResourceType` | Type of resource | `LLM` |
| `ServiceCategory` | Service category | `AI and Machine Learning` |
| `ServiceName` | Service name | `LLM Inference` |
| `SubAccountId` | Sub-account ID (team_id) | `team-123` |
| `SubAccountName` | Sub-account name (team_alias) | `Engineering Team` |
| `Tags` | Additional metadata tags | See below |

### LiteLLM Tags

When `include_tags` is enabled, the following tags are included:

| Tag | Description |
|-----|-------------|
| `litellm:provider` | LLM provider identifier |
| `litellm:model` | Model identifier |
| `litellm:model_group` | Model group name |
| `litellm:user_id` | User ID |
| `litellm:api_key_prefix` | First 8 chars of API key |
| `litellm:api_key_alias` | API key alias |
| `litellm:prompt_tokens` | Number of prompt tokens |
| `litellm:completion_tokens` | Number of completion tokens |
| `litellm:api_requests` | Number of API requests |
| `litellm:successful_requests` | Number of successful requests |
| `litellm:failed_requests` | Number of failed requests |
| `litellm:cache_creation_tokens` | Cache creation tokens |
| `litellm:cache_read_tokens` | Cache read tokens |

## Integration with FinOps Tools

### APTIO

To import FOCUS data into APTIO:

1. Export data using the CSV endpoint
2. Upload the CSV file to APTIO's data import interface
3. Map the FOCUS columns to APTIO's cost model

### Other FinOps Tools

Most FinOps tools that support the FOCUS specification can import the exported data. Consult your tool's documentation for specific import procedures.

## Time Range Filtering

Export data for a specific time range:

```bash
curl -X POST "http://localhost:4000/focus/export" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "start_time_utc": "2024-01-01T00:00:00Z",
    "end_time_utc": "2024-01-31T23:59:59Z"
  }' | jq
```

## Troubleshooting

### Common Issues

1. **No Data Returned**
   - Verify that usage data exists in the database
   - Check that the time range filter includes data
   - Use the dry-run endpoint to debug

2. **Authentication Errors**
   - Ensure you're using an admin API key
   - Verify the API key is valid

3. **Database Connection Issues**
   - Verify database connection is configured
   - Check database connectivity

## Related Links

- [FOCUS Specification](https://focus.finops.org/)
- [FinOps Foundation](https://www.finops.org/)
- [CloudZero Integration](./cloudzero.md) - For automated cost tracking with CloudZero
- [Prometheus Metrics](./prometheus.md) - For real-time cost monitoring
