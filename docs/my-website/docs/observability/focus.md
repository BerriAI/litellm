import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Focus Export (Experimental)

:::caution Experimental feature
Focus Format export is under active development and currently considered experimental.
Interfaces, schema mappings, and configuration options may change as we iterate based on user feedback.
Please treat this integration as a preview and report any issues or suggestions to help us stabilize and improve the workflow.
:::

LiteLLM can emit usage data in the [FinOps FOCUS format](https://focus.finops.org/focus-specification/v1-2/) and push artifacts (for example Parquet files) to destinations such as Amazon S3. This enables downstream cost-analysis tooling to ingest a standardised dataset directly from LiteLLM.

LiteLLM currently conforms to the FinOps FOCUS v1.2 specification when emitting this dataset.

## Overview

| Property | Details |
|----------|---------|
| Destination | Export LiteLLM usage data in FOCUS format to managed storage (currently S3) |
| Callback name | `focus` |
| Supported operations | Automatic scheduled export |
| Data format | FOCUS Normalised Dataset (Parquet) |

## Environment Variables

### Common settings

| Variable | Required | Description |
|----------|----------|-------------|
| `FOCUS_PROVIDER` | No | Destination provider (defaults to `s3`). |
| `FOCUS_FORMAT` | No | Output format (currently only `parquet`). |
| `FOCUS_FREQUENCY` | No | Export cadence. Prefer `hourly` or `daily` for production; `interval` is intended for short test loops. Defaults to `hourly`. |
| `FOCUS_CRON_OFFSET` | No | Minute offset used for hourly/daily cron triggers. Defaults to `5`. |
| `FOCUS_INTERVAL_SECONDS` | No | Interval (seconds) when `FOCUS_FREQUENCY="interval"`. |
| `FOCUS_PREFIX` | No | Object key prefix/folder. Defaults to `focus_exports`. |

### S3 destination

| Variable | Required | Description |
|----------|----------|-------------|
| `FOCUS_S3_BUCKET_NAME` | Yes | Destination bucket for exported files. |
| `FOCUS_S3_REGION_NAME` | No | AWS region for the bucket. |
| `FOCUS_S3_ENDPOINT_URL` | No | Custom endpoint (useful for S3-compatible storage). |
| `FOCUS_S3_ACCESS_KEY` | Yes | AWS access key for uploads. |
| `FOCUS_S3_SECRET_KEY` | Yes | AWS secret key for uploads. |
| `FOCUS_S3_SESSION_TOKEN` | No | AWS session token if using temporary credentials. |

## Setup via Config

### Configure environment variables

```bash
export FOCUS_PROVIDER="s3"
export FOCUS_PREFIX="focus_exports"

# S3 example
export FOCUS_S3_BUCKET_NAME="my-litellm-focus-bucket"
export FOCUS_S3_REGION_NAME="us-east-1"
export FOCUS_S3_ACCESS_KEY="AKIA..."
export FOCUS_S3_SECRET_KEY="..."
```

### Update LiteLLM config

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: sk-your-key

litellm_settings:
  callbacks: ["focus"]
```

### Start the proxy

```bash
litellm --config /path/to/config.yaml
```

During boot LiteLLM registers the Focus logger and a background job that runs according to the configured frequency.

## Planned Enhancements
- Add "Setup on UI" flow alongside the current configuration-based setup.
- Add GCS / Azure Blob to the Destination options.
- Support CSV output alongside Parquet.

## Related Links

- [Focus](https://focus.finops.org/)

