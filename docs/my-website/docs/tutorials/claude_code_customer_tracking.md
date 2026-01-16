# Claude Code - Granular Cost Tracking

Track Claude Code usage by customer or tags using LiteLLM proxy. This enables granular cost attribution for billing, budgeting, and analytics.

## How It Works

Claude Code supports custom headers via `ANTHROPIC_CUSTOM_HEADERS`. LiteLLM automatically tracks requests with specific headers for cost attribution.

## Tracking Options

Choose how you want to attribute costs:

| Track By | Header | Use Case |
|----------|--------|----------|
| Customer | `x-litellm-customer-id` | Bill customers, per-user budgets |
| Tags | `x-litellm-tags` | Project tracking, cost centers, environments |

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `ANTHROPIC_BASE_URL` | LiteLLM proxy URL | `http://localhost:4000` |
| `ANTHROPIC_API_KEY` | LiteLLM API key | `sk-1234` |
| `ANTHROPIC_CUSTOM_HEADERS` | Custom headers (`header-name: value` format) | See examples below |

## Option 1: Track by Customer

Use this to attribute costs to specific customers or end-users.

```bash
export ANTHROPIC_BASE_URL=http://localhost:4000
export ANTHROPIC_API_KEY=sk-1234
export ANTHROPIC_CUSTOM_HEADERS="x-litellm-customer-id: claude-ishaan-local"
```

## Option 2: Track by Tags

Use this to attribute costs to projects, cost centers, or environments. Pass comma-separated tags.

```bash
export ANTHROPIC_BASE_URL=http://localhost:4000
export ANTHROPIC_API_KEY=sk-1234
export ANTHROPIC_CUSTOM_HEADERS="x-litellm-tags: project:acme,env:prod,team:backend"
```


## Quick Start

### 1. Set Environment Variables

```bash
export ANTHROPIC_BASE_URL=http://localhost:4000
export ANTHROPIC_API_KEY=sk-1234
export ANTHROPIC_CUSTOM_HEADERS="x-litellm-customer-id: claude-ishaan-local"
```

### 2. Use Claude Code

```bash
claude
```

All requests will now be tracked under the customer ID `claude-ishaan-local`.

![](https://colony-recorder.s3.amazonaws.com/files/2026-01-16/8f45872e-2d00-4d01-bf3d-4d6ae11d1396/ascreenshot_d2a745b8da4f4a56aaf2cac02871ef53_text_export.jpeg)

![](https://colony-recorder.s3.amazonaws.com/files/2026-01-16/dd41eae3-2592-4bc9-a8d2-d6d02614cd2d/ascreenshot_43ec9ee48ad946cca49732f007e786fc_text_export.jpeg)

![](https://colony-recorder.s3.amazonaws.com/files/2026-01-16/0c30309e-7117-4999-a3df-d22a2d5629c1/ascreenshot_d76a48c53b9a4fad8f6727baf4aa6a9c_text_export.jpeg)

### 3. View Usage in LiteLLM UI

Navigate to the **Logs** tab in the LiteLLM UI.

![](https://colony-recorder.s3.amazonaws.com/files/2026-01-16/ff774392-69f5-483e-83e2-fb749c94ee90/ascreenshot_d264fc04c9ee47edb047f61b6eb8c4d7_text_export.jpeg)

Click on a request to see details.

![](https://colony-recorder.s3.amazonaws.com/files/2026-01-16/5f71589b-5fdd-4759-9b6e-e6874be0eb21/ascreenshot_92dd86dadccb4764b1169c29c10dfe65_text_export.jpeg)

Filter by customer ID to see all requests for that customer.

![](https://colony-recorder.s3.amazonaws.com/files/2026-01-16/dd1c8aba-e75b-4714-9eee-c785e9db99af/ascreenshot_36aaec0fe12f4189b64f704a551e6729_text_export.jpeg)

## Supported Headers

| Header | Description |
|--------|-------------|
| `x-litellm-customer-id` | Track by customer/end-user ID |
| `x-litellm-end-user-id` | Alternative customer ID header |
| `x-litellm-tags` | Comma-separated tags for cost attribution |

## Related

- [Claude Code Quickstart](./claude_responses_api.md)
- [Customer Budgets](../proxy/customers.md)
- [Tag Budgets](../proxy/tag_budgets.md)
- [Track Usage for Coding Tools](./cost_tracking_coding.md)

