import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Endpoint Activity

Track and visualize API endpoint usage directly in the dashboard. Monitor endpoint-level activity analytics, spend breakdowns, and performance metrics to understand which endpoints are receiving the most traffic and how they're performing.

## Overview

Endpoint Activity enables you to track spend and usage for individual API endpoints automatically. Every time you call an endpoint through the LiteLLM proxy, activity is automatically tracked and aggregated. This allows you to:

- Track spend per endpoint automatically
- View endpoint-level usage analytics in the Admin UI
- Monitor token consumption by endpoint
- Analyze success and failure rates per endpoint
- Identify which endpoints are getting the most activity
- View trend data showing endpoint usage over time

<Image img={require('../../img/ui_endpoint_activity.png')} />

## How Endpoint Activity Works

Endpoint activity is **automatically tracked** whenever you make API calls through the LiteLLM proxy. No additional configuration is required - simply call your endpoints as usual and activity will be tracked.

### Example API Call

When you make a request to any endpoint, activity is automatically recorded:

```bash showLineNumbers title="Endpoint activity is automatically tracked"
curl -X POST 'http://0.0.0.0:4000/chat/completions' \ # 👈 ENDPOINT AUTOMATICALLY TRACKED
  --header 'Content-Type: application/json' \
  --header 'Authorization: Bearer sk-1234' \ # 👈 YOUR PROXY KEY
  --data '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {
        "role": "user",
        "content": "What is the capital of France?"
      }
    ]
  }'
```

The endpoint (`/chat/completions`) will be automatically tracked with:

- Token counts (prompt tokens, completion tokens, total tokens)
- Spend for the request
- Request status (success or failure)
- Timestamp and other metadata

## How to View Endpoint Activity

### View Activity in Admin UI

Navigate to the Endpoint Activity tab in the Admin UI to view endpoint-level analytics:

#### 1. Access Endpoint Activity

Go to the Usage page in the Admin UI (`PROXY_BASE_URL/ui/?login=success&page=new_usage`) and click on the **Endpoint Activity** tab.

![](https://colony-recorder.s3.amazonaws.com/files/2026-01-10/67601fc0-8415-49b4-8e55-0673d37540c2/ascreenshot_f609a506dfe745c5aadccd332681c32d_text_export.jpeg)

#### 2. View Endpoint Analytics

The Endpoint Activity dashboard provides:

- **Endpoint usage table**: View all endpoints with aggregated metrics including:
  - Total requests (successful and failed)
  - Success rate percentage
  - Total tokens consumed
  - Total spend per endpoint
- **Success vs Failed requests chart**: Visualize request success and failure rates by endpoint
- **Usage trends**: See how endpoint activity changes over time with daily trend data

![](https://colony-recorder.s3.amazonaws.com/files/2026-01-10/41b2b158-3ab3-4154-a0d0-7233451d3f2b/ascreenshot_ff46db6e09b54ea9bf34ae9028aff58a_text_export.jpeg)

![](https://colony-recorder.s3.amazonaws.com/files/2026-01-10/bce32f99-f0ba-4502-8a3a-76257ff5e47a/ascreenshot_2273d3a94acd42e983ad7d6436722c2a_text_export.jpeg)

#### 3. Understand Endpoint Metrics

Each endpoint displays the following metrics:

- **Successful Requests**: Number of requests that completed successfully
- **Failed Requests**: Number of requests that encountered errors
- **Total Requests**: Sum of successful and failed requests
- **Success Rate**: Percentage of successful requests
- **Total Tokens**: Sum of prompt and completion tokens
- **Spend**: Total cost for all requests to that endpoint

## Use Cases

### Performance Monitoring

Monitor endpoint health and performance:

- Identify endpoints with high failure rates
- Track which endpoints are receiving the most traffic
- Monitor token consumption patterns by endpoint
- Detect anomalies in endpoint usage

### Cost Optimization

Understand spend distribution across endpoints:

- Identify high-cost endpoints
- Optimize expensive endpoints
- Allocate budget based on endpoint usage
- Track cost trends over time

---

## Related Features

- [Customer Usage](./customer_usage.md) - Track spend and usage for individual customers
- [Cost Tracking](./cost_tracking.md) - Comprehensive cost tracking and analytics
- [Spend Logs](./spend_logs.md) - Detailed request-level spend logs
