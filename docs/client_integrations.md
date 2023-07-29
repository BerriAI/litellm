# Data Logging Integrations

| Integration     | Required OS Variables                      | How to Use with litellm Client             |
|-----------------|--------------------------------------------|-------------------------------------------|
| Sentry          | `SENTRY_API_URL`                           | `client = litellm_client(success_callback=["sentry"], failure_callback=["sentry"])`  |
| Posthog         | `POSTHOG_API_KEY`,<br>`POSTHOG_API_URL`   | `client = litellm_client(success_callback=["posthog"], failure_callback=["posthog"])` |
| Slack           | `SLACK_API_TOKEN`,<br>`SLACK_API_SECRET`,<br>`SLACK_API_CHANNEL` | `client = litellm_client(success_callback=["slack"], failure_callback=["slack"])`      |




