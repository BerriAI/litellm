# Data Logging Integrations

| Integration     | Required OS Variables                      | How to Use with callbacks             |
|-----------------|--------------------------------------------|-------------------------------------------|
| Sentry          | `SENTRY_API_URL`                          | `litellm.success_callback=["sentry"], litellm.failure_callback=["sentry"]`  |
| Posthog         | `POSTHOG_API_KEY`,<br>`POSTHOG_API_URL`   | `litellm.success_callback=["posthog"], litellm.failure_callback=["posthog"]` |
| Slack           | `SLACK_API_TOKEN`,<br>`SLACK_API_SECRET`,<br>`SLACK_API_CHANNEL` | `litellm.success_callback=["slack"], litellm.failure_callback=["slack"]`      |
| Helicone           | `HELICONE_API_TOKEN` | `litellm.success_callback=["helicone"]`      |




