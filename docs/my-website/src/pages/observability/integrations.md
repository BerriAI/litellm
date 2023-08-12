# Logging Integrations

| Integration     | Required OS Variables                      | How to Use with callbacks             |
|-----------------|--------------------------------------------|-------------------------------------------|
| Sentry          | `SENTRY_API_URL`                          | `litellm.success_callback=["sentry"]`  |
| Posthog         | `POSTHOG_API_KEY`,`POSTHOG_API_URL`   | `litellm.success_callback=["posthog"]` |
| Slack           | `SLACK_API_TOKEN`,`SLACK_API_SECRET`,`SLACK_API_CHANNEL` | `litellm.success_callback=["slack"]`      |
| Helicone           | `HELICONE_API_TOKEN` | `litellm.success_callback=["helicone"]`      |




