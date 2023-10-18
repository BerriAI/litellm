# Logging Integrations

| Integration | Required OS Variables                                    | How to Use with callbacks                |
| ----------- | -------------------------------------------------------- | ---------------------------------------- |
| Promptlayer   | `PROMPLAYER_API_KEY`                                   | `litellm.success_callback=["promptlayer"]` |
| LLMonitor   | `LLMONITOR_APP_ID`                                       | `litellm.success_callback=["llmonitor"]` |
| LangFuse   | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_PRIVATE_KEY`             | `litellm.success_callback=["langfuse"]` |
| Weights & Biases   | `WANDB_API_KEY`            | `litellm.success_callback=["wandb"]` |
| Sentry      | `SENTRY_API_URL`                                         | `litellm.success_callback=["sentry"]`    |
| Posthog     | `POSTHOG_API_KEY`,`POSTHOG_API_URL`                      | `litellm.success_callback=["posthog"]`   |
| Slack       | `SLACK_API_TOKEN`,`SLACK_API_SECRET`,`SLACK_API_CHANNEL` | `litellm.success_callback=["slack"]`     |
| Traceloop    | `TRACELOOP_API_TOKEN`                                     | `litellm.success_callback=["traceloop"]`  |
| Helicone    | `HELICONE_API_TOKEN`                                     | `litellm.success_callback=["helicone"]`  |
