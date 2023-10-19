# Logging Integrations

| Integration | Required OS Variables                                    | How to Use with callbacks                |
| ----------- | -------------------------------------------------------- | ---------------------------------------- |
| Promptlayer   | `PROMPLAYER_API_KEY`                                   | `litellm.success_callback=["promptlayer"]` |
| LLMonitor   | `LLMONITOR_APP_ID`                                       | `litellm.success_callback=["llmonitor"]` |
| LangFuse   | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_PRIVATE_KEY`             | `litellm.success_callback=["langfuse"]` |
| Langsmith   | `LANGSMITH_PUBLIC_KEY`             | `litellm.success_callback=["langsmith"]` |
| Weights & Biases   | `WANDB_API_KEY`            | `litellm.success_callback=["wandb"]` |
| Sentry      | `SENTRY_API_URL`                                         | `litellm.success_callback=["sentry"]`    |
| Posthog     | `POSTHOG_API_KEY`,`POSTHOG_API_URL`                      | `litellm.success_callback=["posthog"]`   |
| Slack       | `Slack webhook url` | `litellm.success_callback=["slack"]`     |
| Traceloop    | `TRACELOOP_API_TOKEN`                                     | `litellm.success_callback=["traceloop"]`  |
| Helicone    | `HELICONE_API_TOKEN`                                     | `litellm.success_callback=["helicone"]`  |
