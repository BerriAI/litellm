---
sidebar_label: Overview
---

# Webhook Integrations

LiteLLM supports sending logs to various webhook-based services. These integrations use the Generic API Callback system and require minimal configuration.

:::info Auto-Generated

This page is auto-generated from [`generic_api_compatible_callbacks.json`](https://github.com/BerriAI/litellm/blob/main/litellm/integrations/generic_api/generic_api_compatible_callbacks.json).

:::

## Available Integrations

| Integration | Description | Events |
|-------------|-------------|--------|
| [Rubrik](./rubrik.md) | Rubrik is a data security platform. Send LLM logs to Rubrik for data governan... | `llm_api_success` |
| [Sumo Logic](./sumologic.md) | Sumo Logic is a cloud-native machine data analytics platform. Send LLM logs t... | `llm_api_success, llm_api_failure` |

## Quick Setup

All webhook integrations follow the same pattern:

1. Set the required environment variables
2. Add the callback name to your config
3. Start making requests

```yaml
litellm_settings:
  callbacks: ["callback_name"]

environment_variables:
  CALLBACK_WEBHOOK_URL: "https://..."
  CALLBACK_API_KEY: "sk-..."  # if required
```

## Adding New Integrations

Want to add support for a new webhook service? See the [Contributing Guide](../../contribute_integration/custom_webhook_api.md).

Just add your service to `generic_api_compatible_callbacks.json` and the documentation will be auto-generated!
