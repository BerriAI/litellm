---
title: "Add Model Pricing & Context Window"
---

If you just want to add pricing or context window information for a model (without integrating as a full provider), simply make a PR to this file:

**[model_prices_and_context_window.json](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json)**

### Format

```json
{
  "model_name": {
    "max_tokens": 8192,
    "max_input_tokens": 8192,
    "max_output_tokens": 4096,
    "input_cost_per_token": 0.00000025,
    "output_cost_per_token": 0.00000125,
    "litellm_provider": "provider_name",
    "mode": "chat"
  }
}
```

### Example

```json
{
  "gpt-4": {
    "max_tokens": 8192,
    "max_input_tokens": 8192,
    "max_output_tokens": 4096,
    "input_cost_per_token": 0.00003,
    "output_cost_per_token": 0.00006,
    "litellm_provider": "openai",
    "mode": "chat"
  }
}
```

That's it! Your PR will be reviewed and merged.
