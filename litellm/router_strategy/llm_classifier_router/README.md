# LLM Classifier Router

A routing strategy that uses a small LLM to classify prompt complexity and route to a pre-configured tier model. Useful for cost optimization when you want semantic accuracy beyond keyword rules but don't have the budget for a frontier model to do the classification.

## How It Works

A lightweight classifier LLM (e.g. `ollama/qwen2.5:0.5b` running locally, or any cheap hosted model) reads each incoming prompt and outputs a one-word tier label. The router maps that label to a configured backend model. Results are cached by prompt hash, so repeat queries are free.

```
prompt  ──▶  classifier LLM  ──▶  "SIMPLE"  ──▶  gpt-4o-mini
                                              "COMPLEX" ──▶  gpt-4o
```

## Why Not Just Use complexity_router?

The rule-based `complexity_router` is faster (<1ms, no network) but only matches keyword signals. It misses semantically complex prompts that don't contain "step by step" or code keywords. The LLM classifier handles those cases at the cost of one extra small-model call (~100-300ms for a 0.5B model running locally).

## Configuration

### Minimal

```yaml
model_list:
  - model_name: cheap-model
    litellm_params:
      model: gpt-4o-mini

  - model_name: powerful-model
    litellm_params:
      model: gpt-4o

  - model_name: smart-router
    litellm_params:
      model: auto_router/llm_classifier_router
      llm_classifier_router_config:
        classifier_model: ollama/qwen2.5:0.5b
        tiers:
          SIMPLE: cheap-model
          COMPLEX: powerful-model
```

### Full

```yaml
  - model_name: smart-router
    litellm_params:
      model: auto_router/llm_classifier_router
      llm_classifier_router_config:
        classifier_model: ollama/qwen2.5:0.5b
        tiers:
          SIMPLE: gpt-4o-mini
          COMPLEX: gpt-4o
        classifier_timeout: 3.0
        classifier_temperature: 0.0
        classifier_max_tokens: 10
        enable_cache: true
        cache_ttl_seconds: 300
        fallback_tier: SIMPLE
        fallback_to_complexity_router: true
```

## Parameters

| Field | Default | Description |
|---|---|---|
| `classifier_model` | `ollama/qwen2.5:0.5b` | Any litellm-supported model. Smaller = faster + cheaper. |
| `tiers` | `{SIMPLE: gpt-4o-mini, COMPLEX: gpt-4o}` | Map of tier name to backend `model_name`. |
| `classifier_system_prompt` | Built-in 2-tier prompt | Override to use a 4-tier prompt or custom instructions. |
| `classifier_timeout` | `3.0` | Seconds before falling back. |
| `classifier_temperature` | `0.0` | Keep at 0 for deterministic routing. |
| `classifier_max_tokens` | `10` | Output is a single word; no need for more. |
| `classifier_max_input_chars` | `2000` | Truncates very long prompts before classification. |
| `enable_cache` | `True` | Cache by prompt hash to avoid repeat LLM calls. |
| `cache_ttl_seconds` | `300` | Cache entry lifetime. |
| `cache_max_size` | `1000` | Simple dict; on overflow, store is cleared. |
| `fallback_tier` | `SIMPLE` | Last-resort tier when LLM and rule fallback both fail. |
| `fallback_to_complexity_router` | `True` | Use rule-based ComplexityRouter as intermediate fallback. |

## Custom Tier Prompts

The default prompt asks for a 2-tier answer (SIMPLE/COMPLEX). For 4-tier classification, supply your own prompt and tier map. Note: 4-tier accuracy is poor on sub-1B models. Use 3B+ for that.

```yaml
llm_classifier_router_config:
  classifier_model: ollama/qwen2.5:3b
  classifier_system_prompt: |
    Classify the request. Reply with EXACTLY one word: SIMPLE, MEDIUM, COMPLEX, or REASONING.
    SIMPLE: greetings, lookups
    MEDIUM: standard questions
    COMPLEX: multi-step, code
    REASONING: chain-of-thought, analysis
  tiers:
    SIMPLE: gpt-4o-mini
    MEDIUM: gpt-4o-mini
    COMPLEX: gpt-4o
    REASONING: o1-mini
```

## Fallback Behavior

1. **Cache hit** — return cached tier immediately
2. **LLM call** — call `classifier_model` with `classifier_timeout` deadline
3. **ComplexityRouter rule fallback** — if LLM call fails/times out, use the rule-based ComplexityRouter
4. **`fallback_tier`** — if both above fail, route to the cheapest configured tier

Each fallback step is logged with `verbose_router_logger` and the decision method is exposed in `request_kwargs["metadata"]["llm_classifier_router_method"]` for debugging.

## Notes

- The classifier is called via `litellm.acompletion` directly, bypassing the Router, to avoid recursion if the user happens to name `classifier_model` after one of their own deployments.
- No new Python dependencies. All inference goes through litellm's existing provider routing.
- For local Ollama models, the `ollama/` prefix is required (e.g. `ollama/qwen2.5:0.5b`).
