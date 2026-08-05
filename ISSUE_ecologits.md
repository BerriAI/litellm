# Feature: EcoLogits callback — enrich LLM-call metrics with green-cost data

## The feature

We propose to enrich LiteLLM's existing LLM-call metrics with **green-cost metrics**, via an optional EcoLogits callback plugin. On every successful call, the plugin queries the [EcoLogits API](https://api.ecologits.ai/docs) and makes the resulting environmental impacts available to **all downstream observability callbacks** (Langfuse, Datadog, Prometheus, …).

For this first "proxy mode" version, the principle is simple:

- **Per-model zone config.** When registering a model, admins can set a `model_info` argument `ecologits_electricity_mix_zone` (e.g. `FRA`, `DEU`) reflecting where the model physically runs. If omitted, the plugin falls back to the `ECOLOGITS_ELECTRICITY_MIX_ZONE` env var, and finally to EcoLogits' own default (`WOR` — world average).
- **Per-call enrichment.** On each successful call to a provider, the plugin calls the EcoLogits API (public, or an enterprise self-hosted instance via `ECOLOGITS_API_BASE`) and attaches the impacts to the call payload. It runs in the `async_logging_hook`, *before* other callbacks see the data — so any downstream integration the user enables gets the green metrics for free. The enrichment is written where each consumer looks: `litellm_params.metadata` (Langfuse) and `standard_logging_object.metadata` (Datadog, SpendLogs). For Prometheus, dedicated green-cost Counters (energy kWh, GWP kgCO2eq, …) are exposed alongside.
- **Never fatal.** Any failure (timeout, non-200, bad payload) is swallowed — at worst the call is logged without enrichment.

## Motivation

Today LiteLLM lets organizations make deployment and quota decisions on cost and billing, but not on the **ecological cost** of those decisions. For a growing number of organizations this is no longer optional: incoming regulations will require Data/AI departments to report the environmental impact of their LLM usage.

[EcoLogits](https://ecologits.ai/latest/) is a non-profit project backed by institutional and non-institutional partners and hosted by [CodeCarbon](https://codecarbon.io/). Its [LLM-inference methodology](https://ecologits.ai/latest/methodology/llm_inference/) sits at the crossing of state-of-the-art research estimates, real data-center measurements, and common MLOps practice — a credible approximation in a world where major providers don't disclose their model architectures (with acknowledged limitations).

By delegating the calculation to EcoLogits, the green-cost estimation is **decoupled from LiteLLM itself**: LiteLLM developers don't have to maintain a state-of-the-art calculator, and users automatically benefit from future EcoLogits improvements. This integration can of course coexist with any native LiteLLM green-cost evaluation.

## Example configuration

```yaml
litellm_settings:
  # EcoLogits must be registered first so it enriches the payload
  # before any downstream callback observes it.
  callbacks: ["ecologits", "langfuse"]

model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
    model_info:
      ecologits_electricity_mix_zone: "FRA"  # optional; defaults to "WOR"
```

## Scope & limitations

- Proxy mode only for this first version.
- Impacts are **estimates** produced by the EcoLogits methodology, not measured values; accuracy is bounded by what providers disclose about their models.
