# Llama Guard Guardrail

Content-safety guardrail backed by a [Llama Guard](https://github.com/meta-llama/PurpleLlama/tree/main/Llama-Guard3) classifier model. It screens request and/or response text against Meta's MLCommons hazard taxonomy (S1–S14) and blocks content the model flags as `unsafe`.

The Llama Guard model is called through LiteLLM itself, so any provider that serves a Llama Guard model works: `together_ai`, `groq`, `fireworks_ai`, `ollama`, `huggingface`, self-hosted `hosted_vllm`, etc.

## Features

- Pre-call (screen the user prompt), during-call (parallel moderation), and post-call (screen the model's response) modes
- Enforce the full MLCommons taxonomy or a subset of hazard categories
- Fully custom category block via `unsafe_content_categories`
- Provider-agnostic: the classification prompt is self-contained, so it does not depend on the serving provider applying a Llama Guard chat template
- Fails open (logs and allows the request) if the classifier is unreachable, so an outage of the safety model does not take down traffic
- Violation errors name the exact hazard categories that were triggered

## Configuration

### Required parameters

- `model`: the Llama Guard model to call, e.g. `together_ai/meta-llama/Llama-Guard-4-12B`, `groq/llama-guard-3-8b`, `ollama/llama-guard3`.

### Optional parameters

- `api_base`: base URL for the Llama Guard model endpoint.
- `api_key`: API key for the Llama Guard model endpoint (`os.environ/...` is supported).
- `categories`: list of hazard codes to enforce (e.g. `["S1", "S10", "S11"]`). Defaults to the full `S1`–`S14` taxonomy.
- `unsafe_content_categories`: a fully custom category block that overrides the built-in taxonomy text.
- `default_on` (default `false`): apply this guardrail to every request without opting in per request.

### Hazard taxonomy (default)

| Code | Category | Code | Category |
|------|----------|------|----------|
| S1 | Violent Crimes | S8 | Intellectual Property |
| S2 | Non-Violent Crimes | S9 | Indiscriminate Weapons |
| S3 | Sex-Related Crimes | S10 | Hate |
| S4 | Child Sexual Exploitation | S11 | Suicide & Self-Harm |
| S5 | Defamation | S12 | Sexual Content |
| S6 | Specialized Advice | S13 | Elections |
| S7 | Privacy | S14 | Code Interpreter Abuse |

## Usage examples

### Screen the user prompt (pre-call)

```yaml
guardrails:
  - guardrail_name: "llama-guard-input"
    litellm_params:
      guardrail: llama_guard
      mode: pre_call
      default_on: true
      model: together_ai/meta-llama/Llama-Guard-4-12B
      api_key: os.environ/TOGETHERAI_API_KEY
```

### Screen the model's response (post-call)

```yaml
guardrails:
  - guardrail_name: "llama-guard-output"
    litellm_params:
      guardrail: llama_guard
      mode: post_call
      default_on: true
      model: groq/llama-guard-3-8b
      api_key: os.environ/GROQ_API_KEY
```

### Enforce only a subset of categories

```yaml
guardrails:
  - guardrail_name: "llama-guard-strict"
    litellm_params:
      guardrail: llama_guard
      mode: pre_call
      model: ollama/llama-guard3
      api_base: http://localhost:11434
      categories: ["S1", "S9", "S11"]   # violent crimes, weapons, self-harm
```

When Llama Guard flags a request, LiteLLM raises a `content_policy_violation` error that lists the triggered categories, for example:

```
Violated Llama Guard content policy. Categories: S9 (Indiscriminate Weapons)
```
