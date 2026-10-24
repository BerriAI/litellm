# RealmLabs MLS Guardrail Integration

Checks prompts and model replies with RealmLabs MLS. It blocks hazardous prompts and masks detected PII as `[type]`, or blocks PII when masking is disabled

## Configuration

Add the guardrail to your proxy configuration. The [complete example](example_config.yaml) also configures Claude Haiku 4.5 as the model

```yaml
guardrails:
  - guardrail_name: realmlabs-guard
    litellm_params:
      guardrail: realmlabs
      mode: [pre_call, post_call]
      api_key: os.environ/REALMLABS_API_KEY
      default_on: true
      probes: [hazard_prompt]
      hazard_threshold: 0.703
      pii: true
      pii_mask: true
      block_on_error: false
      optional_params:
        enable_thinking: false
        timeout: 15
```

### Credentials and endpoint

| Setting | Meaning |
| --- | --- |
| `api_key` | Required MLS guardrail bearer token; falls back to `REALMLABS_API_KEY` when omitted |
| `api_base` | MLS base URL; falls back to `REALMLABS_API_BASE`, then `https://mls.realmlabs.ai` |

The integration appends `/guardrail` to the base URL. This route uses a guardrail API key, separate from the token for `/llm/*` routes

### Tuning parameters

| Setting | Default | Meaning |
| --- | --- | --- |
| `probes` | `[hazard_prompt]` | Probe names or `all`; only the `hazard_prompt` score is enforced |
| `hazard_threshold` | `0.703` | Block request scores strictly above this value |
| `pii` | `true` | Ask MLS to detect PII |
| `pii_mask` | `true` | Mask detected PII; `false` blocks instead |
| `block_on_error` | `false` | Allow MLS failures through; `true` blocks them |
| `enable_thinking` | `false` | Ask MLS to use thinking mode in its chat template |
| `timeout` | `15` seconds | HTTP timeout for each MLS call, separate from the model request timeout |

All seven tuning parameters support top-level values under `litellm_params` or nested `optional_params`. An explicit, non-null nested value wins, then the top-level value, then the RealmLabs default. False, zero, and empty lists remain explicit overrides

Current configuration parsing limits nested `optional_params.timeout` to 1–60 seconds. For a value outside that range, set top-level `timeout` and omit the nested timeout

## Usage examples

With the proxy running using the example config, send a non-streaming Chat Completions request. Set `LITELLM_MASTER_KEY` in the client terminal to your proxy key

```bash
curl --silent --show-error --include http://localhost:4000/v1/chat/completions \
  --header "Authorization: Bearer ${LITELLM_MASTER_KEY}" \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "claude-haiku-4-5",
    "messages": [{"role": "user", "content": "My name is Alex and my email is alex@example.com. What do you know about me?"}],
    "max_tokens": 100
  }'
```

If MLS permits the prompt and detects both values, the model receives `My name is [name] and my email is [email]. What do you know about me?`. Detections and model replies can vary

| MLS verdict | Result |
| --- | --- |
| No blocking hazard or detected PII | Text passes through unchanged |
| PII detected with `pii_mask: true` | Matching text is masked; overlapping matches are merged |
| PII detected with `pii_mask: false` | The request or reply is blocked |
| Request hazard score exceeds the threshold | The request is blocked before the model is called |

The example's `default_on: true` applies the guardrail automatically. For opt-in use, set it to `false` and include `"guardrails": ["realmlabs-guard"]` in each request that should be checked

## Supported event hooks

| Hook | Behavior |
| --- | --- |
| `pre_call` | Checks the request, enforcing hazard before masking or blocking PII |
| `post_call` | Checks the reply with conversation context, masking or blocking PII; hazard scores do not block replies |

A hazard verdict marked `role_mismatch` is not enforced. Conversation context comes from Chat Completions-style `messages`. Streaming uses LiteLLM's existing delivery settings, which do not enable text rewrites by default

## Error handling

Hazard and PII policy blocks raise `GuardrailRaisedException` with HTTP status 400. A hazard block includes `Blocked by RealmLabs hazard_prompt probe` and the score and threshold

An MLS connection failure, timeout, HTTP error, or unreadable response follows `block_on_error`: the default `false` passes text through unchanged; `true` raises an error. A successful completion alone does not prove MLS returned an allow verdict

Responses must include `results` and `pii_spans` arrays, which may be empty. Each probe result needs a name and a finite probability between zero and one. Missing verdict fields are errors, not clean results. When masking is enabled, a PII span without a nonempty type and text also follows `block_on_error`. When masking is disabled, spans without text still trigger a PII content block

Request fields are defined by `RealmLabsGuardrailRequest` and assembled by `_build_request`. Response types and `_parse_response` define what blocking and masking consume. Additional response fields are ignored, including fields inside probe results and PII spans. Update these boundaries and their behavior tests when adding supported fields or changing the contract

## Unit tests

From the repository root with the development environment installed:

```bash
LITELLM_LOCAL_MODEL_COST_MAP=True .venv/bin/python -m pytest \
  tests/unit/proxy/guardrails/guardrail_hooks/realmlabs -q
```

Tests cover policy decisions, overlapping PII, response scanning, MLS failures, and configuration precedence using simulated MLS HTTP responses
