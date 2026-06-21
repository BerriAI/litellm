import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Bias & Hallucination Estimator

A native, lightweight guardrail that detects bias and hallucinations in LLM responses using local text analysis. No external API calls required — runs in under 1ms per response using regex patterns and statistical analysis.

## Quick Start

### 1. Define the guardrail in your config.yaml

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

guardrails:
  - guardrail_name: "bias-hallucination-detector"
    litellm_params:
      guardrail: bias_hallucination_estimator
      mode: "post_call"
      default_on: true
```

### 2. Start LiteLLM Gateway

```shell
litellm --config config.yaml --detailed_debug
```

### 3. Test the guardrail

<Tabs>
<TabItem label="Blocked (hallucination risk)" value="blocked">

```shell
curl -i http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Tell me about AI adoption."}],
    "guardrails": ["bias-hallucination-detector"]
  }'
```

If the model responds with unsourced statistics or overconfident claims, the request is blocked:

```json
{
  "error": {
    "message": "High bias/hallucination risk detected (68%).",
    "type": "None",
    "param": "None",
    "code": "400"
  }
}
```

</TabItem>
<TabItem label="Allowed (clean response)" value="allowed">

```shell
curl -i http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "What is Python?"}],
    "guardrails": ["bias-hallucination-detector"]
  }'
```

```json
{
  "id": "chatcmpl-abc123",
  "model": "gpt-4o",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "Python is a high-level programming language known for its readable syntax..."
      },
      "finish_reason": "stop"
    }
  ]
}
```

</TabItem>
</Tabs>

## Supported `mode` values

| Mode | Behavior |
|------|----------|
| `post_call` | Analyze model output after the LLM responds (default, recommended) |
| `pre_call` | Analyze user input before the LLM call (requires `check_request: true`) |

## What it detects

### Bias indicators

| Pattern | Examples |
|---------|----------|
| `dogmatic_language` | "obviously", "everyone knows", "clearly", "the fact is" |
| `opinion_as_fact` | "I believe", "should be", "must be", "in my opinion" |
| `overconfidence` | "100%", "guaranteed", "definitely", "cannot be wrong" |
| `sweeping_generalization` | "all engineers are...", "no developer can..." |

### Hallucination risk indicators

| Pattern | Examples |
|---------|----------|
| `unsourced_statistics` | Numbers/percentages without an adjacent source indicator |
| `missing_citations` | "research shows", "studies found", "experts say" without a named source |
| `fabricated_specificity` | "exactly 1,234 cases", overly precise numbers, specific undocumented dates |

Sentences containing source indicators (`according to`, `published in`, `doi:`, `https://`, etc.) are excluded from unsourced-statistics checks.

## Risk scoring

Risk is computed as a weighted combination of bias and hallucination scores:

```
overall_risk = (bias_score * bias_weight + hallucination_score * hallucination_weight)
             / (bias_weight + hallucination_weight)
```

| Risk range | Action |
|------------|--------|
| 0–25% | Pass through |
| 25–50% | Flag (logged, not blocked) |
| >50% | Block (raises exception) |

Individual threshold checks run in addition to the weighted score — a response is also blocked if `bias_score >= bias_threshold` or `hallucination_score >= hallucination_threshold`.

## Configuration reference

```yaml
guardrails:
  - guardrail_name: "bias-hallucination-detector"
    litellm_params:
      guardrail: bias_hallucination_estimator
      mode: "post_call"
      default_on: true

      # Detection thresholds (0.0–1.0)
      bias_threshold: 0.5
      hallucination_threshold: 0.5

      # Risk action thresholds (0.0–1.0)
      risk_flag_threshold: 0.25
      risk_block_threshold: 0.5

      # Behavior
      block_on_high_risk: true    # set false to never block, only flag
      log_only: false             # set true to log all findings without blocking

      # What to check
      check_response: true        # analyze LLM output
      check_request: false        # analyze user input (pre_call mode)

      # Scoring weights
      bias_weight: 0.4
      hallucination_weight: 0.6

      # Custom block message (optional)
      violation_message: "Response blocked due to quality concerns."
```

### All parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bias_threshold` | `0.5` | Bias score that immediately triggers a block, regardless of weighted risk |
| `hallucination_threshold` | `0.5` | Hallucination score that immediately triggers a block |
| `risk_flag_threshold` | `0.25` | Weighted risk above which a response is flagged in logs |
| `risk_block_threshold` | `0.5` | Weighted risk above which a response is blocked |
| `block_on_high_risk` | `true` | Whether to raise an exception on high-risk responses |
| `log_only` | `false` | Log findings but never block, even on high risk |
| `check_response` | `true` | Analyze LLM output |
| `check_request` | `false` | Analyze user input before the LLM call |
| `bias_weight` | `0.4` | Weight of bias score in the combined risk formula |
| `hallucination_weight` | `0.6` | Weight of hallucination score in the combined risk formula |
| `violation_message` | `null` | Custom message returned when a response is blocked |
| `default_on` | `false` | Run on every request without specifying the guardrail in the request body |

## Log-only mode

Use `log_only: true` to observe findings without blocking anything. Risk scores and detected issues are attached to the standard logging payload and surface in DataDog, Langfuse, OpenTelemetry, and Prometheus:

```yaml
guardrails:
  - guardrail_name: "bias-hallucination-monitor"
    litellm_params:
      guardrail: bias_hallucination_estimator
      mode: "post_call"
      default_on: true
      log_only: true
```

## Per-request guardrail selection

Apply only to specific requests by including the guardrail name in the request body rather than using `default_on: true`:

```shell
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Summarize this article."}],
    "guardrails": ["bias-hallucination-detector"]
  }'
```

## Input + output pipeline

Run one guardrail on user input and another on the model response:

```yaml
guardrails:
  - guardrail_name: "input-bias-check"
    litellm_params:
      guardrail: bias_hallucination_estimator
      mode: "pre_call"
      check_request: true
      check_response: false
      bias_threshold: 0.6

  - guardrail_name: "output-hallucination-check"
    litellm_params:
      guardrail: bias_hallucination_estimator
      mode: "post_call"
      check_request: false
      check_response: true
      hallucination_threshold: 0.4
```

## Performance

| Mode | Latency | Notes |
|------|---------|-------|
| Baseline (no grounding) | <1ms | Pure regex and keyword analysis |
| Memory | <5MB | Compiled patterns loaded once at startup |

All detection runs in-process. There are no external API calls, no network I/O, and no ML model inference required.

## Observability

Every evaluation appends an entry to `standard_logging_guardrail_information`:

```json
{
  "guardrail_name": "bias-hallucination-detector",
  "guardrail_provider": "litellm_native",
  "guardrail_status": "guardrail_intervened",
  "guardrail_response": {
    "decision": "blocked",
    "input_type": "response",
    "risk_scores": [
      {
        "overall_risk_percentage": 68,
        "bias_score": 0.18,
        "hallucination_score": 0.96,
        "detected_issues": [
          "hallucination:unsourced_statistics",
          "hallucination:missing_citations"
        ],
        "recommendation": "block"
      }
    ]
  }
}
```

This payload surfaces in all connected logging integrations (Langfuse, DataDog, OTEL, etc.).
