import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# semantix-ai Integration

[semantix-ai](https://github.com/labrat-akhona/semantix-ai) is a semantic type system for AI outputs. It validates that LLM responses match natural language intents using local NLI (Natural Language Inference) models — no external API calls, no added latency beyond ~15ms, and zero per-validation cost.

## What is semantix-ai?

semantix-ai lets you express what an LLM output *should mean* as a plain English intent string, then enforces it at runtime. Under the hood it runs a quantized NLI model locally to check entailment between the model's response and your declared intent.

Key characteristics:

- **Local inference** — NLI model runs on-device; no data leaves your stack
- **~15ms validation** — negligible overhead on top of any LLM call
- **Zero API cost** — validation is entirely compute-local
- **Provider-agnostic** — wraps `litellm.completion()`, so it works with all 100+ LiteLLM-supported models
- **Self-training collector** — mismatches are logged for fine-tuning or threshold calibration

## Prerequisites

```bash
pip install semantix-ai litellm
```

## Quick Start

### Basic validation with `@validate_intent`

The `@validate_intent` decorator intercepts the output of any function that returns a string (or a LiteLLM `ModelResponse`) and checks it against a declared intent.

```python
import litellm
from semantix import validate_intent

@validate_intent("The response confirms the booking was successful")
def book_flight(origin: str, destination: str) -> str:
    response = litellm.completion(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": f"Confirm a flight booking from {origin} to {destination}."
            }
        ]
    )
    return response.choices[0].message.content

# Will raise SemanticValidationError if the output does not entail the intent
result = book_flight("JFK", "LAX")
print(result)
```

### Switching providers

Because semantix-ai wraps `litellm.completion()`, you can swap the underlying model without changing your validation logic:

```python
import litellm
from semantix import validate_intent

@validate_intent("The response provides a concise summary of the article")
def summarise(article: str, model: str = "gpt-4o") -> str:
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": f"Summarise: {article}"}]
    )
    return response.choices[0].message.content

# Works identically across providers
summarise(article_text, model="gpt-4o")                          # OpenAI
summarise(article_text, model="anthropic/claude-3-5-sonnet-20241022")  # Anthropic
summarise(article_text, model="azure/gpt-4o")                    # Azure OpenAI
summarise(article_text, model="gemini/gemini-1.5-pro")           # Google
summarise(article_text, model="groq/llama3-70b-8192")            # Groq
```

### Inline validation (without a decorator)

```python
import litellm
from semantix import SemanticType

response = litellm.completion(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Is the Eiffel Tower in Paris?"}]
)

output = response.choices[0].message.content

# Validate inline
result = SemanticType("The response confirms the Eiffel Tower is in Paris").validate(output)

if result.passed:
    print("Validation passed — entailment score:", result.score)
else:
    print("Validation failed:", result.reason)
```

## Self-Training Collector

semantix-ai ships a `MismatchCollector` that records every failed validation — the raw output, the declared intent, and the NLI score — so you can inspect borderline cases and feed them back as labelled training data.

```python
import litellm
from semantix import validate_intent
from semantix.collector import MismatchCollector

# All validation failures are written to mismatches.jsonl
collector = MismatchCollector(path="mismatches.jsonl")

@validate_intent(
    "The response declines to answer questions outside the scope of customer support",
    collector=collector
)
def support_agent(user_message: str) -> str:
    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a customer support agent. Only answer support questions."
            },
            {"role": "user", "content": user_message}
        ]
    )
    return response.choices[0].message.content

# Off-topic query — mismatch logged if the model answers anyway
support_agent("What is the capital of France?")
```

Each entry in `mismatches.jsonl` contains:

```json
{
  "intent": "The response declines to answer questions outside the scope of customer support",
  "output": "The capital of France is Paris.",
  "score": 0.04,
  "threshold": 0.5,
  "model": "gpt-4o-mini",
  "timestamp": "2025-10-01T12:34:56Z"
}
```

Use the collected data to calibrate thresholds or fine-tune your NLI model.

## Using with LiteLLM Proxy

semantix-ai validates the *content* of responses, so it works equally well when LiteLLM is running as a proxy. Call the proxy endpoint as usual and validate the returned text:

```python
import litellm
from semantix import validate_intent

# Point LiteLLM SDK at your proxy
litellm.api_base = "http://localhost:4000"

@validate_intent("The response lists exactly three action items")
def get_action_items(meeting_notes: str) -> str:
    response = litellm.completion(
        model="gpt-4",   # Must match a model name in your proxy config
        messages=[
            {
                "role": "user",
                "content": f"Extract exactly three action items from: {meeting_notes}"
            }
        ]
    )
    return response.choices[0].message.content
```

## Error Handling

`validate_intent` raises `semantix.SemanticValidationError` when validation fails. Catch it like any other exception:

```python
import litellm
from semantix import validate_intent, SemanticValidationError

@validate_intent("The response is a valid JSON object with a 'status' key")
def call_model(prompt: str) -> str:
    response = litellm.completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

try:
    output = call_model("Return a JSON object with status=ok")
    print(output)
except SemanticValidationError as e:
    print(f"Semantic validation failed: {e.reason} (score={e.score:.2f})")
    # Fall back, retry, or escalate as needed
```

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `threshold` | `0.5` | Minimum NLI entailment score to pass validation |
| `model` | `cross-encoder/nli-MiniLM2-L6-H768` | Local NLI model used for inference |
| `collector` | `None` | `MismatchCollector` instance for logging failures |
| `raise_on_fail` | `True` | Whether to raise `SemanticValidationError` on failure |

```python
from semantix import validate_intent
from semantix.collector import MismatchCollector

@validate_intent(
    "The response politely refuses the request",
    threshold=0.65,          # Stricter threshold
    raise_on_fail=False,     # Return a ValidationResult instead of raising
    collector=MismatchCollector("logs/mismatches.jsonl")
)
def guarded_call(prompt: str) -> str:
    ...
```

## Resources

- [semantix-ai on PyPI](https://pypi.org/project/semantix-ai/)
- [semantix-ai GitHub Repository](https://github.com/labrat-akhona/semantix-ai)
- [LiteLLM Providers](/docs/providers)
- [LiteLLM Proxy Documentation](/docs/simple_proxy)
- [Observability Setup](/docs/integrations/observability_integrations)
