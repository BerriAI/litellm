# semantix-ai Integration

[semantix-ai](https://github.com/labrat-akhona/semantix-ai) is a semantic type system for AI outputs. It validates that LLM responses match natural-language intents using local NLI (Natural Language Inference) models — no external API calls, no added latency beyond ~15 ms, and zero per-validation cost.

## Prerequisites

```bash
pip install semantix-ai litellm
```

## Quick Start

Define an intent as a class with a docstring, then use `@validate_intent` with a return-type annotation to enforce it at runtime.

```python
import litellm
from semantix import Intent, validate_intent, SemanticIntentError

class BookingConfirmation(Intent):
    """The text must confirm that a flight booking was successful."""

@validate_intent
def book_flight(origin: str, destination: str) -> BookingConfirmation:
    response = litellm.completion(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": f"Confirm a flight booking from {origin} to {destination}."}
        ],
    )
    return response.choices[0].message.content

# Raises SemanticIntentError if the output does not match the intent
result = book_flight("JFK", "LAX")
print(result)
```

## Multi-Provider Example

Because validation is driven by the return-type annotation, you can swap providers without changing any validation logic:

```python
import litellm
from semantix import Intent, validate_intent

class ConciseSummary(Intent):
    """The text must be a concise summary of the provided article."""

@validate_intent
def summarise(article: str, model: str = "gpt-4o") -> ConciseSummary:
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": f"Summarise: {article}"}],
    )
    return response.choices[0].message.content

# Works identically across providers
summarise(article_text, model="gpt-4o")                                # OpenAI
summarise(article_text, model="anthropic/claude-3-5-sonnet-20241022")  # Anthropic
summarise(article_text, model="azure/gpt-4o")                         # Azure OpenAI
summarise(article_text, model="gemini/gemini-1.5-pro")                # Google
summarise(article_text, model="groq/llama3-70b-8192")                 # Groq
```

## Self-Healing Retries

Pass `retries` and an explicit judge to `@validate_intent`. When a `semantix_feedback` parameter is present, semantix injects the judge's feedback so the next attempt can self-correct.

```python
from typing import Optional
import litellm
from semantix import Intent, validate_intent, NLIJudge

class PoliteDecline(Intent):
    """The text must politely decline an invitation without being rude."""

@validate_intent(judge=NLIJudge(), retries=2)
def decline(event: str, semantix_feedback: Optional[str] = None) -> PoliteDecline:
    prompt = f"Decline the invitation to: {event}"
    if semantix_feedback:
        prompt += f"\n\nFeedback from previous attempt:\n{semantix_feedback}"
    response = litellm.completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

result = decline("company picnic")
```

## Testing with `assert_semantic`

Use `assert_semantic` in your test suite to make semantic assertions on any text.

```python
from semantix.testing import assert_semantic

output = "We appreciate the invite but unfortunately cannot attend."

# Passes if the text semantically matches the stated intent
assert_semantic(output, "must be polite and professional")
```

## Training Collector

`TrainingCollector` records validation outcomes so you can feed them back as labelled training data or calibrate thresholds.

```python
import litellm
from semantix import Intent, validate_intent
from semantix.training import TrainingCollector

collector = TrainingCollector(path="training_data.jsonl")

class ScopedResponse(Intent):
    """The response must decline to answer questions outside customer support scope."""

@validate_intent(retries=2, collector=collector)
def support_agent(user_message: str) -> ScopedResponse:
    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a customer support agent. Only answer support questions."},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content

support_agent("What is the capital of France?")
```

## Composite Intents

Combine intents with `&` (all must match) and `|` (any must match):

```python
from semantix import Intent

class Polite(Intent):
    """The text must be polite."""

class Concise(Intent):
    """The text must be concise."""

class Casual(Intent):
    """The text must have a casual tone."""

PoliteAndConcise = Polite & Concise  # AllOf — both must match
PoliteOrCasual   = Polite | Casual   # AnyOf — at least one must match
```

## Resources

- [semantix-ai on PyPI](https://pypi.org/project/semantix-ai/)
- [semantix-ai GitHub Repository](https://github.com/labrat-akhona/semantix-ai)
- [LiteLLM Providers](/docs/providers)
- [LiteLLM Proxy Documentation](/docs/simple_proxy)
