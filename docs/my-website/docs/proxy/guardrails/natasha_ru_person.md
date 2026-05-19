# Natasha (Russian person names)

## Why this guardrail exists

Gateways often need to **redact personally identifiable information** before text is sent to an external LLM. For **Russian full names (ФИО)**, regular expressions are a poor tool: capitalised Cyrillic tokens can denote cities, streets, organisations, or surnames, and names inflect (`Иванов` → `Иванова`, `Иванову`). A small **on-device named-entity recogniser** tags spans as person (`PER`) with much higher recall than pattern-only filters, without calling a separate “judge” LLM for detection.

## What it does

- **When:** `pre_call` only (supported event hook is `pre_call`).
- **Where:** `user` and `system` message `content` (string or list of `{ "type": "text", "text": "..." }` chunks).
- **How:** [Natasha](https://github.com/natasha/natasha) `NewsNERTagger` marks `PER` spans; each span is replaced with a placeholder (default `<PER_REDACTED>`).
- **Important:** Detection runs as **local NLP inference** inside the proxy process. **No LLM provider HTTP API** is used for masking (no extra tokens, no data sent to OpenAI/Anthropic/etc. for this step). Models ship inside the `natasha` pip wheel; there is **no runtime download** of weights.

## What it does *not* do

- It is **not** a substitute for secret scanning (API keys, tokens) — combine with `hide-secrets` / content filters as needed.
- It will not perfectly disambiguate **surnames vs toponyms** (inherent NER limitation).

## Installation

The integration is behind an **optional extra** so the default LiteLLM install stays lean:

```bash
pip install "litellm[natasha-ru-person]"
# or with uv, from the litellm repo:
uv sync --extra natasha-ru-person
```

## LiteLLM `config.yaml`

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY

guardrails:
  - guardrail_name: "ru-person-names"
    litellm_params:
      guardrail: natasha_ru_person
      mode: "pre_call"
      default_on: true
```

## Environment variables

| Variable | Description |
|----------|-------------|
| `NATASHA_RU_PERSON_PLACEHOLDER` | Optional; overrides the default redaction string `<PER_REDACTED>`. |

## Performance notes

- **Cold start:** Natasha models load when the guardrail instance is first constructed (typically at proxy startup), not on every request.
- **Per request:** Cost scales with the amount of Cyrillic text processed; ASCII-only payloads are skipped quickly.

## Operational checklist

1. Install the **`natasha-ru-person`** extra on the proxy image / venv.
2. Enable `natasha_ru_person` in `guardrails` as shown above.
3. Restart the proxy after config changes.

## Tests (contributors)

Unit tests live under `tests/test_litellm/proxy/guardrails/test_natasha_ru_person.py`. They exercise **offline** Natasha inference (same process as pytest) — **no chat-completion API calls** for detection. Run:

```bash
uv sync --extra proxy --extra natasha-ru-person --group dev
uv run pytest tests/test_litellm/proxy/guardrails/test_natasha_ru_person.py -v
```

The `proxy` extra is recommended here so imports required by the shared `tests/test_litellm/proxy/conftest.py` fixture chain resolve the same way as in full proxy test jobs.
