# tests/integration/translation

Exact translation cases on the shared fake provider. `README.md` here has the case fields, deployments and capture steps

## Templates

Reuse the client request and expected response in `<endpoint>/templates.py`, and the upstream endpoint's
request and response in `templates/<provider>.py`. Model cases bind those templates with dictionary
unpacking and explicit model-specific overrides. Templates omit `model`. Keep exact captured values,
replace nested values rather than mutating them, and never call production translators to build an
expectation. Native passthrough fields may share literal data. The Sonnet 5.5 pilot demonstrates this
across Chat Completions, Messages and Responses, with cache-hit response variants

## Naming

Each model gets one complete base `TranslationTestCase` in `translation/<endpoint>/bases/<provider>.py`,
named `<MODEL>_TEST_CASE` after its deployment (`anthropic/claude-sonnet-4-6` is
`CLAUDE_SONNET_4_6_TEST_CASE`). A feature case is `<MODEL>_<SCENARIO>_TEST_CASE`. Import a base under its
own name, never aliased to `BASE`, so every case shows which model it derives from

```python
from integration.translation.messages.bases.anthropic import CLAUDE_SONNET_4_6_TEST_CASE

CLAUDE_SONNET_4_6_THINKING_BUDGET_TEST_CASE: Final = replace(
    CLAUDE_SONNET_4_6_TEST_CASE,
    scenario="thinking_budget",
    litellm_request={**CLAUDE_SONNET_4_6_TEST_CASE.litellm_request, "max_tokens": 2048, "thinking": ...},
    ...
)
```
