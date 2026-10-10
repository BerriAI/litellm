# tests/integration/translation

Exact translation cases on the shared fake provider. `README.md` here has the case fields, deployments and capture steps

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
