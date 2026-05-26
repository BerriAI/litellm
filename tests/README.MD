**In total litellm runs 1000+ tests**

[02/20/2025] Update:

To make it easier to contribute and map what behavior is tested,

we've started mapping the litellm directory in `tests/test_litellm`

This folder can only run mock tests.

## Overriding CI model names

The most-frequently used model identifiers in this test suite are exposed
as env-var-driven constants in [`tests/ci_models.py`](./ci_models.py). When
a provider deprecates a model, every test that imports the constant can be
swapped to the new identifier without a code change:

```bash
# Run with defaults.
pytest tests/llm_translation/

# Override one model name -- every test that imports the constant
# (e.g. GPT_4O) picks up the new value:
CI_MODEL_GPT_4O="gpt-4o-2024-08-06" pytest tests/llm_translation/

# Inspect every constant + its current resolved value:
python -m tests.ci_models
```

To use these constants in a test, import them from ``tests.ci_models``:

```python
from tests.ci_models import GPT_4O, CLAUDE_SONNET_4_5

response = litellm.completion(model=GPT_4O, messages=...)
```

For model names without a dedicated constant, use :func:`get_ci_model`:

```python
from tests.ci_models import get_ci_model

model = get_ci_model("CI_MODEL_MY_MODEL", default="anthropic/claude-haiku")
```

See `tests/ci_models.py` for the full list of constants and the
environment-variable names that override them.
