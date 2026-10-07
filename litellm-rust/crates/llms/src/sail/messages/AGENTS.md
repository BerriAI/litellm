# structure

- Sail's native Messages projection delegates its compatible wire behavior and registry settings to OpenAI-like

# boundaries

- Spend settlement and selection of this adapter belong to call orchestration

# invariants

- Messages ignores service tiers and never derives a completion window from them

# references

- `litellm/llms/openai_like/messages/transformation.py`
- `litellm/llms/openai_like/providers.json`
- `tests/unit/llms/sail/messages/test_sail_messages_transformation.py`
