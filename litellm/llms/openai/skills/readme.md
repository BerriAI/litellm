# OpenAI Skills API

LiteLLM exposes the native OpenAI Skills API for `openai` and `azure` providers. The native response schema is preserved; the existing Anthropic Skills schema is unchanged.

```python
import litellm

skill = litellm.create_skill(
    files=[("SKILL.md", b"# Example\n")],
    custom_llm_provider="openai",
)

versions = litellm.list_skill_versions(
    skill_id=skill.id,
    custom_llm_provider="openai",
)
```

Supported operations are create/list/get/delete, default-version update, version create/list/get/delete, and bundle content retrieval. Azure uses the existing LiteLLM Azure `api_base`, `api_version`, API-key, and Entra ID credential resolution.

When a model alias is routed through `model_list`, the returned native skill ID is LiteLLM-wrapped so later Skills operations can use the same deployment. Native Responses `skill_reference` IDs are unwrapped before forwarding upstream.
