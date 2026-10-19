from __future__ import annotations

import litellm


def resolved_provider(model: str, custom_llm_provider: str | None) -> tuple[str, str]:
    try:
        resolved_model, provider, _, _ = litellm.get_llm_provider(model=model, custom_llm_provider=custom_llm_provider)
    except Exception:  # noqa: BLE001  # an unroutable model still shapes as a bare Anthropic id
        return model, custom_llm_provider or "anthropic"
    return resolved_model, provider
