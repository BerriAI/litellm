"""Which provider serves a model: one answer for dispatch and for the Rust host after callbacks."""

from __future__ import annotations

from typing import Final

from litellm.exceptions import BadRequestError
from litellm.litellm_core_utils.get_llm_provider_logic import declared_authenticating_provider, get_llm_provider


def resolve_provider(
    model: str, custom_llm_provider: str | None, api_base: str | None = None
) -> tuple[str, str] | None:
    authenticating: Final = declared_authenticating_provider(model, custom_llm_provider)
    if authenticating is not None:
        return model.removeprefix(f"{authenticating}/"), authenticating
    try:
        resolved_model, provider, _, _ = get_llm_provider(model, custom_llm_provider, api_base)
    except BadRequestError:
        return None
    return resolved_model, provider
