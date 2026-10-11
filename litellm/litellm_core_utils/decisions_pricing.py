from collections.abc import Mapping
from typing import Final


def decisions_pricing_key(custom_llm_provider: str, model: str) -> str:
    return f"{custom_llm_provider}/decisions/{model}"


def is_decisions_pricing_key(key: str, custom_llm_provider: str) -> bool:
    return key.startswith(decisions_pricing_key(custom_llm_provider, ""))


def decisions_pricing_model(
    custom_llm_provider: str,
    model: str,
    model_cost: Mapping[str, object],
) -> str:
    decisions_model: Final = decisions_pricing_key(custom_llm_provider, model)
    return decisions_model if decisions_model in model_cost else f"{custom_llm_provider}/{model}"
