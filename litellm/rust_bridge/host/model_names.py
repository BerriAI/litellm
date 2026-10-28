"""The candidate names ``get_model_info`` tries for a model, for the Rust resolver"""

from typing import Final

from litellm.litellm_core_utils.get_llm_provider_logic import provider_banner_suppressed


def candidates(
    model: str, custom_llm_provider: str | None
) -> tuple[str | None, str | None, str | None, str | None, str | None, str | None, str | None]:
    """Main's ``get_potential_model_names`` with the provider banner suppressed. The names come back in the order
    ``combined_model_name``, ``region_free_combined_model_name``, ``split_model``, ``combined_stripped_model_name``,
    ``stripped_model_name``, ``provider_prefixed_model_name``, then the provider rows must match"""
    from litellm.litellm_core_utils.get_llm_provider_logic import declared_authenticating_provider
    from litellm.utils import get_potential_model_names

    with provider_banner_suppressed():
        names: Final = get_potential_model_names(
            model=model, custom_llm_provider=custom_llm_provider or declared_authenticating_provider(model)
        )
    return (
        names["combined_model_name"],
        names["region_free_combined_model_name"],
        names["split_model"],
        names["combined_stripped_model_name"],
        names["stripped_model_name"],
        names["provider_prefixed_model_name"],
        names["custom_llm_provider"],
    )
