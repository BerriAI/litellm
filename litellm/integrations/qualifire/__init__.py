import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from litellm.integrations.custom_prompt_management import CustomPromptManagement
    from litellm.types.prompts.init_prompts import PromptLiteLLMParams, PromptSpec

from litellm.types.prompts.init_prompts import SupportedPromptIntegrations

from .qualifire_prompt_manager import QualifirePromptManager


def prompt_initializer(
    litellm_params: "PromptLiteLLMParams", prompt_spec: "PromptSpec"
) -> "CustomPromptManagement":
    """
    Initialize a prompt from Qualifire.
    """
    api_key = getattr(litellm_params, "api_key", None) or os.environ.get(
        "QUALIFIRE_API_KEY"
    )
    api_base = getattr(litellm_params, "api_base", None) or "https://api.qualifire.ai"

    if not api_key:
        raise ValueError(
            "api_key is required for Qualifire prompt integration. "
            "Set it in litellm_params or via QUALIFIRE_API_KEY environment variable."
        )

    try:
        qualifire_prompt_manager = QualifirePromptManager(
            api_key=api_key,
            api_base=api_base,
        )
        return qualifire_prompt_manager
    except Exception as e:
        raise e


prompt_initializer_registry = {
    SupportedPromptIntegrations.QUALIFIRE.value: prompt_initializer,
}
