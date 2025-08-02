from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .prompt_manager import PromptManager, PromptTemplate
    from litellm.types.prompts.init_prompts import PromptLiteLLMParams, PromptSpec
    from litellm.integrations.custom_prompt_management import CustomPromptManagement

from litellm.types.prompts.init_prompts import SupportedPromptIntegrations

from .dotprompt_manager import DotpromptManager

# Global instances
global_prompt_directory: Optional[str] = None
global_prompt_manager: Optional["PromptManager"] = None


def set_global_prompt_directory(directory: str) -> None:
    """
    Set the global prompt directory for dotprompt files.

    Args:
        directory: Path to directory containing .prompt files
    """
    import litellm

    litellm.global_prompt_directory = directory  # type: ignore


def prompt_initializer(
    litellm_params: "PromptLiteLLMParams", prompt_spec: "PromptSpec"
) -> "CustomPromptManagement":
    """
    Initialize a prompt from a .prompt file.
    """
    prompt_directory = getattr(litellm_params, "prompt_directory", None)
    if not prompt_directory:
        raise ValueError("prompt_directory is required for dotprompt")

    return DotpromptManager(prompt_directory)


prompt_initializer_registry = {
    SupportedPromptIntegrations.DOT_PROMPT.value: prompt_initializer,
}

# Export public API
__all__ = [
    "PromptManager",
    "DotpromptManager",
    "PromptTemplate",
    "set_global_prompt_directory",
    "global_prompt_directory",
    "global_prompt_manager",
]
