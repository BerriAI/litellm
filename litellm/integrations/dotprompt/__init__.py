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
    prompt_data = getattr(litellm_params, "prompt_data", None)
    prompt_id = getattr(litellm_params, "prompt_id", None)
    if prompt_directory:
        raise ValueError(
            "Cannot set prompt_directory when working with prompt_initializer. Needs to be a specific dotprompt file"
        )

    prompt_file = getattr(litellm_params, "prompt_file", None)

    try:
        dot_prompt_manager = DotpromptManager(
            prompt_directory=prompt_directory,
            prompt_data=prompt_data,
            prompt_file=prompt_file,
            prompt_id=prompt_id,
        )

        return dot_prompt_manager
    except Exception as e:

        raise e


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
