from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from .prompt_manager import PromptManager, PromptTemplate
    from litellm.types.prompts.init_prompts import PromptLiteLLMParams, PromptSpec
    from litellm.integrations.custom_prompt_management import CustomPromptManagement

from litellm.types.prompts.init_prompts import SupportedPromptIntegrations

from .dotprompt_manager import DotpromptManager

# Global instances
global_prompt_directory: Optional[str] = None
global_prompt_manager: Optional["PromptManager"] = None


def _convert_dotprompt_content_to_json(
    dotprompt_content: str, prompt_id: str
) -> Dict[str, Dict[str, Any]]:
    """
    Helper function to convert dotprompt file content to JSON format.
    
    Args:
        dotprompt_content: Raw dotprompt file content (YAML frontmatter + template)
        prompt_id: The prompt ID to use as the key
        
    Returns:
        Dictionary in format: {prompt_id: {content: str, metadata: dict}}
    """
    from .prompt_manager import PromptManager

    # Use PromptManager's static parser to convert dotprompt string to JSON
    metadata, content = PromptManager._parse_frontmatter(dotprompt_content)
    
    # Convert to prompt_data format expected by PromptManager
    return {
        prompt_id: {
            "content": content.strip(),
            "metadata": metadata
        }
    }


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
    dotprompt_content = getattr(litellm_params, "dotprompt_content", None)
    prompt_id = getattr(litellm_params, "prompt_id", None)
    if prompt_directory:
        raise ValueError(
            "Cannot set prompt_directory when working with prompt_initializer. Needs to be a specific dotprompt file"
        )

    prompt_file = getattr(litellm_params, "prompt_file", None)

    # If dotprompt_content is provided and prompt_data is empty or None, convert it
    if dotprompt_content:
        if not prompt_id:
            raise ValueError("prompt_id is required when dotprompt_content is provided")
        prompt_data = _convert_dotprompt_content_to_json(
            dotprompt_content=dotprompt_content, 
            prompt_id=prompt_id
        )

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
