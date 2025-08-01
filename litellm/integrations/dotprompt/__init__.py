from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .prompt_manager import PromptManager, PromptTemplate

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


# Export public API
__all__ = [
    "PromptManager",
    "DotpromptManager",
    "PromptTemplate",
    "set_global_prompt_directory",
    "global_prompt_directory",
    "global_prompt_manager",
]
