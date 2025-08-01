from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .dotprompt_manager import DotpromptManager
    from .prompt_manager import PromptManager, PromptTemplate

# Global instances
default_prompt_directory: Optional[str] = None
default_prompt_manager: Optional["PromptManager"] = None
global_dotprompt_manager: Optional["DotpromptManager"] = None


def get_dotprompt_manager(prompt_directory: Optional[str] = None) -> "DotpromptManager":
    """
    Get or create the global dotprompt manager instance.

    Args:
        prompt_directory: Directory containing .prompt files. If None, uses global default.

    Returns:
        DotpromptManager instance
    """
    global global_dotprompt_manager, default_prompt_directory

    # Use provided directory or fall back to global default
    directory = prompt_directory or default_prompt_directory

    if (
        global_dotprompt_manager is None
        or global_dotprompt_manager.prompt_directory != directory
    ):
        from .dotprompt_manager import DotpromptManager

        global_dotprompt_manager = DotpromptManager(directory)

    return global_dotprompt_manager


def set_global_prompt_directory(directory: str) -> None:
    """
    Set the global prompt directory for dotprompt files.

    Args:
        directory: Path to directory containing .prompt files
    """
    global default_prompt_directory, global_dotprompt_manager

    default_prompt_directory = directory

    # Reset the global manager to force reload with new directory
    if global_dotprompt_manager:
        global_dotprompt_manager.set_prompt_directory(directory)


# Export public API
__all__ = [
    "PromptManager",
    "DotpromptManager",
    "PromptTemplate",
    "get_dotprompt_manager",
    "set_global_prompt_directory",
    "default_prompt_directory",
    "default_prompt_manager",
    "global_dotprompt_manager",
]
