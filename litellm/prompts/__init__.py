from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .prompt_manager import PromptManager


default_prompt_directory: Optional[str] = None
default_prompt_manager: Optional[PromptManager] = None


def load_default_prompt_manager(default_prompt_directory: str) -> PromptManager:
    """
    Loads the default prompt directory from the environment variable LITTELM_PROMPT_DIRECTORY.
    """
    from .prompt_manager import PromptManager

    _prompt_manager = PromptManager(default_prompt_directory)

    return _prompt_manager
