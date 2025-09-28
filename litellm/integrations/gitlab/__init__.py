from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .gitlab_prompt_manager import GitLabPromptManager
    from litellm.types.prompts.init_prompts import PromptLiteLLMParams, PromptSpec
    from litellm.integrations.custom_prompt_management import CustomPromptManagement

from litellm.types.prompts.init_prompts import SupportedPromptIntegrations

from .gitlab_prompt_manager import GitLabPromptManager

# Global instances
global_gitlab_config: Optional[dict] = None


def set_global_gitlab_config(config: dict) -> None:
    """
    Set the global BitBucket configuration for prompt management.

    Args:
        config: Dictionary containing BitBucket configuration
                - workspace: BitBucket workspace name
                - repository: Repository name
                - access_token: BitBucket access token
                - branch: Branch to fetch prompts from (default: main)
    """
    import litellm

    litellm.global_gitlab_config = config  # type: ignore


def prompt_initializer(
    litellm_params: "PromptLiteLLMParams", prompt_spec: "PromptSpec"
) -> "CustomPromptManagement":
    """
    Initialize a prompt from a BitBucket repository.
    """
    gitlab_config = getattr(litellm_params, "gitlab_config", None)
    prompt_id = getattr(litellm_params, "prompt_id", None)


    if not gitlab_config:
        raise ValueError(
            "bitbucket_config is required for BitBucket prompt integration"
        )

    try:
        bitbucket_prompt_manager = GitLabPromptManager(
            gitlab_config=gitlab_config,
            prompt_id=prompt_id,
        )

        return bitbucket_prompt_manager
    except Exception as e:
        raise e


prompt_initializer_registry = {
    SupportedPromptIntegrations.BITBUCKET.value: prompt_initializer,
}

# Export public API
__all__ = [
    "GitLabPromptManager",
    "set_global_gitlab_config",
    "global_gitlab_config",
]
