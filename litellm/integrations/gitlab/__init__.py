from typing import TYPE_CHECKING, Optional, Dict, Any

if TYPE_CHECKING:
    from .gitlab_prompt_manager import GitLabPromptManager
    from litellm.types.prompts.init_prompts import PromptLiteLLMParams, PromptSpec
    from litellm.integrations.custom_prompt_management import CustomPromptManagement

from litellm.types.prompts.init_prompts import SupportedPromptIntegrations
from litellm.integrations.custom_prompt_management import CustomPromptManagement
from litellm.types.prompts.init_prompts import PromptSpec, PromptLiteLLMParams
from .gitlab_prompt_manager import GitLabPromptManager, GitLabPromptCache

# Global instances
global_gitlab_config: Optional[dict] = None


def set_global_gitlab_config(config: dict) -> None:
    """
    Set the global gitlab configuration for prompt management.

    Args:
        config: Dictionary containing gitlab configuration
                - workspace: gitlab workspace name
                - repository: Repository name
                - access_token: gitlab access token
                - branch: Branch to fetch prompts from (default: main)
    """
    import litellm

    litellm.global_gitlab_config = config  # type: ignore


def prompt_initializer(
    litellm_params: "PromptLiteLLMParams", prompt_spec: "PromptSpec"
) -> "CustomPromptManagement":
    """
    Initialize a prompt from a Gitlab repository.
    """
    gitlab_config = getattr(litellm_params, "gitlab_config", None)
    prompt_id = getattr(litellm_params, "prompt_id", None)


    if not gitlab_config:
        raise ValueError(
            "gitlab_config is required for gitlab prompt integration"
        )

    try:
        gitlab_prompt_manager = GitLabPromptManager(
            gitlab_config=gitlab_config,
            prompt_id=prompt_id,
        )

        return gitlab_prompt_manager
    except Exception as e:
        raise e

def _gitlab_prompt_initializer(
        litellm_params: PromptLiteLLMParams,
        prompt: PromptSpec,
) -> CustomPromptManagement:
    """
    Build a GitLab-backed prompt manager for this prompt.
    Expected fields on litellm_params:
      - prompt_integration="gitlab"  (handled by the caller)
      - gitlab_config: Dict[str, Any] (project/access_token/branch/prompts_path/etc.)
      - git_ref (optional): per-prompt tag/branch/SHA override
    """
    # You can store arbitrary integration-specific config on PromptLiteLLMParams.
    # If your dataclass doesn't have these attributes, add them or put inside
    # `litellm_params.extra` and pull them from there.
    gitlab_config: Dict[str, Any] = getattr(litellm_params, "gitlab_config", None) or {}
    git_ref: Optional[str] = getattr(litellm_params, "git_ref", None)

    if not gitlab_config:
        raise ValueError("gitlab_config is required for gitlab prompt integration")

    # prompt.prompt_id can map to a file path under prompts_path (e.g. "chat/greet/hi")
    return GitLabPromptManager(
        gitlab_config=gitlab_config,
        prompt_id=prompt.prompt_id,
        ref=git_ref,
    )


prompt_initializer_registry = {
    SupportedPromptIntegrations.GITLAB.value: _gitlab_prompt_initializer,
}

# Export public API
__all__ = [
    "GitLabPromptManager",
    "GitLabPromptCache",
    "set_global_gitlab_config",
    "global_gitlab_config",
]
