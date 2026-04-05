"""Langfuse integration for LiteLLM Prompt Management."""

from .langfuse_prompt_management import LangfusePromptManagement


def initialize_prompt(litellm_params, prompt_spec):
    """
    Initialization function that prompt_registry.py will call.
    """
    return LangfusePromptManagement(
        langfuse_public_key=getattr(litellm_params, "langfuse_public_key", None),
        langfuse_secret=getattr(litellm_params, "langfuse_secret", None),
        langfuse_host=getattr(litellm_params, "langfuse_host", None),
    )


prompt_initializer_registry = {
    "langfuse": initialize_prompt,
}

__all__ = ["initialize_prompt", "prompt_initializer_registry"]
