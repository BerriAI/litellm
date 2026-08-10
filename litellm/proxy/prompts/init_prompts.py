"""
Similar to init_guardrails.py, but for prompts.
"""

from typing import Final

from litellm._logging import verbose_proxy_logger


def init_prompts(
    all_prompts: list[dict],
    config_file_path: str | None = None,
):
    from litellm.types.prompts.init_prompts import PromptSpec

    from .prompt_registry import IN_MEMORY_PROMPT_REGISTRY

    prompt_list: Final[list[PromptSpec]] = []

    for prompt in all_prompts:
        initialized_prompt = IN_MEMORY_PROMPT_REGISTRY.initialize_prompt(
            prompt=PromptSpec(**prompt),
            config_file_path=config_file_path,
        )
        if initialized_prompt:
            prompt_list.append(initialized_prompt)

    verbose_proxy_logger.debug("\nPrompt List:%s\n", prompt_list)
