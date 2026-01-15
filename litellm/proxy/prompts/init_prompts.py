"""
Similar to init_guardrails.py, but for prompts.
"""

from typing import Dict, List, Optional

from litellm._logging import verbose_proxy_logger


def init_prompts(
    all_prompts: List[Dict],
    config_file_path: Optional[str] = None,
):
    from litellm.types.prompts.init_prompts import PromptSpec

    from .prompt_registry import IN_MEMORY_PROMPT_REGISTRY

    prompt_list: List[PromptSpec] = []

    for prompt in all_prompts:
        initialized_prompt = IN_MEMORY_PROMPT_REGISTRY.initialize_prompt(
            prompt=PromptSpec(**prompt),
            config_file_path=config_file_path,
        )
        if initialized_prompt:
            prompt_list.append(initialized_prompt)

    verbose_proxy_logger.debug(f"\nPrompt List:{prompt_list}\n")
