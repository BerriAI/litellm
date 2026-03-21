"""
Skill Applicator for Gateway Skills.

Handles provider-specific strategies for applying skills to LLM requests.
Uses get_llm_provider() to resolve models to providers, then checks the
centralized beta headers config to determine if the model's provider
supports native skills (skills-2025-10-02 beta). If not, falls back to
system prompt injection.
"""

from typing import List, Optional

from litellm._logging import verbose_logger
from litellm.constants import ANTHROPIC_SKILLS_API_BETA_VERSION
from litellm.proxy._types import LiteLLM_SkillsTable


class SkillApplicator:
    """
    Applies gateway skills to LLM requests using provider-specific strategies.

    Provider resolution is delegated to litellm.get_llm_provider().
    Native skills support is determined by the centralized beta headers
    config (anthropic_beta_headers_config.json) — if the provider maps
    skills-2025-10-02 to a non-null value, native skills are supported.
    """

    def __init__(self):
        from litellm.llms.litellm_proxy.skills.prompt_injection import (
            SkillPromptInjectionHandler,
        )

        self.prompt_handler = SkillPromptInjectionHandler()

    def supports_native_skills(self, provider: str) -> bool:
        """
        Check if a provider supports native skills by consulting the
        centralized beta headers config.
        """
        from litellm.anthropic_beta_headers_manager import is_beta_header_supported

        return is_beta_header_supported(
            beta_header=ANTHROPIC_SKILLS_API_BETA_VERSION,
            provider=provider,
        )

    async def apply_skills(
        self,
        data: dict,
        skills: List[LiteLLM_SkillsTable],
        provider: str,
    ) -> dict:
        """
        Apply skills to a request based on provider.

        Args:
            data: The request data dict
            skills: List of skills to apply
            provider: The LLM provider name (from get_llm_provider)

        Returns:
            Modified request data with skills applied
        """
        if not skills:
            return data

        if self.supports_native_skills(provider):
            verbose_logger.debug(
                f"SkillApplicator: Applying {len(skills)} skills via native API "
                f"for provider={provider}"
            )
            return self._apply_tool_conversion_strategy(data, skills)

        verbose_logger.debug(
            f"SkillApplicator: Applying {len(skills)} skills via system prompt "
            f"for provider={provider}"
        )
        return self._apply_system_prompt_strategy(data, skills)

    def _apply_system_prompt_strategy(
        self,
        data: dict,
        skills: List[LiteLLM_SkillsTable],
    ) -> dict:
        """
        Apply skills by injecting content into system prompt.

        Format:
        ---
        ## Skill: {display_title}
        **Description:** {description}

        ### Instructions
        {SKILL.md body content}
        ---
        """
        skill_contents: List[str] = []

        for skill in skills:
            content = self._format_skill_content(skill)
            if content:
                skill_contents.append(content)

        if not skill_contents:
            return data

        return self.prompt_handler.inject_skill_content_to_messages(
            data, skill_contents, use_anthropic_format=False
        )

    def _apply_tool_conversion_strategy(
        self,
        data: dict,
        skills: List[LiteLLM_SkillsTable],
    ) -> dict:
        """
        Apply skills by converting to Anthropic-style tools + system prompt.
        """
        tools = data.get("tools", [])
        skill_contents: List[str] = []

        for skill in skills:
            tools.append(self.prompt_handler.convert_skill_to_anthropic_tool(skill))

            content = self.prompt_handler.extract_skill_content(skill)
            if content:
                skill_contents.append(content)

        if tools:
            data["tools"] = tools

        if skill_contents:
            data = self.prompt_handler.inject_skill_content_to_messages(
                data, skill_contents, use_anthropic_format=True
            )

        return data

    def _format_skill_content(self, skill: LiteLLM_SkillsTable) -> Optional[str]:
        """
        Format skill content for system prompt injection.
        """
        content = self.prompt_handler.extract_skill_content(skill)

        if not content:
            content = skill.instructions

        if not content:
            return None

        title = skill.display_title or skill.skill_id
        parts = [f"## Skill: {title}"]

        if skill.description:
            parts.append(f"**Description:** {skill.description}")

        parts.append("")
        parts.append("### Instructions")
        parts.append(content)

        return "\n".join(parts)


def get_provider_from_model(model: str) -> str:
    """
    Determine the provider from a model string.

    Uses LiteLLM's get_llm_provider to resolve the provider.
    """
    try:
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        _, custom_llm_provider, _, _ = get_llm_provider(model=model)
        return custom_llm_provider or "openai"
    except Exception as e:
        verbose_logger.warning(
            f"SkillApplicator: Failed to determine provider for model {model}: {e}"
        )
        return "openai"
