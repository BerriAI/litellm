"""
Skill Applicator for Gateway Skills.

Handles provider-specific strategies for applying skills to LLM requests.
Routes skills to appropriate injection method based on model provider.
"""

from typing import Any, Dict, List, Optional

from litellm._logging import verbose_logger
from litellm.proxy._types import LiteLLM_SkillsTable


class SkillApplicator:
    """
    Applies gateway skills to LLM requests using provider-specific strategies.

    Strategies:
    - system_prompt: Inject skill content into system message (OpenAI, Azure, Bedrock, etc.)
    - tool_conversion: Convert skills to tools + system prompt (existing behavior)
    """

    # Provider to strategy mapping
    PROVIDER_STRATEGIES: Dict[str, str] = {
        # System prompt injection
        "openai": "system_prompt",
        "azure": "system_prompt",
        "azure_ai": "system_prompt",
        "bedrock": "system_prompt",
        "vertex_ai": "system_prompt",
        "vertex_ai_beta": "system_prompt",
        "gemini": "system_prompt",
        "ollama": "system_prompt",
        "ollama_chat": "system_prompt",
        "groq": "system_prompt",
        "together_ai": "system_prompt",
        "deepseek": "system_prompt",
        "fireworks_ai": "system_prompt",
        "mistral": "system_prompt",
        "cohere": "system_prompt",
        "cohere_chat": "system_prompt",
        "ai21": "system_prompt",
        "replicate": "system_prompt",
        "sagemaker": "system_prompt",
        "perplexity": "system_prompt",
        "anyscale": "system_prompt",
        "openrouter": "system_prompt",
        "huggingface": "system_prompt",
        "text-completion-openai": "system_prompt",
        "text-completion-codestral": "system_prompt",
        # Tool conversion (existing Anthropic behavior)
        "anthropic": "tool_conversion",
    }

    def __init__(self):
        from litellm.llms.litellm_proxy.skills.prompt_injection import (
            SkillPromptInjectionHandler,
        )

        self.prompt_handler = SkillPromptInjectionHandler()

    def get_strategy(self, provider: str) -> str:
        """
        Get the skill application strategy for a provider.

        Args:
            provider: The LLM provider name

        Returns:
            Strategy name ("system_prompt" or "tool_conversion")
        """
        return self.PROVIDER_STRATEGIES.get(provider, "system_prompt")

    async def apply_skills(
        self,
        data: dict,
        skills: List[LiteLLM_SkillsTable],
        provider: str,
    ) -> dict:
        """
        Apply skills to a request based on provider strategy.

        Args:
            data: The request data dict
            skills: List of skills to apply
            provider: The LLM provider name

        Returns:
            Modified request data with skills applied
        """
        if not skills:
            return data

        strategy = self.get_strategy(provider)

        verbose_logger.debug(
            f"SkillApplicator: Applying {len(skills)} skills with strategy={strategy} "
            f"for provider={provider}"
        )

        if strategy == "system_prompt":
            return self._apply_system_prompt_strategy(data, skills)
        elif strategy == "tool_conversion":
            return self._apply_tool_conversion_strategy(data, skills)
        else:
            verbose_logger.warning(
                f"SkillApplicator: Unknown strategy {strategy}, using system_prompt"
            )
            return self._apply_system_prompt_strategy(data, skills)

    def _apply_system_prompt_strategy(
        self,
        data: dict,
        skills: List[LiteLLM_SkillsTable],
    ) -> dict:
        """
        Apply skills by injecting content into system prompt.

        This strategy appends skill content to the system message:
        ---
        ## Skill: {display_title}
        **Description:** {description}

        ### Instructions
        {SKILL.md body content}
        ---

        Args:
            data: The request data dict
            skills: List of skills to apply

        Returns:
            Modified request data with skill content in system message
        """
        skill_contents: List[str] = []

        for skill in skills:
            content = self._format_skill_content(skill)
            if content:
                skill_contents.append(content)

        if not skill_contents:
            return data

        # Inject into system message
        return self.prompt_handler.inject_skill_content_to_messages(
            data, skill_contents, use_anthropic_format=False
        )

    def _apply_tool_conversion_strategy(
        self,
        data: dict,
        skills: List[LiteLLM_SkillsTable],
    ) -> dict:
        """
        Apply skills by converting to tools (existing Anthropic behavior).

        This uses the existing SkillPromptInjectionHandler logic for
        tool conversion and skill content injection.

        Args:
            data: The request data dict
            skills: List of skills to apply

        Returns:
            Modified request data with tools and skill content
        """
        tools = data.get("tools", [])
        skill_contents: List[str] = []

        for skill in skills:
            # Convert skill to Anthropic-style tool
            tools.append(self.prompt_handler.convert_skill_to_anthropic_tool(skill))

            # Extract skill content
            content = self.prompt_handler.extract_skill_content(skill)
            if content:
                skill_contents.append(content)

        if tools:
            data["tools"] = tools

        if skill_contents:
            # For Anthropic, use top-level 'system' param
            data = self.prompt_handler.inject_skill_content_to_messages(
                data, skill_contents, use_anthropic_format=True
            )

        return data

    def _format_skill_content(self, skill: LiteLLM_SkillsTable) -> Optional[str]:
        """
        Format skill content for system prompt injection.

        Produces:
        ---
        ## Skill: {display_title}
        **Description:** {description}

        ### Instructions
        {extracted content or instructions}
        ---

        Args:
            skill: The skill to format

        Returns:
            Formatted skill content string, or None if no content
        """
        # Try to extract content from file first
        content = self.prompt_handler.extract_skill_content(skill)

        # Fall back to instructions
        if not content:
            content = skill.instructions

        if not content:
            return None

        # Build formatted skill section
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

    Uses LiteLLM's get_llm_provider function to resolve the provider.

    Args:
        model: The model identifier

    Returns:
        The provider name (e.g., "openai", "anthropic", "bedrock")
    """
    try:
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        _, custom_llm_provider, _, _ = get_llm_provider(model=model)
        return custom_llm_provider or "openai"
    except Exception as e:
        verbose_logger.warning(
            f"SkillApplicator: Failed to determine provider for model {model}: {e}"
        )
        # Default to OpenAI-compatible
        return "openai"
