"""
Skills Injection Hook for LiteLLM Proxy

This hook intercepts requests and processes skills from the container.skills parameter:
- For skills with 'litellm:' prefix: Fetches from LiteLLM DB and converts to tools for non-Anthropic
- For skills without prefix: Passes through to Anthropic API as native skills

Usage:
    In messages API request:
    {
        "model": "gpt-4",
        "container": {
            "skills": [
                {"type": "custom", "skill_id": "litellm:skill_abc123"}
            ]
        },
        "messages": [...]
    }
"""

from typing import Any, Dict, List, Optional, Union

from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import LiteLLM_SkillsTable, UserAPIKeyAuth
from litellm.types.utils import CallTypesLiteral


class SkillsInjectionHook(CustomLogger):
    """
    Pre-call hook that processes skills from container.skills parameter.

    - Skills with 'litellm:' prefix are fetched from LiteLLM DB
    - For Anthropic models: native skills pass through, LiteLLM skills need to be on Anthropic
    - For non-Anthropic models: LiteLLM skills are converted to tools array
    """

    def __init__(self, **kwargs):
        self.optional_params = kwargs
        super().__init__(**kwargs)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallTypesLiteral,
    ) -> Optional[Union[Exception, str, dict]]:
        """
        Process skills from container.skills before the LLM call.

        1. Check if container.skills exists in request
        2. Separate skills by prefix (litellm: vs native)
        3. Fetch LiteLLM skills from database
        4. For Anthropic: keep native skills in container
        5. For non-Anthropic: convert LiteLLM skills to tools, remove container
        """
        # Only process completion-type calls
        if call_type not in ["completion", "acompletion", "anthropic_messages"]:
            return data

        container = data.get("container")
        if not container or not isinstance(container, dict):
            return data

        skills = container.get("skills")
        if not skills or not isinstance(skills, list):
            return data

        verbose_proxy_logger.debug(f"SkillsInjectionHook: Processing {len(skills)} skills")

        litellm_skills: List[LiteLLM_SkillsTable] = []
        anthropic_skills: List[Dict[str, Any]] = []

        # Separate skills by prefix
        for skill in skills:
            if not isinstance(skill, dict):
                continue

            skill_id = skill.get("skill_id", "")
            if skill_id.startswith("litellm:"):
                # Fetch from LiteLLM DB
                actual_id = skill_id.replace("litellm:", "", 1)
                db_skill = await self._fetch_skill_from_db(actual_id)
                if db_skill:
                    litellm_skills.append(db_skill)
                else:
                    verbose_proxy_logger.warning(
                        f"SkillsInjectionHook: Skill '{actual_id}' not found in LiteLLM DB"
                    )
            else:
                # Native Anthropic skill - pass through
                anthropic_skills.append(skill)

        # Determine if this is an Anthropic model
        model = data.get("model", "")
        is_anthropic = self._is_anthropic_model(model)

        if is_anthropic:
            # For Anthropic models: keep native skills in container
            # LiteLLM skills would need to be created on Anthropic first
            if anthropic_skills:
                data["container"]["skills"] = anthropic_skills
            else:
                # No native skills, remove container if only litellm skills were requested
                if litellm_skills and not anthropic_skills:
                    # Convert LiteLLM skills to tools for Anthropic too
                    tools = data.get("tools", [])
                    for skill in litellm_skills:
                        tools.append(self._convert_skill_to_tool(skill))
                    data["tools"] = tools
                    data.pop("container", None)

            verbose_proxy_logger.debug(
                f"SkillsInjectionHook: Anthropic model - {len(anthropic_skills)} native skills, "
                f"{len(litellm_skills)} LiteLLM skills converted to tools"
            )
        else:
            # For non-Anthropic models: convert LiteLLM skills to tools array
            tools = data.get("tools", [])
            for skill in litellm_skills:
                tools.append(self._convert_skill_to_tool(skill))
            if tools:
                data["tools"] = tools

            # Remove container for non-Anthropic (they don't support it)
            data.pop("container", None)

            verbose_proxy_logger.debug(
                f"SkillsInjectionHook: Non-Anthropic model - converted {len(litellm_skills)} skills to tools"
            )

        return data

    async def _fetch_skill_from_db(self, skill_id: str) -> Optional[LiteLLM_SkillsTable]:
        """
        Fetch a skill from the LiteLLM database.

        Args:
            skill_id: The skill ID (without 'litellm:' prefix)

        Returns:
            LiteLLM_SkillsTable or None if not found
        """
        try:
            from litellm.proxy.management_endpoints.skills_management_endpoints import (
                LiteLLMSkillsHandler,
            )

            return await LiteLLMSkillsHandler.fetch_skill_from_db(skill_id)
        except Exception as e:
            verbose_proxy_logger.warning(
                f"SkillsInjectionHook: Error fetching skill {skill_id}: {e}"
            )
            return None

    def _is_anthropic_model(self, model: str) -> bool:
        """
        Check if the model is an Anthropic model using get_llm_provider.

        Args:
            model: The model name/identifier

        Returns:
            True if Anthropic model, False otherwise
        """
        try:
            from litellm.litellm_core_utils.get_llm_provider_logic import (
                get_llm_provider,
            )

            _, custom_llm_provider, _, _ = get_llm_provider(model=model)
            return custom_llm_provider == "anthropic"
        except Exception:
            # Fallback to simple check if get_llm_provider fails
            return "claude" in model.lower() or model.lower().startswith("anthropic/")

    def _convert_skill_to_tool(self, skill: LiteLLM_SkillsTable) -> Dict[str, Any]:
        """
        Convert a LiteLLM skill to an OpenAI-style tool.

        The skill's instructions are used as the function description,
        allowing the model to understand when and how to use the skill.

        Args:
            skill: The skill from LiteLLM database

        Returns:
            OpenAI-style tool definition
        """
        # Create a function name from skill_id (sanitize for function naming)
        func_name = skill.skill_id.replace("-", "_").replace(" ", "_")

        # Use instructions as description, fall back to description or title
        description = (
            skill.instructions
            or skill.description
            or skill.display_title
            or f"Skill: {skill.skill_id}"
        )

        # Truncate description if too long (OpenAI has limits)
        max_desc_length = 1024
        if len(description) > max_desc_length:
            description = description[: max_desc_length - 3] + "..."

        tool: Dict[str, Any] = {
            "type": "function",
            "function": {
                "name": func_name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }

        # If skill has metadata with parameter definitions, use them
        if skill.metadata and isinstance(skill.metadata, dict):
            params = skill.metadata.get("parameters")
            if params and isinstance(params, dict):
                tool["function"]["parameters"] = params

        return tool


# Global instance for registration
skills_injection_hook = SkillsInjectionHook()

