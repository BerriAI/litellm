"""
AI Policy Suggester - uses LLM tool calling to suggest policy templates
based on user-provided attack examples and descriptions.
"""

import json
from typing import List, Optional

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import DEFAULT_COMPETITOR_DISCOVERY_MODEL

SUGGEST_TOOL = {
    "type": "function",
    "function": {
        "name": "select_policy_templates",
        "description": "Select one or more policy templates that best match the user's security requirements",
        "parameters": {
            "type": "object",
            "properties": {
                "selected_templates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "template_id": {
                                "type": "string",
                                "description": "The ID of the selected template",
                            },
                            "reason": {
                                "type": "string",
                                "description": "Brief reason why this template matches",
                            },
                        },
                        "required": ["template_id", "reason"],
                    },
                    "description": "List of templates that match the user's requirements",
                },
                "explanation": {
                    "type": "string",
                    "description": "Overall explanation of why these templates were suggested",
                },
            },
            "required": ["selected_templates", "explanation"],
        },
    },
}


class AiPolicySuggester:
    """Suggests policy templates using LLM tool calling."""

    async def suggest(
        self,
        templates: list,
        attack_examples: List[str],
        description: str,
        model: Optional[str] = None,
    ) -> dict:
        system_prompt = self._build_system_prompt(templates)
        user_prompt = self._build_user_prompt(attack_examples, description)
        model = model or DEFAULT_COMPETITOR_DISCOVERY_MODEL

        try:
            response = await litellm.acompletion(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tools=[SUGGEST_TOOL],
                tool_choice={
                    "type": "function",
                    "function": {"name": "select_policy_templates"},
                },
                temperature=0.2,
            )

            tool_calls = response.choices[0].message.tool_calls  # type: ignore
            if not tool_calls:
                return {
                    "selected_templates": [],
                    "explanation": "No templates could be matched to your requirements.",
                }

            result = json.loads(tool_calls[0].function.arguments)

            valid_ids = {t["id"] for t in templates}
            result["selected_templates"] = [
                s
                for s in result.get("selected_templates", [])
                if s.get("template_id") in valid_ids
            ]

            return result
        except Exception as e:
            verbose_proxy_logger.error("AI policy suggestion failed: %s", e)
            raise

    def _build_system_prompt(self, templates: list) -> str:
        template_descriptions = []
        for t in templates:
            examples = t.get("example_sentences", [])
            examples_str = (
                ", ".join(f'"{e}"' for e in examples) if examples else "none"
            )
            entry = (
                f"- ID: {t['id']}\n"
                f"  Title: {t['title']}\n"
                f"  Description: {t['description']}\n"
                f"  Example attacks it protects against: {examples_str}"
            )
            template_descriptions.append(entry)

        return (
            "You are a security policy advisor. The user will describe attacks or content "
            "they want to block. Your job is to select the most relevant policy templates "
            "from the available set. Use the select_policy_templates tool to return your "
            "selections. Only select templates that are clearly relevant to what the user "
            "wants to block.\n\n"
            "Available templates:\n\n" + "\n\n".join(template_descriptions)
        )

    def _build_user_prompt(
        self, attack_examples: List[str], description: str
    ) -> str:
        parts = []
        filtered_examples = [e for e in attack_examples if e.strip()]
        if filtered_examples:
            parts.append("Example attack prompts I want to block:")
            for i, ex in enumerate(filtered_examples, 1):
                parts.append(f"  {i}. {ex}")
        if description.strip():
            parts.append(f"\nDescription of what I want to block: {description}")
        return "\n".join(parts)
