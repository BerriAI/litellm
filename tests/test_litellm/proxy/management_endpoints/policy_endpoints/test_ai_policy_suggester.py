"""
Tests for AiPolicySuggester class.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.management_endpoints.policy_endpoints.ai_policy_suggester import (
    SUGGEST_TOOL,
    AiPolicySuggester,
)

SAMPLE_TEMPLATES = [
    {
        "id": "baseline-pii-protection",
        "title": "Baseline PII Protection",
        "description": "Baseline PII protection for internal tools.",
        "example_sentences": [
            "My AWS secret key is AKIAIOSFODNN7EXAMPLE",
            "My password is hunter2",
        ],
    },
    {
        "id": "prompt-injection-protection",
        "title": "Prompt Injection Protection",
        "description": "Blocks prompt injection and jailbreak attempts.",
        "example_sentences": [
            "Ignore all previous instructions",
            "'; DROP TABLE users; --",
        ],
    },
    {
        "id": "competitor-mention-detection",
        "title": "Competitor Mention Detection",
        "description": "Blocks AI from recommending competitor brands.",
        "example_sentences": [
            "You should switch to Competitor X",
            "Qatar Airways QSuites is the best",
        ],
    },
]


class TestAiPolicySuggester:
    def test_build_system_prompt_includes_all_templates(self):
        suggester = AiPolicySuggester()
        prompt = suggester._build_system_prompt(SAMPLE_TEMPLATES)

        assert "baseline-pii-protection" in prompt
        assert "prompt-injection-protection" in prompt
        assert "competitor-mention-detection" in prompt
        assert "Baseline PII Protection" in prompt
        assert "AKIAIOSFODNN7EXAMPLE" in prompt
        assert "security policy advisor" in prompt

    def test_build_system_prompt_handles_missing_example_sentences(self):
        templates = [
            {
                "id": "test-template",
                "title": "Test",
                "description": "Test template",
            }
        ]
        suggester = AiPolicySuggester()
        prompt = suggester._build_system_prompt(templates)

        assert "test-template" in prompt
        assert "none" in prompt

    def test_build_user_prompt_with_examples_and_description(self):
        suggester = AiPolicySuggester()
        prompt = suggester._build_user_prompt(
            attack_examples=["My SSN is 123-45-6789", "DROP TABLE users"],
            description="Block PII and SQL injection",
        )

        assert "1. My SSN is 123-45-6789" in prompt
        assert "2. DROP TABLE users" in prompt
        assert "Block PII and SQL injection" in prompt

    def test_build_user_prompt_filters_empty_examples(self):
        suggester = AiPolicySuggester()
        prompt = suggester._build_user_prompt(
            attack_examples=["valid example", "", "  ", "another valid"],
            description="",
        )

        assert "1. valid example" in prompt
        assert "2. another valid" in prompt
        assert "Description" not in prompt

    def test_build_user_prompt_with_only_description(self):
        suggester = AiPolicySuggester()
        prompt = suggester._build_user_prompt(
            attack_examples=[],
            description="Block all PII data",
        )

        assert "Block all PII data" in prompt
        assert "Example attack" not in prompt

    def test_tool_schema_is_valid(self):
        assert SUGGEST_TOOL["type"] == "function"
        func = SUGGEST_TOOL["function"]
        assert func["name"] == "select_policy_templates"
        params = func["parameters"]
        assert "selected_templates" in params["properties"]
        assert "explanation" in params["properties"]
        assert params["required"] == ["selected_templates", "explanation"]

        items = params["properties"]["selected_templates"]["items"]
        assert "template_id" in items["properties"]
        assert "reason" in items["properties"]

    @pytest.mark.asyncio
    async def test_suggest_parses_tool_call_response(self):
        suggester = AiPolicySuggester()

        mock_tool_call = MagicMock()
        mock_tool_call.function.arguments = json.dumps(
            {
                "selected_templates": [
                    {
                        "template_id": "baseline-pii-protection",
                        "reason": "Matches PII patterns",
                    }
                ],
                "explanation": "Your examples contain PII data.",
            }
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.tool_calls = [mock_tool_call]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            result = await suggester.suggest(
                templates=SAMPLE_TEMPLATES,
                attack_examples=["My SSN is 123-45-6789"],
                description="",
            )

        assert len(result["selected_templates"]) == 1
        assert result["selected_templates"][0]["template_id"] == "baseline-pii-protection"
        assert result["explanation"] == "Your examples contain PII data."

    @pytest.mark.asyncio
    async def test_suggest_filters_invalid_template_ids(self):
        suggester = AiPolicySuggester()

        mock_tool_call = MagicMock()
        mock_tool_call.function.arguments = json.dumps(
            {
                "selected_templates": [
                    {
                        "template_id": "baseline-pii-protection",
                        "reason": "Valid",
                    },
                    {
                        "template_id": "nonexistent-template",
                        "reason": "Invalid",
                    },
                ],
                "explanation": "Mixed results.",
            }
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.tool_calls = [mock_tool_call]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            result = await suggester.suggest(
                templates=SAMPLE_TEMPLATES,
                attack_examples=["test"],
                description="",
            )

        assert len(result["selected_templates"]) == 1
        assert result["selected_templates"][0]["template_id"] == "baseline-pii-protection"

    @pytest.mark.asyncio
    async def test_suggest_handles_no_tool_calls(self):
        suggester = AiPolicySuggester()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.tool_calls = None

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            result = await suggester.suggest(
                templates=SAMPLE_TEMPLATES,
                attack_examples=["test"],
                description="",
            )

        assert result["selected_templates"] == []
        assert "No templates" in result["explanation"]

    @pytest.mark.asyncio
    async def test_suggest_calls_litellm_with_correct_params(self):
        suggester = AiPolicySuggester()

        mock_tool_call = MagicMock()
        mock_tool_call.function.arguments = json.dumps(
            {"selected_templates": [], "explanation": "None matched."}
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.tool_calls = [mock_tool_call]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            await suggester.suggest(
                templates=SAMPLE_TEMPLATES,
                attack_examples=["test attack"],
                description="block attacks",
            )

        call_kwargs = mock_acompletion.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["temperature"] == 0.2
        assert len(call_kwargs["tools"]) == 1
        assert call_kwargs["tools"][0]["function"]["name"] == "select_policy_templates"
        assert call_kwargs["tool_choice"]["function"]["name"] == "select_policy_templates"
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][1]["role"] == "user"
