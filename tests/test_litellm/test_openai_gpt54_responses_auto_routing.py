"""
Test automatic routing to OpenAI Responses API for GPT-5.4+ models with tools and reasoning
"""
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath("../.."))

import pytest
import litellm
from litellm.main import responses_api_bridge_check


class TestOpenAIGPT54ResponsesAutoRouting:
    """Test that OpenAI GPT-5.4+ requests with tools and reasoning automatically route to Responses API"""

    def test_gpt_54_with_tools_and_reasoning_routes_to_responses(self):
        """Test that GPT-5.4 with tools and reasoning routes to Responses API"""
        model = "gpt-5.4"
        custom_llm_provider = "openai"
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    }
                }
            }
        ]
        reasoning_effort = "high"

        model_info, updated_model = responses_api_bridge_check(
            model=model,
            custom_llm_provider=custom_llm_provider,
            tools=tools,
            reasoning_effort=reasoning_effort,
        )

        assert model_info.get("mode") == "responses"
        assert updated_model == model

    def test_gpt_54_pro_with_tools_and_reasoning_routes_to_responses(self):
        """Test that GPT-5.4-pro with tools and reasoning routes to Responses API"""
        model = "gpt-5.4-pro"
        custom_llm_provider = "openai"
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform calculation",
                }
            }
        ]
        reasoning_effort = "medium"

        model_info, updated_model = responses_api_bridge_check(
            model=model,
            custom_llm_provider=custom_llm_provider,
            tools=tools,
            reasoning_effort=reasoning_effort,
        )

        assert model_info.get("mode") == "responses"
        assert updated_model == model

    def test_gpt_54_with_tools_but_no_reasoning_does_not_route(self):
        """Test that GPT-5.4 with tools but no reasoning does NOT route to Responses API"""
        model = "gpt-5.4"
        custom_llm_provider = "openai"
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                }
            }
        ]
        reasoning_effort = None

        model_info, updated_model = responses_api_bridge_check(
            model=model,
            custom_llm_provider=custom_llm_provider,
            tools=tools,
            reasoning_effort=reasoning_effort,
        )

        assert model_info.get("mode") != "responses"
        assert updated_model == model

    def test_gpt_54_with_reasoning_none_does_not_route(self):
        """Test that GPT-5.4 with reasoning_effort='none' does NOT route to Responses API"""
        model = "gpt-5.4"
        custom_llm_provider = "openai"
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                }
            }
        ]
        reasoning_effort = "none"

        model_info, updated_model = responses_api_bridge_check(
            model=model,
            custom_llm_provider=custom_llm_provider,
            tools=tools,
            reasoning_effort=reasoning_effort,
        )

        assert model_info.get("mode") != "responses"
        assert updated_model == model

    def test_gpt_54_with_reasoning_but_no_tools_does_not_route(self):
        """Test that GPT-5.4 with reasoning but no tools does NOT route to Responses API"""
        model = "gpt-5.4"
        custom_llm_provider = "openai"
        tools = None
        reasoning_effort = "high"

        model_info, updated_model = responses_api_bridge_check(
            model=model,
            custom_llm_provider=custom_llm_provider,
            tools=tools,
            reasoning_effort=reasoning_effort,
        )

        assert model_info.get("mode") != "responses"
        assert updated_model == model

    def test_gpt_53_with_tools_and_reasoning_does_not_route(self):
        """Test that GPT-5.3 (below 5.4) does NOT route to Responses API"""
        model = "gpt-5.3"
        custom_llm_provider = "openai"
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                }
            }
        ]
        reasoning_effort = "high"

        model_info, updated_model = responses_api_bridge_check(
            model=model,
            custom_llm_provider=custom_llm_provider,
            tools=tools,
            reasoning_effort=reasoning_effort,
        )

        assert model_info.get("mode") != "responses"
        assert updated_model == model

    def test_gpt_52_with_tools_and_reasoning_does_not_route(self):
        """Test that GPT-5.2 (below 5.4) does NOT route to Responses API"""
        model = "gpt-5.2"
        custom_llm_provider = "openai"
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                }
            }
        ]
        reasoning_effort = "high"

        model_info, updated_model = responses_api_bridge_check(
            model=model,
            custom_llm_provider=custom_llm_provider,
            tools=tools,
            reasoning_effort=reasoning_effort,
        )

        assert model_info.get("mode") != "responses"
        assert updated_model == model

    def test_gpt_55_with_tools_and_reasoning_routes_to_responses(self):
        """Test that GPT-5.5 (above 5.4) routes to Responses API"""
        model = "gpt-5.5"
        custom_llm_provider = "openai"
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                }
            }
        ]
        reasoning_effort = "high"

        model_info, updated_model = responses_api_bridge_check(
            model=model,
            custom_llm_provider=custom_llm_provider,
            tools=tools,
            reasoning_effort=reasoning_effort,
        )

        assert model_info.get("mode") == "responses"
        assert updated_model == model

    def test_azure_provider_with_gpt_54_does_not_route(self):
        """Test that Azure provider does NOT auto-route (only OpenAI)"""
        model = "gpt-5.4"
        custom_llm_provider = "azure"
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                }
            }
        ]
        reasoning_effort = "high"

        model_info, updated_model = responses_api_bridge_check(
            model=model,
            custom_llm_provider=custom_llm_provider,
            tools=tools,
            reasoning_effort=reasoning_effort,
        )

        assert model_info.get("mode") != "responses"
        assert updated_model == model

    def test_gpt_54_with_empty_tools_list_does_not_route(self):
        """Test that GPT-5.4 with empty tools list does NOT route to Responses API"""
        model = "gpt-5.4"
        custom_llm_provider = "openai"
        tools = []
        reasoning_effort = "high"

        model_info, updated_model = responses_api_bridge_check(
            model=model,
            custom_llm_provider=custom_llm_provider,
            tools=tools,
            reasoning_effort=reasoning_effort,
        )

        assert model_info.get("mode") != "responses"
        assert updated_model == model

    def test_gpt_54_with_responses_prefix_still_routes(self):
        """Test that responses/ prefix still works for GPT-5.4"""
        model = "responses/gpt-5.4"
        custom_llm_provider = "openai"
        tools = None
        reasoning_effort = None

        model_info, updated_model = responses_api_bridge_check(
            model=model,
            custom_llm_provider=custom_llm_provider,
            tools=tools,
            reasoning_effort=reasoning_effort,
        )

        assert model_info.get("mode") == "responses"
        assert updated_model == "gpt-5.4"

    def test_gpt_54_with_various_reasoning_levels(self):
        """Test different reasoning_effort levels"""
        model = "gpt-5.4"
        custom_llm_provider = "openai"
        tools = [{"type": "function", "function": {"name": "test"}}]

        for reasoning_level in ["low", "medium", "high", "xhigh"]:
            model_info, updated_model = responses_api_bridge_check(
                model=model,
                custom_llm_provider=custom_llm_provider,
                tools=tools,
                reasoning_effort=reasoning_level,
            )

            assert model_info.get("mode") == "responses", f"Failed for reasoning_effort={reasoning_level}"
            assert updated_model == model

    def test_gpt_54_model_with_date_suffix(self):
        """Test GPT-5.4 model with date suffix (e.g., gpt-5.4-2026-03-05)"""
        model = "gpt-5.4-2026-03-05"
        custom_llm_provider = "openai"
        tools = [{"type": "function", "function": {"name": "test"}}]
        reasoning_effort = "high"

        model_info, updated_model = responses_api_bridge_check(
            model=model,
            custom_llm_provider=custom_llm_provider,
            tools=tools,
            reasoning_effort=reasoning_effort,
        )

        assert model_info.get("mode") == "responses"
        assert updated_model == model

    def test_gpt_54_with_openai_prefix(self):
        """Test GPT-5.4 with openai/ prefix"""
        model = "openai/gpt-5.4"
        custom_llm_provider = "openai"
        tools = [{"type": "function", "function": {"name": "test"}}]
        reasoning_effort = "high"

        model_info, updated_model = responses_api_bridge_check(
            model=model,
            custom_llm_provider=custom_llm_provider,
            tools=tools,
            reasoning_effort=reasoning_effort,
        )

        assert model_info.get("mode") == "responses"
        assert updated_model == model


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
