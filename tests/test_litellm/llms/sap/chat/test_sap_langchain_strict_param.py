"""
Test that LangChain's strict parameter is handled correctly for SAP GenAI Hub.

LangChain agents automatically set strict=true when using response_format with
json_schema. This parameter gets passed at the top level of optional_params,
but SAP AI Core Orchestration API does not accept it as a model parameter for
GPT models (returns 400 error).

These tests verify that:
1. The strict parameter is filtered from model_params
2. The strict parameter inside response_format.json_schema is preserved
3. The fix works for both GPT and Anthropic models
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from litellm.llms.sap.chat.transformation import GenAIHubOrchestrationConfig


class TestLangChainAgentCompatibility:
    """Test compatibility with LangChain agent's strict parameter behavior."""

    def test_langchain_create_react_agent_style_params(self):
        """Simulate LangChain's create_react_agent with response_format schema.

        LangChain's create_react_agent passes parameters like:
        {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "AgentResponse",
                    "strict": true,
                    "schema": {...}
                }
            },
            "strict": true,  # Also passed at top level
            "temperature": 0
        }
        """
        config = GenAIHubOrchestrationConfig()

        # Simulate LangChain agent parameters
        langchain_params = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "AgentResponse",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "thought": {"type": "string"},
                            "action": {"type": "string"},
                            "action_input": {"type": "string"}
                        },
                        "required": ["thought", "action", "action_input"]
                    }
                }
            },
            "strict": True,  # LangChain adds this at top level
            "temperature": 0
        }

        request = config.transform_request(
            model="gpt-4o",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            optional_params=langchain_params,
            litellm_params={},
            headers={},
        )

        # Verify strict is NOT in model.params (would cause 400 error)
        model_params = request["config"]["modules"]["prompt_templating"]["model"]["params"]
        assert "strict" not in model_params, "strict should be filtered from model.params"

        # Verify other params are preserved
        assert model_params.get("temperature") == 0

        # Verify strict inside json_schema IS preserved
        prompt_config = request["config"]["modules"]["prompt_templating"]["prompt"]
        assert prompt_config["response_format"]["json_schema"]["strict"] is True

    def test_langchain_structured_output_style_params(self):
        """Simulate LangChain's with_structured_output() method.

        When using model.with_structured_output(Schema), LangChain passes:
        {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "Schema", "strict": true, "schema": {...}}
            },
            "strict": true
        }
        """
        config = GenAIHubOrchestrationConfig()

        # Pydantic-style schema from LangChain
        structured_output_params = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "SearchQuery",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "max_results": {"type": "integer", "default": 10}
                        },
                        "required": ["query"]
                    }
                }
            },
            "strict": True,
            "max_tokens": 1000
        }

        request = config.transform_request(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Search for Python tutorials"}],
            optional_params=structured_output_params,
            litellm_params={},
            headers={},
        )

        model_params = request["config"]["modules"]["prompt_templating"]["model"]["params"]
        assert "strict" not in model_params
        assert model_params.get("max_tokens") == 1000

    def test_langchain_tool_calling_agent_params(self):
        """Simulate LangChain tool calling agent with strict mode.

        Tool calling agents may pass strict for both tools and response_format.
        """
        config = GenAIHubOrchestrationConfig()

        tool_agent_params = {
            "tools": [{
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    },
                    "strict": True  # Tool-level strict
                }
            }],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "ToolResponse",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"result": {"type": "string"}}
                    }
                }
            },
            "strict": True,  # Top-level strict from LangChain
            "tool_choice": "auto"
        }

        request = config.transform_request(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Search for weather"}],
            optional_params=tool_agent_params,
            litellm_params={},
            headers={},
        )

        # Top-level strict should be filtered
        model_params = request["config"]["modules"]["prompt_templating"]["model"]["params"]
        assert "strict" not in model_params

        # Tools should be included
        prompt_config = request["config"]["modules"]["prompt_templating"]["prompt"]
        assert "tools" in prompt_config
        assert len(prompt_config["tools"]) == 1

        # Tool's strict parameter should be preserved (it's inside the tool definition)
        assert prompt_config["tools"][0]["function"]["strict"] is True

    def test_gpt4_model_with_langchain_strict(self):
        """Test that gpt-4 (which doesn't support response_format) handles strict gracefully."""
        config = GenAIHubOrchestrationConfig()

        # Even if LangChain passes strict, it should be filtered
        request = config.transform_request(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={"strict": True, "temperature": 0.5},
            litellm_params={},
            headers={},
        )

        model_params = request["config"]["modules"]["prompt_templating"]["model"]["params"]
        assert "strict" not in model_params
        assert model_params.get("temperature") == 0.5

    def test_anthropic_model_preserves_strict(self):
        """Test Anthropic models preserve top-level strict (SAP API accepts it for Anthropic)."""
        config = GenAIHubOrchestrationConfig()

        langchain_params = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "Response",
                    "strict": True,
                    "schema": {"type": "object", "properties": {"answer": {"type": "string"}}}
                }
            },
            "strict": True,
            "max_tokens": 2000
        }

        request = config.transform_request(
            model="anthropic--claude-3-5-sonnet",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params=langchain_params,
            litellm_params={},
            headers={},
        )

        model_params = request["config"]["modules"]["prompt_templating"]["model"]["params"]
        # Anthropic models CAN have strict in model.params (SAP API accepts it)
        assert model_params.get("strict") is True
        assert model_params.get("max_tokens") == 2000

        # json_schema strict should also be preserved
        prompt_config = request["config"]["modules"]["prompt_templating"]["prompt"]
        assert prompt_config["response_format"]["json_schema"]["strict"] is True


class TestLangChainRequestPayloadStructure:
    """Test the final request payload structure matches SAP API expectations."""

    def test_request_payload_structure_without_strict_in_params(self):
        """Verify the complete request structure is correct for SAP Orchestration API."""
        config = GenAIHubOrchestrationConfig()

        request = config.transform_request(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"}
            ],
            optional_params={
                "strict": True,
                "temperature": 0.7,
                "max_tokens": 500,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "MathAnswer",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {"answer": {"type": "integer"}},
                            "required": ["answer"]
                        }
                    }
                }
            },
            litellm_params={},
            headers={},
        )

        # Verify overall structure
        assert "config" in request
        assert "modules" in request["config"]
        assert "prompt_templating" in request["config"]["modules"]

        prompt_templating = request["config"]["modules"]["prompt_templating"]

        # Verify model section
        assert "model" in prompt_templating
        assert prompt_templating["model"]["name"] == "gpt-4o"
        assert "params" in prompt_templating["model"]

        model_params = prompt_templating["model"]["params"]
        # These should be in params
        assert model_params["temperature"] == 0.7
        assert model_params["max_tokens"] == 500
        # strict should NOT be in params
        assert "strict" not in model_params

        # Verify prompt section
        assert "prompt" in prompt_templating
        prompt = prompt_templating["prompt"]

        # response_format should be in prompt, not in model.params
        assert "response_format" in prompt
        assert prompt["response_format"]["type"] == "json_schema"
        assert prompt["response_format"]["json_schema"]["strict"] is True

    def test_no_strict_anywhere_in_model_params_section(self):
        """Ensure strict never appears in model.params regardless of input."""
        config = GenAIHubOrchestrationConfig()

        # Try various ways strict might be passed
        test_cases = [
            {"strict": True},
            {"strict": False},
            {"strict": True, "temperature": 0},
            {"response_format": {"type": "json_object"}, "strict": True},
        ]

        for params in test_cases:
            request = config.transform_request(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Test"}],
                optional_params=params.copy(),
                litellm_params={},
                headers={},
            )

            model_params = request["config"]["modules"]["prompt_templating"]["model"]["params"]
            assert "strict" not in model_params, f"strict leaked into model.params with input: {params}"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_strict_false_also_filtered(self):
        """Even strict=false should be filtered (it's not a valid model param)."""
        config = GenAIHubOrchestrationConfig()

        request = config.transform_request(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Test"}],
            optional_params={"strict": False, "temperature": 0.5},
            litellm_params={},
            headers={},
        )

        model_params = request["config"]["modules"]["prompt_templating"]["model"]["params"]
        assert "strict" not in model_params

    def test_empty_optional_params(self):
        """Should work with empty optional_params."""
        config = GenAIHubOrchestrationConfig()

        request = config.transform_request(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Test"}],
            optional_params={},
            litellm_params={},
            headers={},
        )

        model_params = request["config"]["modules"]["prompt_templating"]["model"]["params"]
        assert "strict" not in model_params

    def test_only_strict_in_params(self):
        """Should work when strict is the only param."""
        config = GenAIHubOrchestrationConfig()

        request = config.transform_request(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Test"}],
            optional_params={"strict": True},
            litellm_params={},
            headers={},
        )

        model_params = request["config"]["modules"]["prompt_templating"]["model"]["params"]
        assert "strict" not in model_params
        # model_params might be empty or have other defaults, but no strict

    def test_gpt_models_filter_strict(self):
        """Verify strict is filtered for all GPT model variants."""
        config = GenAIHubOrchestrationConfig()

        gpt_models = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
        ]

        for model in gpt_models:
            request = config.transform_request(
                model=model,
                messages=[{"role": "user", "content": "Test"}],
                optional_params={"strict": True, "temperature": 0.5},
                litellm_params={},
                headers={},
            )

            model_params = request["config"]["modules"]["prompt_templating"]["model"]["params"]
            assert "strict" not in model_params, f"strict should be filtered for GPT model: {model}"
            assert model_params.get("temperature") == 0.5, f"temperature missing for model: {model}"

    def test_non_gpt_models_preserve_strict(self):
        """Verify strict is preserved for non-GPT models (Anthropic, Gemini, Mistral, etc.)."""
        config = GenAIHubOrchestrationConfig()

        non_gpt_models = [
            "anthropic--claude-3-5-sonnet",
            "anthropic--claude-3-opus",
            "gemini-1.5-pro",
            "mistral-large",
        ]

        for model in non_gpt_models:
            request = config.transform_request(
                model=model,
                messages=[{"role": "user", "content": "Test"}],
                optional_params={"strict": True, "temperature": 0.5},
                litellm_params={},
                headers={},
            )

            model_params = request["config"]["modules"]["prompt_templating"]["model"]["params"]
            assert model_params.get("strict") is True, f"strict should be preserved for non-GPT model: {model}"
            assert model_params.get("temperature") == 0.5, f"temperature missing for model: {model}"
