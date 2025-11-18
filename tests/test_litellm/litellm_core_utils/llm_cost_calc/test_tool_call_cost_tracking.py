import json
import os
import sys
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import litellm
from litellm.litellm_core_utils.llm_cost_calc.tool_call_cost_tracking import (
    StandardBuiltInToolCostTracking,
)
from litellm.types.llms.openai import FileSearchTool, WebSearchOptions
from litellm.types.utils import ModelInfo, ModelResponse, StandardBuiltInToolsParams

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


# Test basic web search cost calculations
def test_web_search_cost_low():
    web_search_options = WebSearchOptions(search_context_size="low")
    model_info = litellm.get_model_info("gpt-4o-search-preview")

    cost = StandardBuiltInToolCostTracking.get_cost_for_web_search(
        web_search_options=web_search_options, model_info=model_info
    )

    assert (
        cost == model_info["search_context_cost_per_query"]["search_context_size_low"]
    )


def test_web_search_cost_medium():
    web_search_options = WebSearchOptions(search_context_size="medium")
    model_info = litellm.get_model_info("gpt-4o-search-preview")

    cost = StandardBuiltInToolCostTracking.get_cost_for_web_search(
        web_search_options=web_search_options, model_info=model_info
    )

    assert (
        cost
        == model_info["search_context_cost_per_query"]["search_context_size_medium"]
    )


def test_web_search_cost_high():
    web_search_options = WebSearchOptions(search_context_size="high")
    model_info = litellm.get_model_info("gpt-4o-search-preview")

    cost = StandardBuiltInToolCostTracking.get_cost_for_web_search(
        web_search_options=web_search_options, model_info=model_info
    )

    assert (
        cost == model_info["search_context_cost_per_query"]["search_context_size_high"]
    )


# Test file search cost calculation
def test_file_search_cost():
    file_search = FileSearchTool(type="file_search")
    cost = StandardBuiltInToolCostTracking.get_cost_for_file_search(
        file_search=file_search
    )
    assert cost == 0.0025  # $2.50/1000 calls = 0.0025 per call


# Test edge cases
def test_none_inputs():
    # Test with None inputs
    assert (
        StandardBuiltInToolCostTracking.get_cost_for_web_search(
            web_search_options=None, model_info=None
        )
        == 0.0
    )
    assert (
        StandardBuiltInToolCostTracking.get_cost_for_file_search(file_search=None)
        == 0.0
    )


# Test the main get_cost_for_built_in_tools method
def test_get_cost_for_built_in_tools_web_search():
    model = "gpt-4"
    standard_built_in_tools_params = StandardBuiltInToolsParams(
        web_search_options=WebSearchOptions(search_context_size="medium")
    )

    cost = StandardBuiltInToolCostTracking.get_cost_for_built_in_tools(
        model=model,
        usage=None,
        response_object=None,
        standard_built_in_tools_params=standard_built_in_tools_params,
    )

    assert isinstance(cost, float)


def test_get_cost_for_built_in_tools_file_search():
    """
    Test that the cost for a file search is 0.00 when no response object is provided
    """
    model = "gpt-4"
    standard_built_in_tools_params = StandardBuiltInToolsParams(
        file_search=FileSearchTool(type="file_search")
    )

    cost = StandardBuiltInToolCostTracking.get_cost_for_built_in_tools(
        model=model,
        response_object=None,
        usage=None,
        standard_built_in_tools_params=standard_built_in_tools_params,
    )

    assert cost == 0.00


def test_get_cost_for_anthropic_web_search():
    """
    Test that the cost for a web search is 0.00 when no response object is provided
    """
    from litellm.types.utils import ServerToolUse, Usage

    model = "claude-3-7-sonnet-latest"
    usage = Usage(server_tool_use=ServerToolUse(web_search_requests=1))
    cost = StandardBuiltInToolCostTracking.get_cost_for_built_in_tools(
        model=model,
        usage=usage,
        response_object=None,
        standard_built_in_tools_params=None,
    )
    assert cost > 0.0


@pytest.mark.parametrize(
    "model", ["gemini/gemini-2.0-flash-001", "gemini-2.0-flash-001"]
)
def test_get_cost_for_gemini_web_search(model):
    """
    Test that the cost for a web search is 0.00 when no response object is provided
    """
    from litellm.types.utils import PromptTokensDetailsWrapper, Usage

    usage = Usage(
        prompt_tokens_details=PromptTokensDetailsWrapper(web_search_requests=1)
    )
    cost = StandardBuiltInToolCostTracking.get_cost_for_built_in_tools(
        model=model,
        usage=usage,
        response_object=None,
        standard_built_in_tools_params=None,
    )
    assert cost > 0.0


@pytest.mark.parametrize(
    "model,custom_llm_provider",
    [
        ("vertex_ai/gemini-2.5-flash", "vertex_ai"),
        ("gemini-2.5-flash", "vertex_ai"),
    ],
)
def test_get_cost_for_vertex_ai_gemini_web_search(model, custom_llm_provider):
    """
    Test that Vertex AI Gemini web search costs are tracked when passing
    a ModelResponse with usage.prompt_tokens_details.web_search_requests.

    This tests the fix for: https://github.com/BerriAI/litellm/issues/XXXXX

    The issue: When a ModelResponse is passed, the detection logic only checks
    for url_citation annotations, not usage.prompt_tokens_details.web_search_requests.
    This causes Vertex AI grounding costs to not be tracked.
    """
    from litellm.types.utils import PromptTokensDetailsWrapper, Usage, Choices, Message

    # Create a realistic ModelResponse like what Vertex AI returns
    response = ModelResponse(
        id="test-id",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    content="Test response with grounding",
                    role="assistant"
                )
            )
        ],
        created=1234567890,
        model=model,
        object="chat.completion",
        system_fingerprint=None,
    )

    # Add usage with web_search_requests (how Vertex AI indicates grounding was used)
    usage = Usage(
        prompt_tokens=11,
        completion_tokens=100,
        total_tokens=111,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            text_tokens=11,
            web_search_requests=1  # This should trigger grounding cost
        )
    )
    response.usage = usage

    # Calculate cost - should include grounding cost
    cost = StandardBuiltInToolCostTracking.get_cost_for_built_in_tools(
        model=model,
        usage=usage,
        response_object=response,  # Pass the ModelResponse
        custom_llm_provider=custom_llm_provider,
        standard_built_in_tools_params=None,
    )

    # Vertex AI charges $0.035 per grounded request
    assert cost == 0.035, f"Expected $0.035 grounding cost, got ${cost}"


def test_azure_assistant_features_integrated_cost_tracking():
    """
    Test integrated cost tracking for Azure assistant features.
    """
    model = "azure/gpt-4o"
    
    # Test with multiple Azure assistant features
    standard_built_in_tools_params = StandardBuiltInToolsParams(
        vector_store_usage={"storage_gb": 1.0, "days": 10},
        computer_use_usage={"input_tokens": 1000, "output_tokens": 500},
        code_interpreter_sessions=2,
    )

    cost = StandardBuiltInToolCostTracking.get_cost_for_built_in_tools(
        model=model,
        response_object=None,
        usage=None,
        custom_llm_provider="azure",
        standard_built_in_tools_params=standard_built_in_tools_params,
    )
    
    # Should calculate costs for:
    # - Vector store: 1.0 * 10 * 0.1 = $1.00
    # - Computer use: (1000/1000 * 3.0) + (500/1000 * 12.0) = $9.00  
    # - Code interpreter: 2 * 0.03 = $0.06
    # Total: $10.06
    expected_cost = 1.0 + 9.0 + 0.06
    assert abs(cost - expected_cost) < 0.01, f"Expected ~{expected_cost}, got {cost}"


# Note: File search integration test removed due to complex annotation detection logic
# The unit tests in test_azure_assistant_cost_tracking.py provide comprehensive coverage
