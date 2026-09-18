import pytest

import litellm
from litellm.litellm_core_utils.llm_cost_calc.tool_call_cost_tracking import (
    StandardBuiltInToolCostTracking,
)
from litellm.types.llms.openai import FileSearchTool, ResponsesAPIResponse, WebSearchOptions
from litellm.types.utils import ModelResponse, StandardBuiltInToolsParams


def test_web_search_cost_low():
    web_search_options = WebSearchOptions(search_context_size="low")
    model_info = litellm.get_model_info("gpt-4o-search-preview")

    cost = StandardBuiltInToolCostTracking.get_cost_for_web_search(
        web_search_options=web_search_options, model_info=model_info
    )

    assert cost == model_info["search_context_cost_per_query"]["search_context_size_low"]


def test_web_search_cost_medium():
    web_search_options = WebSearchOptions(search_context_size="medium")
    model_info = litellm.get_model_info("gpt-4o-search-preview")

    cost = StandardBuiltInToolCostTracking.get_cost_for_web_search(
        web_search_options=web_search_options, model_info=model_info
    )

    assert cost == model_info["search_context_cost_per_query"]["search_context_size_medium"]


def test_web_search_cost_high():
    web_search_options = WebSearchOptions(search_context_size="high")
    model_info = litellm.get_model_info("gpt-4o-search-preview")

    cost = StandardBuiltInToolCostTracking.get_cost_for_web_search(
        web_search_options=web_search_options, model_info=model_info
    )

    assert cost == model_info["search_context_cost_per_query"]["search_context_size_high"]


# Test file search cost calculation
def test_file_search_cost():
    file_search = FileSearchTool(type="file_search")
    cost = StandardBuiltInToolCostTracking.get_cost_for_file_search(file_search=file_search)
    assert cost == 0.0025  # $2.50/1000 calls = 0.0025 per call


# Test edge cases
def test_none_inputs():
    # Test with None inputs
    assert StandardBuiltInToolCostTracking.get_cost_for_web_search(web_search_options=None, model_info=None) == 0.0
    assert StandardBuiltInToolCostTracking.get_cost_for_file_search(file_search=None) == 0.0


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
    standard_built_in_tools_params = StandardBuiltInToolsParams(file_search=FileSearchTool(type="file_search"))

    cost = StandardBuiltInToolCostTracking.get_cost_for_built_in_tools(
        model=model,
        response_object=None,
        usage=None,
        standard_built_in_tools_params=standard_built_in_tools_params,
    )

    assert cost == 0.00


def test_get_cost_for_anthropic_web_search():
    """
    Test that Anthropic web search cost is tracked when usage.server_tool_use.web_search_requests
    is set. Use claude-3-7-sonnet-20250219 (has search_context_cost_per_query) and
    custom_llm_provider=anthropic so get_cost_for_anthropic_web_search is invoked.
    """
    from litellm.types.utils import ServerToolUse, Usage

    model = "claude-3-7-sonnet-20250219"
    usage = Usage(server_tool_use=ServerToolUse(web_search_requests=1))
    cost = StandardBuiltInToolCostTracking.get_cost_for_built_in_tools(
        model=model,
        usage=usage,
        response_object=None,
        standard_built_in_tools_params=None,
        custom_llm_provider="anthropic",
    )
    assert cost > 0.0


def test_get_cost_for_anthropic_web_search_with_server_tool_use_dict():
    """
    Anthropic-compatible passthrough responses can construct Usage from a raw
    usage payload. Ensure dict server_tool_use values are normalized before
    built-in tool cost tracking reads server_tool_use.web_search_requests.
    """
    from litellm.types.utils import ServerToolUse, Usage

    usage = Usage(server_tool_use={"web_search_requests": 1})

    assert isinstance(usage.server_tool_use, ServerToolUse)
    assert StandardBuiltInToolCostTracking.response_object_includes_web_search_call(response_object=None, usage=usage)


def test_anthropic_web_search_cost_from_raw_response_dict_when_usage_drops_server_tool_use():
    """
    Regression: on the Anthropic /v1/messages sync cost path the response is the raw
    Anthropic dict while the reconstructed OpenAI-shape Usage drops server_tool_use.
    The web-search fee must still be charged by reading the count off the raw dict,
    and the passed-in Usage must not be mutated.
    """
    from litellm.types.utils import Usage

    model = "claude-3-7-sonnet-20250219"
    web_search_requests = 3
    raw_response = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "server_tool_use": {"web_search_requests": web_search_requests},
        },
    }
    usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    assert getattr(usage, "server_tool_use", None) is None

    cost = StandardBuiltInToolCostTracking.get_cost_for_built_in_tools(
        model=model,
        usage=usage,
        response_object=raw_response,
        custom_llm_provider="anthropic",
        standard_built_in_tools_params=None,
    )

    per_query_cost = litellm.get_model_info(model)["search_context_cost_per_query"]["search_context_size_medium"]
    assert cost == per_query_cost * web_search_requests
    assert cost > 0.0
    assert getattr(usage, "server_tool_use", None) is None


def test_anthropic_web_search_cost_from_raw_response_dict_when_usage_is_none():
    """
    Regression: when a caller hands the cost tracker a raw Anthropic dict without a
    parallel Usage object, the web-search fee must still be priced per request from
    usage.server_tool_use.web_search_requests on the dict instead of falling back to
    the flat search_context_size_medium tier.
    """
    model = "claude-3-7-sonnet-20250219"
    web_search_requests = 4
    raw_response = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "server_tool_use": {"web_search_requests": web_search_requests},
        },
    }

    cost = StandardBuiltInToolCostTracking.get_cost_for_built_in_tools(
        model=model,
        usage=None,
        response_object=raw_response,
        custom_llm_provider="anthropic",
        standard_built_in_tools_params=None,
    )

    per_query_cost = litellm.get_model_info(model)["search_context_cost_per_query"]["search_context_size_medium"]
    assert cost == per_query_cost * web_search_requests


def test_anthropic_web_search_zero_requests_from_raw_response_charges_zero():
    """
    Regression: a raw Anthropic dict reporting zero web search requests must price
    the call at zero rather than charging the default medium-tier fee.
    """
    model = "claude-3-7-sonnet-20250219"
    raw_response = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "server_tool_use": {"web_search_requests": 0},
        },
    }

    cost = StandardBuiltInToolCostTracking.get_cost_for_built_in_tools(
        model=model,
        usage=None,
        response_object=raw_response,
        custom_llm_provider="anthropic",
        standard_built_in_tools_params=None,
    )

    assert cost == 0.0


def test_anthropic_response_usage_block_preserves_server_tool_use():
    """
    Regression: AnthropicResponse.model_validate(...).model_dump() must keep
    server_tool_use so the /v1/messages logging fallback does not strip the
    web-search usage before cost tracking sees it.
    """
    from litellm.types.llms.anthropic import AnthropicResponse

    raw_response = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-7-sonnet-20250219",
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "server_tool_use": {"web_search_requests": 2},
        },
    }

    dumped_usage = AnthropicResponse.model_validate(raw_response).model_dump()["usage"]

    assert dumped_usage["server_tool_use"] == {"web_search_requests": 2}


@pytest.mark.parametrize("model", ["gemini/gemini-2.0-flash-001", "gemini-2.0-flash-001"])
def test_get_cost_for_gemini_web_search(model):
    """
    Test that the cost for a web search is 0.00 when no response object is provided
    """
    from litellm.types.utils import PromptTokensDetailsWrapper, Usage

    usage = Usage(prompt_tokens_details=PromptTokensDetailsWrapper(web_search_requests=1))
    cost = StandardBuiltInToolCostTracking.get_cost_for_built_in_tools(
        model=model,
        usage=usage,
        response_object=None,
        standard_built_in_tools_params=None,
    )
    assert cost > 0.0


def test_completion_cost_includes_web_search_without_standard_built_in_tools_params():
    """
    Test that completion_cost includes web search cost even when
    standard_built_in_tools_params is None.

    Regression test: the early-exit guard `if standard_built_in_tools_params:`
    in completion_cost was skipping get_cost_for_built_in_tools entirely,
    causing under-counted costs for providers like Vertex AI Gemini that
    report web search usage via usage.prompt_tokens_details.web_search_requests.
    """
    from litellm.types.utils import Choices, Message, PromptTokensDetailsWrapper, Usage

    response = ModelResponse(
        id="test-id",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(content="test", role="assistant"),
            )
        ],
        created=1234567890,
        model="gemini-2.5-flash",
        object="chat.completion",
    )
    response.usage = Usage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        prompt_tokens_details=PromptTokensDetailsWrapper(web_search_requests=1),
    )

    cost = litellm.completion_cost(
        completion_response=response,
        model="gemini-2.5-flash",
        custom_llm_provider="vertex_ai",
        standard_built_in_tools_params=None,
    )

    web_search_cost = StandardBuiltInToolCostTracking.get_cost_for_built_in_tools(
        model="gemini-2.5-flash",
        usage=response.usage,
        response_object=response,
        standard_built_in_tools_params=None,
        custom_llm_provider="vertex_ai",
    )

    assert web_search_cost > 0, "Web search cost should be non-zero"
    assert cost >= web_search_cost, f"completion_cost ({cost}) should include web search cost ({web_search_cost})"


def test_gemini_combined_search_and_maps_costs_are_additive(local_model_cost_map):
    """A prompt grounded with both Google Search and Google Maps pays both fees."""
    from litellm.types.utils import PromptTokensDetailsWrapper, Usage

    model = "gemini/gemini-3.5-flash"
    model_info = litellm.get_model_info(model)
    search_rate = model_info["search_context_cost_per_query"]["search_context_size_medium"]
    maps_rate = model_info["google_maps_grounding_cost_per_query"]

    usage = Usage(
        prompt_tokens=15,
        completion_tokens=100,
        total_tokens=115,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            text_tokens=15, web_search_requests=2, google_maps_grounding_requests=1
        ),
    )
    cost = StandardBuiltInToolCostTracking.get_cost_for_built_in_tools(
        model=model,
        usage=usage,
        response_object=None,
        custom_llm_provider="gemini",
        standard_built_in_tools_params=None,
    )
    assert cost == pytest.approx(search_rate * 2 + maps_rate)


def _openai_responses_with_web_search_calls(model, num_calls):
    from openai.types.responses.response_function_web_search import (
        ActionSearch,
        ResponseFunctionWebSearch,
    )

    output = [
        ResponseFunctionWebSearch(
            id=f"ws_{i}",
            type="web_search_call",
            status="completed",
            action=ActionSearch(type="search", query="latest news"),
        )
        for i in range(num_calls)
    ]
    return ResponsesAPIResponse(
        id="resp_1",
        created_at=0,
        model=model,
        object="response",
        output=output,
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


def test_openai_responses_web_search_multiplied_by_call_count(local_model_cost_map):
    """
    Regression for LIT-5013 bug 2: web_search_call detection was binary, so a Responses output with
    multiple web searches was charged once. gpt-4o-search-preview carries per-call pricing; N calls
    must bill N times, and a single call must still bill exactly once.
    """
    from litellm.types.utils import Usage

    model = "gpt-4o-search-preview"
    per_call = litellm.get_model_info(model)["search_context_cost_per_query"]["search_context_size_medium"]
    usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    for num_calls in (1, 3):
        response = _openai_responses_with_web_search_calls(model, num_calls=num_calls)
        cost = StandardBuiltInToolCostTracking.get_cost_for_built_in_tools(
            model=model,
            response_object=response,
            usage=usage,
            custom_llm_provider="openai",
            standard_built_in_tools_params=None,
        )
        assert cost == pytest.approx(num_calls * per_call), (
            f"{num_calls} web searches must bill {num_calls} x ${per_call}, got ${cost}"
        )


def test_web_search_call_count_reads_dict_output_items(local_model_cost_map):
    """
    Regression: output items that fail OpenAI SDK validation (e.g. xAI web_search_call
    items without an "action" field) stay plain dicts in the output union. The per-call
    counter must read their "type" key like the detection gate does, instead of flooring
    a multi-search response to a single billable search.
    """
    from litellm.types.utils import Usage

    model = "gpt-4o-search-preview"
    per_call = litellm.get_model_info(model)["search_context_cost_per_query"]["search_context_size_medium"]

    response = ResponsesAPIResponse.model_validate(
        {
            "id": "resp_1",
            "created_at": 1754900000,
            "model": model,
            "object": "response",
            "status": "completed",
            "output": [{"type": "web_search_call", "id": f"ws_{i}", "status": "completed"} for i in range(3)],
        }
    )
    assert all(isinstance(item, dict) for item in response.output)

    cost = StandardBuiltInToolCostTracking.get_cost_for_built_in_tools(
        model=model,
        response_object=response,
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        custom_llm_provider="openai",
        standard_built_in_tools_params=None,
    )

    assert cost == pytest.approx(3 * per_call), f"3 dict-shaped web searches must bill 3 x ${per_call}, got ${cost}"


# Note: File search integration test removed due to complex annotation detection logic
# The unit tests in test_azure_assistant_cost_tracking.py provide comprehensive coverage


def test_response_includes_output_type_reads_dict_output_items():
    """
    Regression: output items that fail OpenAI SDK validation (e.g. xAI web_search_call
    items without an "action" field) stay plain dicts in the output union. The gate must
    read their "type" key instead of returning False and skipping the web search fee.
    """

    response = ResponsesAPIResponse.model_validate(
        {
            "id": "resp_1",
            "created_at": 1754900000,
            "model": "grok-4",
            "object": "response",
            "status": "completed",
            "output": [{"type": "web_search_call", "id": "ws_1", "status": "completed"}],
        }
    )

    assert isinstance(response.output[0], dict)
    assert StandardBuiltInToolCostTracking.response_includes_output_type(
        response_object=response, output_type="web_search_call"
    )
    assert not StandardBuiltInToolCostTracking.response_includes_output_type(
        response_object=response, output_type="file_search_call"
    )


def test_web_search_gate_reads_server_side_tool_usage_details_without_citations():
    """
    Regression: xAI chat responses bridged from the Responses API only carry
    usage.server_side_tool_usage_details; a searched answer with no url_citation
    annotations must still be billed for its web search calls.
    """
    from litellm.llms.xai.cost_calculator import _DEFAULT_WEB_SEARCH_COST_PER_CALL
    from litellm.types.utils import Usage

    usage = Usage(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        server_side_tool_usage_details={"web_search_calls": 3},
    )
    response = ModelResponse(model="xai/grok-4.5")

    assert StandardBuiltInToolCostTracking.response_object_includes_web_search_call(
        response_object=response, usage=usage
    )
    assert not StandardBuiltInToolCostTracking.response_object_includes_web_search_call(
        response_object=response,
        usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )

    cost = StandardBuiltInToolCostTracking.get_cost_for_built_in_tools(
        model="xai/grok-4.5",
        response_object=response,
        usage=usage,
        custom_llm_provider="xai",
        standard_built_in_tools_params=None,
    )
    assert cost == 3 * _DEFAULT_WEB_SEARCH_COST_PER_CALL


_BEDROCK_MANTLE_WEB_SEARCH_MODELS = (
    "bedrock_mantle/openai.gpt-5.6-sol",
    "bedrock_mantle/openai.gpt-5.6-terra",
    "bedrock_mantle/openai.gpt-5.6-luna",
    "bedrock_mantle/openai.gpt-5.5",
    "bedrock_mantle/openai.gpt-5.4",
)

_BEDROCK_MANTLE_WEB_SEARCH_RATE = 0.012
