import os
import sys

import pytest

import litellm
from litellm.litellm_core_utils.llm_cost_calc.tool_call_cost_tracking import (
    StandardBuiltInToolCostTracking,
)
from litellm.types.llms.openai import FileSearchTool, WebSearchOptions
from litellm.types.utils import ModelResponse, StandardBuiltInToolsParams

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


@pytest.fixture
def local_model_cost_map(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))


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
    assert StandardBuiltInToolCostTracking.response_object_includes_web_search_call(
        response_object=None, usage=usage
    )


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

    per_query_cost = litellm.get_model_info(model)["search_context_cost_per_query"][
        "search_context_size_medium"
    ]
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

    per_query_cost = litellm.get_model_info(model)["search_context_cost_per_query"][
        "search_context_size_medium"
    ]
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
                    content="Test response with grounding", role="assistant"
                ),
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
            text_tokens=11, web_search_requests=1  # This should trigger grounding cost
        ),
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
    # Force use of local model cost map for CI/CD consistency
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

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
    assert (
        cost >= web_search_cost
    ), f"completion_cost ({cost}) should include web search cost ({web_search_cost})"


@pytest.mark.parametrize(
    "model",
    [
        "vertex_ai/gemini-3.1-flash-lite",  # resolves directly via get_model_info
        "gemini/gemini-3.1-flash-lite",  # provider-prefixed, resolves via model_cost fallback
    ],
)
def test_gemini_3x_web_search_billed_per_query(model, local_model_cost_map):
    """
    Gemini 3.x bills web search per individual query (web_search_billing_unit == "per_query"),
    so N searches cost N * $0.014.

    Regression for the bug where the billing unit was dropped between the pricing JSON and the
    cost calculator: the field was missing from the ModelInfoBase TypedDict and from the
    ModelInfoBase(...) constructor in _get_model_info_helper, so get_model_info returned it as
    None and cost_per_web_search_request fell back to the per_prompt clamp, collapsing N queries
    to a single charge. The "gemini/..." case additionally covers response_cost_calculator
    resolving a provider-prefixed model name that get_model_info cannot map under vertex_ai.
    """
    from litellm.types.utils import PromptTokensDetailsWrapper, Usage

    web_search_requests = 2
    model_info = litellm.get_model_info(model)
    assert model_info["web_search_billing_unit"] == "per_query"
    per_query_cost = model_info["search_context_cost_per_query"][
        "search_context_size_medium"
    ]
    expected_cost = per_query_cost * web_search_requests

    usage = Usage(
        prompt_tokens=11,
        completion_tokens=100,
        total_tokens=111,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            text_tokens=11, web_search_requests=web_search_requests
        ),
    )

    cost = StandardBuiltInToolCostTracking.get_cost_for_built_in_tools(
        model=model,
        usage=usage,
        response_object=None,
        custom_llm_provider="vertex_ai",
        standard_built_in_tools_params=None,
    )

    assert cost == pytest.approx(expected_cost), (
        f"Expected {web_search_requests} x ${per_query_cost} = ${expected_cost} "
        f"per_query search fee, got ${cost}"
    )


def test_gemini_2x_web_search_still_billed_per_prompt(local_model_cost_map):
    """
    Gemini 2.x bills web search per grounded prompt: multiple internal queries are one flat
    $0.035 fee. Guards the per_prompt clamp against the per_query plumbing, which makes
    web_search_billing_unit always present on the resolved ModelInfo (None for 2.x), so the
    clamp must treat a None billing unit as per_prompt rather than skipping the clamp.
    """
    from litellm.types.utils import PromptTokensDetailsWrapper, Usage

    model = "vertex_ai/gemini-2.5-flash"
    model_info = litellm.get_model_info(model)
    assert not model_info.get("web_search_billing_unit")
    expected_cost = model_info["search_context_cost_per_query"][
        "search_context_size_medium"
    ]

    usage = Usage(
        prompt_tokens=11,
        completion_tokens=100,
        total_tokens=111,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            text_tokens=11, web_search_requests=2
        ),
    )

    cost = StandardBuiltInToolCostTracking.get_cost_for_built_in_tools(
        model=model,
        usage=usage,
        response_object=None,
        custom_llm_provider="vertex_ai",
        standard_built_in_tools_params=None,
    )

    assert cost == pytest.approx(expected_cost), (
        f"Expected flat ${expected_cost} per_prompt search fee (2 queries clamped to 1), "
        f"got ${cost}"
    )


def test_web_search_provider_prefix_fallback_does_not_misprice_non_gemini_model(
    local_model_cost_map,
):
    """
    Regression for the provider-prefix fallback in _handle_web_search_cost. When the initial
    get_model_info lookup fails for a "/"-containing model, the retry re-resolves model_info from
    the prefix and must adopt that prefix's provider for routing. Otherwise an unrelated model
    (here OpenRouter, which carries no web search pricing) is re-resolved but still routed through
    the request's vertex_ai Gemini calculator, which charges its $0.035 per_prompt default for a
    model that should cost nothing for web search.
    """
    from litellm.types.utils import PromptTokensDetailsWrapper, Usage

    model = "openrouter/google/gemini-3.1-flash-lite"
    model_info = litellm.get_model_info(model)
    assert model_info["litellm_provider"] == "openrouter"
    assert not model_info.get("search_context_cost_per_query")

    usage = Usage(
        prompt_tokens=11,
        completion_tokens=100,
        total_tokens=111,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            text_tokens=11, web_search_requests=2
        ),
    )

    cost = StandardBuiltInToolCostTracking.get_cost_for_built_in_tools(
        model=model,
        usage=usage,
        response_object=None,
        custom_llm_provider="vertex_ai",
        standard_built_in_tools_params=None,
    )

    assert cost == 0.0, (
        "A non-Gemini provider-prefixed model with no web search pricing must not be charged "
        f"the vertex_ai per_prompt default via the prefix fallback, got ${cost}"
    )


# Note: File search integration test removed due to complex annotation detection logic
# The unit tests in test_azure_assistant_cost_tracking.py provide comprehensive coverage


def _responses_api_response(output, model="gpt-5.5", input_tokens=21384, output_tokens=845):
    """Build a minimal Responses API response carrying the given output items."""
    from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse

    return ResponsesAPIResponse(
        id="resp_1",
        created_at=0,
        model=model,
        output=output,
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        usage=ResponseAPIUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )


def _web_search_call(action_type, status="completed", index=0):
    action = {"type": action_type}
    if action_type == "search":
        action["query"] = f"query {index}"
    return {
        "id": f"ws_{index}",
        "type": "web_search_call",
        "status": status,
        "action": action,
    }


def _assistant_message():
    return {
        "id": "msg_1",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "hi", "annotations": []}],
    }


OPENAI_WEB_SEARCH_LIST_PRICE_PER_CALL = 10.0 / 1000
AZURE_WEB_SEARCH_LIST_PRICE_PER_CALL = 14.0 / 1000


def test_openai_responses_web_search_charges_every_billable_search(local_model_cost_map):
    """
    An agentic Responses API turn runs many searches in one request, and only `search`
    actions that completed carry the per-call fee. Every billable search must be charged,
    not just the first one, and the free navigation actions must not be.
    """
    billable_searches = 3
    response = _responses_api_response(
        output=[
            *(_web_search_call("search", index=i) for i in range(billable_searches)),
            _web_search_call("open_page", index=8),
            _web_search_call("find_in_page", index=9),
            _web_search_call("search", status="failed", index=10),
            _assistant_message(),
        ]
    )

    cost = StandardBuiltInToolCostTracking.get_cost_for_built_in_tools(
        model="gpt-5.5",
        usage=None,
        response_object=response,
        custom_llm_provider="openai",
        standard_built_in_tools_params=None,
    )

    assert cost == pytest.approx(billable_searches * OPENAI_WEB_SEARCH_LIST_PRICE_PER_CALL)


def test_azure_responses_web_search_uses_bing_grounding_price(local_model_cost_map):
    """
    Azure OpenAI bills the web search tool through Grounding with Bing Search, which is
    priced above OpenAI's own per-call fee. The Azure request must not be charged the
    OpenAI rate.
    """
    billable_searches = 2
    output = [
        *(_web_search_call("search", index=i) for i in range(billable_searches)),
        _assistant_message(),
    ]

    azure_cost = StandardBuiltInToolCostTracking.get_cost_for_built_in_tools(
        model="azure/gpt-5.5",
        usage=None,
        response_object=_responses_api_response(output, model="gpt-5.5"),
        custom_llm_provider="azure",
        standard_built_in_tools_params=None,
    )

    assert azure_cost == pytest.approx(billable_searches * AZURE_WEB_SEARCH_LIST_PRICE_PER_CALL)
    assert azure_cost > billable_searches * OPENAI_WEB_SEARCH_LIST_PRICE_PER_CALL


def test_openai_responses_web_search_prefers_model_pricing_over_list_price(local_model_cost_map):
    """
    A model priced in the cost map wins over the provider list price, so per-deployment
    and custom pricing keeps working while models absent from the map are still charged.
    """
    model = "gpt-4o-search-preview"
    per_query_cost = litellm.get_model_info(model)["search_context_cost_per_query"]["search_context_size_medium"]
    assert per_query_cost != OPENAI_WEB_SEARCH_LIST_PRICE_PER_CALL

    cost = StandardBuiltInToolCostTracking.get_cost_for_built_in_tools(
        model=model,
        usage=None,
        response_object=_responses_api_response(
            [_web_search_call("search", index=0), _web_search_call("search", index=1)],
            model=model,
        ),
        custom_llm_provider="openai",
        standard_built_in_tools_params=None,
    )

    assert cost == pytest.approx(2 * per_query_cost)


def test_openai_responses_web_search_charges_nothing_without_a_billable_search(local_model_cost_map):
    """
    Reading pages inside an earlier turn's search results is free. A response whose only
    web_search_call items are navigation actions must not be charged a search fee.
    """
    response = _responses_api_response(
        [
            _web_search_call("open_page", index=0),
            _web_search_call("find_in_page", index=1),
            _assistant_message(),
        ]
    )

    cost = StandardBuiltInToolCostTracking.get_cost_for_built_in_tools(
        model="gpt-5.5",
        usage=None,
        response_object=response,
        custom_llm_provider="openai",
        standard_built_in_tools_params=None,
    )

    assert cost == 0.0


def test_openai_chat_completion_web_search_still_priced_by_search_context_size(local_model_cost_map):
    """
    Chat completions report no per-search count, only url_citation annotations, so they
    stay on search_context_size pricing. Counting Responses API items must not disturb it.
    """
    from litellm.types.utils import Choices, Message, ModelResponse

    model = "gpt-4o-search-preview"
    response = ModelResponse(
        choices=[
            Choices(
                message=Message(
                    role="assistant",
                    content="hi",
                    annotations=[{"type": "url_citation", "url_citation": {"url": "https://x"}}],
                )
            )
        ]
    )

    cost = StandardBuiltInToolCostTracking.get_cost_for_built_in_tools(
        model=model,
        usage=None,
        response_object=response,
        custom_llm_provider="openai",
        standard_built_in_tools_params=StandardBuiltInToolsParams(
            web_search_options=WebSearchOptions(search_context_size="high")
        ),
    )

    assert cost == litellm.get_model_info(model)["search_context_cost_per_query"]["search_context_size_high"]


def test_completion_cost_includes_openai_web_search_fee(local_model_cost_map):
    """
    End-to-end: the response cost a caller sees for a Responses API request must be the
    token cost plus one fee per billable search, not the token cost alone.
    """
    model = "gpt-5.5"
    input_tokens = 21384
    output_tokens = 845
    billable_searches = 4
    response = _responses_api_response(
        output=[
            *(_web_search_call("search", index=i) for i in range(billable_searches)),
            _assistant_message(),
        ],
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    model_info = litellm.get_model_info(model)
    expected_token_cost = (
        input_tokens * model_info["input_cost_per_token"] + output_tokens * model_info["output_cost_per_token"]
    )

    cost = litellm.completion_cost(completion_response=response, model=model, custom_llm_provider="openai")

    assert cost == pytest.approx(expected_token_cost + billable_searches * OPENAI_WEB_SEARCH_LIST_PRICE_PER_CALL)


def test_openai_responses_web_search_honors_explicit_zero_price(local_model_cost_map):
    """
    A deployment is free to price web search at zero, e.g. when the fee is covered elsewhere.
    An explicit 0.0 in the cost map must be charged as zero rather than falling back to the
    provider list price, which would inflate response costs, spend records and budgets.
    """
    model = "openai-zero-priced-web-search-model"
    litellm.register_model(
        {
            model: {
                "litellm_provider": "openai",
                "mode": "responses",
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
                "search_context_cost_per_query": {
                    "search_context_size_low": 0.0,
                    "search_context_size_medium": 0.0,
                    "search_context_size_high": 0.0,
                },
            }
        }
    )

    cost = StandardBuiltInToolCostTracking.get_cost_for_built_in_tools(
        model=model,
        usage=None,
        response_object=_responses_api_response(
            [_web_search_call("search", index=0), _web_search_call("search", index=1)],
            model=model,
        ),
        custom_llm_provider="openai",
        standard_built_in_tools_params=None,
    )

    assert cost == 0.0, f"an explicit zero web search price must not fall back to the list price, got ${cost}"


def test_web_search_count_on_usage_wins_over_the_raw_response_count(local_model_cost_map):
    """
    When the provider reported a count in usage, that count is authoritative and the raw
    response is not re-read. Otherwise a response that also carries a count would silently
    override reconciled usage and bill a different number of searches.
    """
    from litellm.types.utils import ServerToolUse, Usage

    model = "claude-3-7-sonnet-20250219"
    raw_response = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": "hi"}],
        "usage": {"input_tokens": 100, "output_tokens": 50, "server_tool_use": {"web_search_requests": 5}},
    }
    usage = Usage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        server_tool_use=ServerToolUse(web_search_requests=1),
    )

    cost = StandardBuiltInToolCostTracking.get_cost_for_built_in_tools(
        model=model,
        usage=usage,
        response_object=raw_response,
        custom_llm_provider="anthropic",
        standard_built_in_tools_params=None,
    )

    per_query_cost = litellm.get_model_info(model)["search_context_cost_per_query"]["search_context_size_medium"]
    assert cost == pytest.approx(per_query_cost), f"expected 1 search from usage, got ${cost}"
