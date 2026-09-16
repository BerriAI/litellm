import os

import pytest

import litellm
from litellm.llms.gemini.cost_calculator import (
    cost_per_google_maps_grounding_request,
    cost_per_web_search_request,
)
from litellm.llms.gemini.image_edit.cost_calculator import (
    cost_calculator as gemini_image_edit_cost_calculator,
)
from litellm.llms.gemini.image_generation.cost_calculator import (
    cost_calculator as gemini_image_generation_cost_calculator,
)
from litellm.types.utils import (
    ImageObject,
    ImageResponse,
    ImageUsage,
    ImageUsageInputTokensDetails,
    PromptTokensDetailsWrapper,
    Usage,
)


def _make_usage(web_search_requests: int) -> Usage:
    return Usage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            web_search_requests=web_search_requests,
        ),
    )


def test_per_query_billing():
    """web_search_billing_unit=per_query charges per search query."""
    model_info = {
        "key": "gemini/gemini-3-flash-preview",
        "web_search_billing_unit": "per_query",
        "search_context_cost_per_query": {
            "search_context_size_medium": 0.014,
        },
    }
    cost = cost_per_web_search_request(usage=_make_usage(3), model_info=model_info)
    assert cost == pytest.approx(0.014 * 3)


def test_per_prompt_billing():
    """web_search_billing_unit=per_prompt (default) clamps to 1."""
    model_info = {
        "key": "gemini/gemini-2.5-flash",
        "search_context_cost_per_query": {
            "search_context_size_medium": 0.035,
        },
    }
    cost = cost_per_web_search_request(usage=_make_usage(3), model_info=model_info)
    assert cost == pytest.approx(0.035 * 1)


def test_default_billing_unit_is_per_prompt():
    """Without web_search_billing_unit, defaults to per_prompt (clamp to 1)."""
    model_info = {"key": "gemini/gemini-2.0-flash"}
    cost = cost_per_web_search_request(usage=_make_usage(2), model_info=model_info)
    assert cost == pytest.approx(0.035 * 1)


def test_zero_requests():
    """Zero web search requests should return zero cost."""
    model_info = {
        "key": "gemini/gemini-3-flash-preview",
        "web_search_billing_unit": "per_query",
    }
    cost = cost_per_web_search_request(usage=_make_usage(0), model_info=model_info)
    assert cost == 0.0


def test_no_usage_details():
    """Missing prompt_tokens_details should return zero cost."""
    model_info = {"key": "gemini/gemini-3-flash-preview"}
    usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    cost = cost_per_web_search_request(usage=usage, model_info=model_info)
    assert cost == 0.0


def _make_server_tool_use_usage(web_search_requests: int) -> Usage:
    from litellm.types.utils import ServerToolUse

    return Usage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        server_tool_use=ServerToolUse(web_search_requests=web_search_requests),
    )


def test_server_tool_use_fallback_per_query_billing():
    """Usage reconstructed from an Anthropic-format response carries the count in
    server_tool_use, not prompt_tokens_details; per_query billing prices each request."""
    model_info = {
        "key": "gemini/gemini-3-flash-preview",
        "web_search_billing_unit": "per_query",
        "search_context_cost_per_query": {
            "search_context_size_medium": 0.014,
        },
    }
    cost = cost_per_web_search_request(usage=_make_server_tool_use_usage(3), model_info=model_info)
    assert cost == pytest.approx(0.014 * 3)


def test_server_tool_use_fallback_per_prompt_clamps_to_one():
    """per_prompt billing clamps the server_tool_use count to one grounded prompt."""
    model_info = {
        "key": "gemini/gemini-2.5-flash",
        "search_context_cost_per_query": {
            "search_context_size_medium": 0.035,
        },
    }
    cost = cost_per_web_search_request(usage=_make_server_tool_use_usage(4), model_info=model_info)
    assert cost == pytest.approx(0.035 * 1)


def test_prompt_tokens_details_take_precedence_over_server_tool_use():
    """The native Gemini field wins when both counts are present."""
    from litellm.types.utils import ServerToolUse

    model_info = {
        "key": "gemini/gemini-3-flash-preview",
        "web_search_billing_unit": "per_query",
        "search_context_cost_per_query": {
            "search_context_size_medium": 0.014,
        },
    }
    usage = Usage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        prompt_tokens_details=PromptTokensDetailsWrapper(web_search_requests=2),
        server_tool_use=ServerToolUse(web_search_requests=5),
    )
    cost = cost_per_web_search_request(usage=usage, model_info=model_info)
    assert cost == pytest.approx(0.014 * 2)


def _make_maps_usage(google_maps_grounding_requests: int) -> Usage:
    return Usage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            google_maps_grounding_requests=google_maps_grounding_requests,
        ),
    )


def test_maps_per_query_billing():
    """web_search_billing_unit=per_query charges per Maps query."""
    model_info = {
        "key": "gemini/gemini-3.5-flash",
        "web_search_billing_unit": "per_query",
        "google_maps_grounding_cost_per_query": 0.014,
    }
    cost = cost_per_google_maps_grounding_request(usage=_make_maps_usage(3), model_info=model_info)
    assert cost == pytest.approx(0.014 * 3)


def test_maps_per_prompt_billing_clamps_to_one():
    """Without web_search_billing_unit, Maps grounding is one flat fee per grounded prompt."""
    model_info = {
        "key": "gemini/gemini-2.5-flash",
        "google_maps_grounding_cost_per_query": 0.025,
    }
    cost = cost_per_google_maps_grounding_request(usage=_make_maps_usage(3), model_info=model_info)
    assert cost == pytest.approx(0.025)


def test_maps_default_rate_per_query():
    """A per_query model missing the pricing key falls back to Google's $14/1K queries."""
    model_info = {"key": "gemini/gemini-3.9-flash", "web_search_billing_unit": "per_query"}
    cost = cost_per_google_maps_grounding_request(usage=_make_maps_usage(2), model_info=model_info)
    assert cost == pytest.approx(0.014 * 2)


def test_maps_default_rate_per_prompt():
    """A per_prompt model missing the pricing key falls back to Google's $25/1K grounded prompts."""
    model_info = {"key": "gemini/gemini-2.6-flash"}
    cost = cost_per_google_maps_grounding_request(usage=_make_maps_usage(2), model_info=model_info)
    assert cost == pytest.approx(0.025)


def test_maps_zero_requests():
    model_info = {"key": "gemini/gemini-3.5-flash", "web_search_billing_unit": "per_query"}
    assert cost_per_google_maps_grounding_request(usage=_make_maps_usage(0), model_info=model_info) == 0.0


def test_maps_no_usage_details():
    usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    model_info = {"key": "gemini/gemini-3.5-flash"}
    assert cost_per_google_maps_grounding_request(usage=usage, model_info=model_info) == 0.0


def test_gemini_image_edit_cost_prefers_token_usage_metadata(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    model = "gemini/gemini-3-pro-image-preview"
    model_info = litellm.get_model_info(model=model, custom_llm_provider="gemini")

    input_text_tokens = 20
    input_image_tokens = 1120
    output_image_tokens = 1120
    prompt_tokens = input_text_tokens + input_image_tokens
    image_response = ImageResponse(
        data=[ImageObject(b64_json="img1"), ImageObject(b64_json="img2")],
        usage=ImageUsage(
            input_tokens=prompt_tokens,
            input_tokens_details=ImageUsageInputTokensDetails(
                text_tokens=input_text_tokens,
                image_tokens=input_image_tokens,
            ),
            output_tokens=output_image_tokens,
            total_tokens=prompt_tokens + output_image_tokens,
        ),
    )

    cost = gemini_image_edit_cost_calculator(
        model=model,
        image_response=image_response,
    )

    expected_cost = (
        prompt_tokens * model_info["input_cost_per_token"]
        + output_image_tokens * model_info["output_cost_per_image_token"]
    )
    flat_image_cost = (
        len(image_response.data or []) * model_info["output_cost_per_image"]
    )
    assert round(cost, 10) == round(expected_cost, 10)
    assert cost != flat_image_cost


def test_gemini_image_edit_cost_uses_output_token_details(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    model = "gemini/gemini-3-pro-image-preview"
    model_info = litellm.get_model_info(model=model, custom_llm_provider="gemini")

    input_text_tokens = 20
    output_text_tokens = 213
    output_image_tokens = 1120
    output_tokens = output_text_tokens + output_image_tokens
    image_response = ImageResponse(
        data=[ImageObject(b64_json="img1")],
        usage=ImageUsage(
            input_tokens=input_text_tokens,
            input_tokens_details=ImageUsageInputTokensDetails(
                text_tokens=input_text_tokens,
                image_tokens=0,
            ),
            output_tokens=output_tokens,
            total_tokens=input_text_tokens + output_tokens,
            prompt_tokens=input_text_tokens,
            completion_tokens=output_tokens,
            prompt_tokens_details={
                "text_tokens": input_text_tokens,
                "image_tokens": 0,
            },
            completion_tokens_details={
                "text_tokens": output_text_tokens,
                "image_tokens": output_image_tokens,
            },
            output_tokens_details={
                "text_tokens": output_text_tokens,
                "image_tokens": output_image_tokens,
            },
        ),
    )

    cost = gemini_image_edit_cost_calculator(
        model=model,
        image_response=image_response,
    )

    expected_cost = (
        input_text_tokens * model_info["input_cost_per_token"]
        + output_text_tokens * model_info["output_cost_per_token"]
        + output_image_tokens * model_info["output_cost_per_image_token"]
    )
    all_output_as_image_cost = (
        input_text_tokens * model_info["input_cost_per_token"]
        + (output_text_tokens + output_image_tokens)
        * model_info["output_cost_per_image_token"]
    )
    assert round(cost, 10) == round(expected_cost, 10)
    assert cost != all_output_as_image_cost


def test_gemini_image_generation_cost_uses_output_token_details(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    model = "gemini/gemini-3-pro-image-preview"
    model_info = litellm.get_model_info(model=model, custom_llm_provider="gemini")

    input_text_tokens = 20
    output_text_tokens = 213
    output_image_tokens = 1120
    output_tokens = output_text_tokens + output_image_tokens
    image_response = ImageResponse(
        data=[ImageObject(b64_json="img1")],
        usage=ImageUsage(
            input_tokens=input_text_tokens,
            input_tokens_details=ImageUsageInputTokensDetails(
                text_tokens=input_text_tokens,
                image_tokens=0,
            ),
            output_tokens=output_tokens,
            total_tokens=input_text_tokens + output_tokens,
            prompt_tokens=input_text_tokens,
            completion_tokens=output_tokens,
            prompt_tokens_details={
                "text_tokens": input_text_tokens,
                "image_tokens": 0,
            },
            completion_tokens_details={
                "text_tokens": output_text_tokens,
                "image_tokens": output_image_tokens,
            },
            output_tokens_details={
                "text_tokens": output_text_tokens,
                "image_tokens": output_image_tokens,
            },
        ),
    )

    cost = gemini_image_generation_cost_calculator(
        model=model,
        image_response=image_response,
    )

    expected_cost = (
        input_text_tokens * model_info["input_cost_per_token"]
        + output_text_tokens * model_info["output_cost_per_token"]
        + output_image_tokens * model_info["output_cost_per_image_token"]
    )
    all_output_as_image_cost = (
        input_text_tokens * model_info["input_cost_per_token"]
        + (output_text_tokens + output_image_tokens)
        * model_info["output_cost_per_image_token"]
    )
    assert round(cost, 10) == round(expected_cost, 10)
    assert cost != all_output_as_image_cost


def test_gemini_image_edit_cost_falls_back_to_flat_image_pricing(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    model = "gemini/gemini-3-pro-image-preview"
    model_info = litellm.get_model_info(model=model, custom_llm_provider="gemini")
    image_response = ImageResponse(
        data=[ImageObject(b64_json="img1"), ImageObject(b64_json="img2")]
    )

    cost = gemini_image_edit_cost_calculator(
        model=model,
        image_response=image_response,
    )

    assert cost == len(image_response.data or []) * model_info["output_cost_per_image"]


def _image_response_with_web_search(web_search_requests):
    usage = ImageUsage(
        input_tokens=20,
        input_tokens_details=ImageUsageInputTokensDetails(
            text_tokens=20,
            image_tokens=0,
        ),
        output_tokens=1120,
        total_tokens=1140,
    )
    if web_search_requests is not None:
        usage.web_search_requests = web_search_requests
    return ImageResponse(data=[ImageObject(b64_json="img1")], usage=usage)


def test_gemini_image_generation_cost_adds_web_search_grounding(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    model = "gemini/gemini-3-pro-image-preview"
    model_info = litellm.get_model_info(model=model, custom_llm_provider="gemini")

    grounded = gemini_image_generation_cost_calculator(
        model=model,
        image_response=_image_response_with_web_search(2),
    )
    ungrounded = gemini_image_generation_cost_calculator(
        model=model,
        image_response=_image_response_with_web_search(None),
    )

    expected_web_search_cost = cost_per_web_search_request(
        usage=_make_usage(2), model_info=model_info
    )
    assert expected_web_search_cost > 0
    assert round(grounded - ungrounded, 10) == round(expected_web_search_cost, 10)


def test_gemini_image_generation_cost_no_web_search_when_absent(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    model = "gemini/gemini-3-pro-image-preview"

    cost_zero = gemini_image_generation_cost_calculator(
        model=model,
        image_response=_image_response_with_web_search(0),
    )
    cost_none = gemini_image_generation_cost_calculator(
        model=model,
        image_response=_image_response_with_web_search(None),
    )

    assert cost_zero == cost_none


@pytest.mark.parametrize(
    "traffic_type, expected_service_tier",
    [
        ("ON_DEMAND", None),
        ("ON_DEMAND_PRIORITY", "priority"),
        ("FLEX", "flex"),
        ("BATCH", "flex"),
        # Vertex AI reports flex/shared-capacity traffic as ON_DEMAND_FLEX.
        ("ON_DEMAND_FLEX", "flex"),
        # trafficType is matched case-insensitively.
        ("on_demand_flex", "flex"),
        (None, None),
        ("SOMETHING_UNKNOWN", None),
    ],
)
def test_map_traffic_type_to_service_tier(
    traffic_type: str | None, expected_service_tier: str | None
):
    """
    Gemini/Vertex usageMetadata.trafficType maps to the LiteLLM service_tier
    that selects flex/priority cost keys. ON_DEMAND_FLEX (Vertex's flex opt-in
    value) must map to "flex" so flex-tier requests are not billed as standard.
    """
    from litellm.cost_calculator import _map_traffic_type_to_service_tier

    assert (
        _map_traffic_type_to_service_tier(traffic_type) == expected_service_tier
    )


@pytest.mark.parametrize(
    "model,custom_llm_provider,expected_cache_read_cost",
    [
        ("gemini/gemini-flash-latest", "gemini", 3e-08),
        ("gemini/gemini-flash-lite-latest", "gemini", 1e-08),
        ("gemini/gemini-2.5-flash-preview-09-2025", "gemini", 3e-08),
        ("gemini/gemini-2.5-flash-lite-preview-06-17", "gemini", 1e-08),
        ("vertex_ai/gemini-2.5-flash-preview-09-2025", "vertex_ai", 3e-08),
        ("vertex_ai/gemini-2.5-flash-lite-preview-06-17", "vertex_ai", 1e-08),
    ],
)
def test_flash_alias_cache_read_is_ten_percent_of_input(
    monkeypatch, model, custom_llm_provider, expected_cache_read_cost
):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))

    model_info = litellm.get_model_info(
        model=model, custom_llm_provider=custom_llm_provider
    )

    assert model_info["cache_read_input_token_cost"] == expected_cache_read_cost
    assert model_info["cache_read_input_token_cost"] == pytest.approx(
        0.10 * model_info["input_cost_per_token"]
    )


@pytest.mark.parametrize(
    "prefixed,bare",
    [
        ("gemini/gemini-flash-latest", "gemini-flash-latest"),
        ("gemini/gemini-flash-lite-latest", "gemini-flash-lite-latest"),
    ],
)
def test_flash_latest_alias_spellings_price_identically(monkeypatch, prefixed, bare):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))

    prefixed_entry = litellm.model_cost[prefixed]
    bare_entry = litellm.model_cost[bare]

    for cost_key in (
        "input_cost_per_token",
        "output_cost_per_token",
        "cache_read_input_token_cost",
    ):
        assert prefixed_entry[cost_key] == bare_entry[cost_key]
