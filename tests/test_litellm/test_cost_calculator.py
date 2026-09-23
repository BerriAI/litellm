import datetime
import time
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Final, cast

import pytest
from pydantic import BaseModel

import litellm
from litellm.cost_calculator import (
    BaseTokenUsageProcessor,
    RealtimeAPITokenUsageProcessor,
    ResponsesWebSocketTokenUsageProcessor,
    completion_cost,
    cost_per_token,
    handle_realtime_stream_cost_calculation,
    response_cost_calculator,
)
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.base_llm.ocr.transformation import OCRPage, OCRResponse, OCRUsageInfo
from litellm.types.llms.base import CachedTokensDetails
from litellm.types.llms.openai import OpenAIRealtimeStreamList, ResponseAPIUsage, ResponsesAPIResponse
from litellm.types.rerank import RerankResponse
from litellm.types.utils import (
    CacheCreationTokenDetails,
    CallTypes,
    Choices,
    ImageObject,
    ImageResponse,
    ImageUsage,
    ImageUsageInputTokensDetails,
    LiteLLMRealtimeStreamLoggingObject,
    Message,
    ModelInfo,
    ModelResponse,
    PromptTokensDetailsWrapper,
    Usage,
)
from litellm.types.videos.main import VideoObject
from litellm.utils import supports_prompt_caching


@pytest.fixture
def _local_model_cost_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))


def test_cost_per_token_duplicate_openai_prefix_matches_model_cost(monkeypatch):
    """
    Router/proxy configs may use deployment ids like openai/openai/<model>. Cost lookup must
    resolve to model_prices keys (e.g. gpt-5.5), not fail or multiply prefixes.
    """
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))

    prompt_usd, completion_usd = cost_per_token(
        model="openai/openai/gpt-5.5",
        prompt_tokens=100,
        completion_tokens=50,
        custom_llm_provider="openai",
    )

    assert prompt_usd + completion_usd > 0


def test_cost_per_token_non_string_model_does_not_hang():
    """
    The provider-prefix dedup loop must not spin forever when `model` is a
    non-string object (e.g. a MagicMock from a mocked transport). It should
    return or raise promptly instead of looping on a truthy `.startswith()`.
    """
    import threading
    from unittest.mock import MagicMock

    result: dict = {}

    def _run():
        try:
            cost_per_token(
                model=MagicMock(),
                prompt_tokens=10,
                completion_tokens=5,
                custom_llm_provider="anthropic",
            )
            result["status"] = "returned"
        except Exception:
            result["status"] = "raised"

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout=10)

    assert not worker.is_alive(), "cost_per_token hung on a non-string model"
    assert result.get("status") in ("returned", "raised")


def test_completion_cost_uses_response_model_for_dynamic_routing(_local_model_cost_map):
    """
    Test that completion_cost uses the model from the response object
    when the input model (e.g., azure-model-router) is not in model_cost.
    This supports Azure Model Router and similar dynamic routing scenarios.
    """

    # Simulate Azure Model Router: input is generic router, response has actual model
    response = ModelResponse(
        id="test-id",
        model="azure_ai/gpt-4o-2024-08-06",  # Response contains actual model used
        choices=[],
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )

    # Should calculate cost using the response model, not the input model
    cost = completion_cost(
        completion_response=response,
        model="azure_ai/azure-model-router",  # Input model doesn't exist in model_cost
        custom_llm_provider="azure_ai",
    )

    assert cost > 0, "Cost should be calculated using response model"


def test_completion_cost_strips_dated_azure_snapshot_model(_local_model_cost_map: None) -> None:
    dated_response = ModelResponse(
        model="gpt-5.6-luna-2099-01-01",
        choices=[Choices(index=0, message=Message(role="assistant", content="hi"), finish_reason="stop")],
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )
    dated_response._hidden_params = {"custom_llm_provider": "azure"}

    undated_response = ModelResponse(
        model="gpt-5.6-luna",
        choices=[Choices(index=0, message=Message(role="assistant", content="hi"), finish_reason="stop")],
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )
    undated_response._hidden_params = {"custom_llm_provider": "azure"}

    dated_cost = litellm.completion_cost(completion_response=dated_response)
    undated_cost = litellm.completion_cost(completion_response=undated_response)

    assert dated_cost == undated_cost
    assert dated_cost > 0


def test_cost_calculator_with_response_cost_in_additional_headers():
    class MockResponse(BaseModel):
        _hidden_params = {"additional_headers": {"llm_provider-x-litellm-response-cost": 1000}}

    result = response_cost_calculator(
        response_object=MockResponse(),
        model="",
        custom_llm_provider=None,
        call_type="",
        optional_params={},
        cache_hit=None,
        base_model=None,
    )

    assert result == 1000






def test_realtime_stream_combines_text_and_audio_token_details():
    """Realtime response.done usage with input_token_details / output_token_details."""
    from litellm.cost_calculator import RealtimeAPITokenUsageProcessor

    results: OpenAIRealtimeStreamList = [
        {"type": "session.created", "session": {"model": "gpt-4o-realtime-preview"}},
        {
            "type": "response.done",
            "response": {
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "total_tokens": 30,
                    "input_token_details": {"text_tokens": 8, "audio_tokens": 2},
                    "output_token_details": {"text_tokens": 12, "audio_tokens": 8},
                }
            },
        },
        {
            "type": "response.done",
            "response": {
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 15,
                    "total_tokens": 20,
                    "input_token_details": {"text_tokens": 3, "audio_tokens": 2},
                    "output_token_details": {"text_tokens": 5, "audio_tokens": 10},
                }
            },
        },
    ]

    combined = RealtimeAPITokenUsageProcessor.collect_and_combine_usage_from_realtime_stream_results(
        results=results,
    )

    assert combined.prompt_tokens_details is not None
    assert combined.prompt_tokens_details.text_tokens == 11
    assert combined.prompt_tokens_details.audio_tokens == 4

    assert combined.completion_tokens_details is not None
    assert combined.completion_tokens_details.text_tokens == 17
    assert combined.completion_tokens_details.audio_tokens == 18


def test_realtime_logging_object_allows_null_transcript_in_conversation_item_added():
    results: OpenAIRealtimeStreamList = [
        {
            "type": "conversation.item.added",
            "event_id": "event_added",
            "item": {
                "id": "item_123",
                "type": "message",
                "role": "assistant",
                "status": "in_progress",
                "content": [{"type": "audio", "transcript": None}],
            },
        },
        {
            "type": "response.done",
            "event_id": "event_done",
            "response": {
                "id": "resp_123",
                "object": "realtime.response",
                "status": "completed",
                "usage": {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
            },
        },
    ]

    usage = RealtimeAPITokenUsageProcessor.collect_and_combine_usage_from_realtime_stream_results(results=results)
    logging_result = RealtimeAPITokenUsageProcessor.create_logging_realtime_object(
        usage=usage,
        results=results,
    )
    assert logging_result.usage.total_tokens == 18
    assert logging_result.results[0]["item"]["content"][0]["transcript"] is None
    assert logging_result.results[0]["item"]["content"][0]["transcript"] is None


def test_realtime_logging_object_does_not_validate_unknown_event_types():
    """
    A realtime session emits events outside the OpenAIRealtimeEvents union (e.g.
    rate_limits.updated, response.function_call_arguments.delta). Building the
    logging object must not revalidate every event against the union; doing so
    produces thousands of Pydantic ValidationErrors per session, blocks the event
    loop, and the raised error discards the session's usage. The events must
    survive verbatim, the combined usage must be preserved, and serialization
    must stay clean.
    """
    import warnings

    results: OpenAIRealtimeStreamList = [
        {"type": "session.created", "event_id": "ev0", "session": {"id": "s"}},
    ]
    for i in range(50):
        results += [
            {
                "type": "rate_limits.updated",
                "event_id": f"rl{i}",
                "rate_limits": [{"name": "requests", "limit": 1000, "remaining": 900}],
            },
            {
                "type": "response.function_call_arguments.delta",
                "event_id": f"fc{i}",
                "delta": "{}",
            },
            {
                "type": "response.done",
                "event_id": f"rd{i}",
                "response": {
                    "usage": {
                        "input_tokens": 4,
                        "output_tokens": 6,
                        "total_tokens": 10,
                    }
                },
            },
        ]

    usage = RealtimeAPITokenUsageProcessor.collect_and_combine_usage_from_realtime_stream_results(results=results)
    # On unfixed code this raises pydantic ValidationError instead of returning.
    logging_result = RealtimeAPITokenUsageProcessor.create_logging_realtime_object(
        usage=usage,
        results=results,
    )

    assert logging_result.usage.total_tokens == 500
    assert len(logging_result.results) == len(results)
    unknown_types = {
        r["type"]
        for r in logging_result.results
        if r["type"] in ("rate_limits.updated", "response.function_call_arguments.delta")
    }
    assert unknown_types == {
        "rate_limits.updated",
        "response.function_call_arguments.delta",
    }

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dumped = logging_result.model_dump()
    assert len(dumped["results"]) == len(results)


def test_realtime_transcription_no_completed_events_is_zero(monkeypatch):
    """A realtime stream without transcription completed events adds no extra cost."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))

    from litellm.cost_calculator import handle_realtime_transcription_cost_calculation

    results: OpenAIRealtimeStreamList = [
        {"type": "session.created", "session": {"model": "gpt-realtime-whisper"}},
        {"type": "response.done", "response": {"usage": {}}},
    ]
    assert (
        handle_realtime_transcription_cost_calculation(
            results=results,
            custom_llm_provider="openai",
            litellm_model_name="gpt-realtime-whisper",
        )
        == 0.0
    )


def test_transcription_usage_cost_returns_zero_for_unknown_type():
    """An unrecognized usage type yields 0 (safe fallback, no exception)."""
    from litellm.cost_calculator import _transcription_usage_cost

    assert _transcription_usage_cost({"type": "future_billing_type"}, {}) == 0.0
    assert _transcription_usage_cost({}, {}) == 0.0


def test_get_transcription_model_falls_back_to_session_model(monkeypatch):
    """session.model is used when transcription-specific model fields are absent."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))

    from litellm.cost_calculator import _get_transcription_model_name_from_results

    results: OpenAIRealtimeStreamList = [
        {"type": "session.created", "session": {"model": "gpt-realtime-whisper"}},
    ]
    assert _get_transcription_model_name_from_results(results) == "gpt-realtime-whisper"

    from litellm import Router

    router = Router(
        model_list=[
            {
                "model_name": "prod/claude-3-5-sonnet-20240620",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-5-20250929",
                    "api_key": "test_api_key",
                },
                "model_info": {
                    "id": "my-unique-model-id",
                    "input_cost_per_token": 0.000006,
                    "output_cost_per_token": 0.00003,
                    "cache_creation_input_token_cost": 0.0000075,
                    "cache_read_input_token_cost": 0.0000006,
                },
            },
            {
                "model_name": "claude-3-5-sonnet-20240620",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-5-20250929",
                    "api_key": "test_api_key",
                },
                "model_info": {
                    "input_cost_per_token": 100,
                    "output_cost_per_token": 200,
                },
            },
        ]
    )

    result = router.completion(
        model="claude-3-5-sonnet-20240620",
        messages=[{"role": "user", "content": "Hello, world!"}],
        mock_response=True,
    )

    result_2 = router.completion(
        model="prod/claude-3-5-sonnet-20240620",
        messages=[{"role": "user", "content": "Hello, world!"}],
        mock_response=True,
    )

    assert result._hidden_params["response_cost"] > result_2._hidden_params["response_cost"]

    model_info = router.get_deployment_model_info(
        model_id="my-unique-model-id", model_name="anthropic/claude-sonnet-4-5-20250929"
    )
    assert model_info is not None
    assert model_info["input_cost_per_token"] == 0.000006
    assert model_info["output_cost_per_token"] == 0.00003
    assert model_info["cache_creation_input_token_cost"] == 0.0000075
    assert model_info["cache_read_input_token_cost"] == 0.0000006


def test_custom_pricing_cost_calc_uses_router_model_id_from_litellm_metadata():
    """When custom pricing is in litellm_metadata.model_info,
    use_custom_pricing_for_model should return True and
    _select_model_name_for_cost_calc should use router_model_id.

    This tests the full chain that was broken for /messages and /responses
    endpoints. Regression test for #23185.
    """
    from litellm.cost_calculator import _select_model_name_for_cost_calc
    from litellm.litellm_core_utils.litellm_logging import use_custom_pricing_for_model

    custom_model_id = "claude-sonnet-4-custom-pricing-test"
    custom_pricing_info = {
        "input_cost_per_token": 0.0003,
        "output_cost_per_token": 0.0015,
        "max_tokens": 8192,
        "litellm_provider": "anthropic",
    }
    litellm.register_model(model_cost={custom_model_id: custom_pricing_info})

    litellm_params = {
        "litellm_metadata": {
            "model_info": {
                "id": custom_model_id,
                "input_cost_per_token": 0.0003,
                "output_cost_per_token": 0.0015,
            },
        },
    }

    custom_pricing = use_custom_pricing_for_model(litellm_params)
    assert custom_pricing is True

    # _select_model_name_for_cost_calc appends provider prefix to the
    # selected router_model_id, so the result is "anthropic/<model_id>"
    selected_model = _select_model_name_for_cost_calc(
        model="anthropic/claude-sonnet-4-20250514",
        completion_response=None,
        custom_pricing=custom_pricing,
        custom_llm_provider="anthropic",
        router_model_id=custom_model_id,
    )
    assert selected_model is not None
    assert custom_model_id in selected_model

    # Without custom_pricing, the router_model_id is NOT selected
    selected_model_no_custom = _select_model_name_for_cost_calc(
        model="anthropic/claude-sonnet-4-20250514",
        completion_response=None,
        custom_pricing=False,
        custom_llm_provider="anthropic",
        router_model_id=custom_model_id,
    )
    assert custom_model_id not in (selected_model_no_custom or "")


def test_per_request_custom_pricing_with_router():
    """When custom pricing is passed as per-request kwargs (not in model_list),
    _select_model_name_for_cost_calc should fall back to the model name
    (where register_model stored the pricing) instead of the router_model_id
    (which has no pricing data).

    Regression test for the bug where response._hidden_params["response_cost"]
    returned 0.0 for per-request custom pricing via Router.
    """
    from litellm import Router
    from litellm.cost_calculator import _select_model_name_for_cost_calc

    router = Router(
        model_list=[
            {
                "model_name": "openai/gpt-3.5-turbo",
                "litellm_params": {
                    "model": "openai/gpt-3.5-turbo",
                    "api_key": "test_api_key",
                },
            },
        ]
    )

    # Get the deployment's model_id (hash) that the router registered
    deployment = router.model_list[0]
    router_model_id = deployment["model_info"]["id"]

    # The router registered this hash in model_cost but without custom pricing
    assert router_model_id in litellm.model_cost
    entry = litellm.model_cost[router_model_id]
    # No custom pricing was set in model_list, so these should be None
    assert entry.get("input_cost_per_token") is None

    # Now simulate what completion() does: register custom pricing under the model name
    litellm.register_model(
        {
            "openai/gpt-3.5-turbo": {
                "input_cost_per_token": 2.0,
                "output_cost_per_token": 2.0,
                "litellm_provider": "openai",
            }
        }
    )

    # _select_model_name_for_cost_calc should pick the model name (which has pricing),
    # NOT the router_model_id (which has no pricing)
    selected = _select_model_name_for_cost_calc(
        model="openai/gpt-3.5-turbo",
        completion_response=None,
        custom_pricing=True,
        custom_llm_provider="openai",
        router_model_id=router_model_id,
    )
    assert selected is not None
    assert router_model_id not in selected
    assert "gpt-3.5-turbo" in selected


def test_tiered_pricing_only_deployment_selects_router_model_id():
    """A deployment priced solely via ``tiered_pricing`` (no flat
    input/output cost) must resolve cost against its ``router_model_id``
    entry, which holds the tiered table, instead of the shared backend alias
    that has custom pricing fields stripped. Regression for tier-only models
    (e.g. dashscope/qwen3.7-plus) being billed as free.
    """
    from litellm import Router
    from litellm.cost_calculator import _select_model_name_for_cost_calc

    router = Router(
        model_list=[
            {
                "model_name": "qwen-tier-only",
                "litellm_params": {
                    "model": "dashscope/qwen-tier-only-test",
                    "api_key": "sk-fake",
                },
                "model_info": {
                    "tiered_pricing": [
                        {
                            "input_cost_per_token": 4e-07,
                            "output_cost_per_token": 1.6e-06,
                            "range": [0, 256000],
                        },
                    ],
                },
            },
        ]
    )
    router_model_id = router.model_list[0]["model_info"]["id"]

    entry = litellm.model_cost[router_model_id]
    assert entry.get("input_cost_per_token") is None
    assert entry.get("tiered_pricing") is not None
    # The stripped shared alias must not carry tiered pricing.
    assert litellm.model_cost["dashscope/qwen-tier-only-test"].get("tiered_pricing") is None

    selected = _select_model_name_for_cost_calc(
        model="dashscope/qwen-tier-only-test",
        completion_response=None,
        custom_pricing=True,
        custom_llm_provider="dashscope",
        router_model_id=router_model_id,
    )
    assert selected is not None
    assert router_model_id in selected


@pytest.mark.parametrize("metadata_key", ["metadata", "litellm_metadata"])
def test_completion_cost_image_generation_reads_deployment_model_info_price_from_logging_metadata(
    _local_model_cost_map: None, metadata_key: str
) -> None:
    cost = completion_cost(
        completion_response=ImageResponse(data=[ImageObject(url="https://example.com/img.png")]),
        model="fal_ai/fal-ai/unlisted-image-model",
        call_type="image_generation",
        custom_pricing=True,
        litellm_logging_obj=SimpleNamespace(
            litellm_params={metadata_key: {"model_info": {"output_cost_per_image": 0.08}}}
        ),
    )

    assert cost == pytest.approx(0.08)


def test_completion_cost_image_generation_registered_deployment_price_keeps_map_token_rates(
    _local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    deployment_id: Final = "gemini-image-deployment-priced-per-image"
    monkeypatch.setitem(
        litellm.model_cost,
        deployment_id,
        {"mode": "image_generation", "litellm_provider": "gemini", "output_cost_per_image": 0.1},
    )
    map_model: Final = "gemini/gemini-3.1-flash-image"
    row: Final = litellm.model_cost[map_model]
    usage: Final = ImageUsage(
        input_tokens=10,
        input_tokens_details=ImageUsageInputTokensDetails(image_tokens=0, text_tokens=10),
        output_tokens=1290,
        total_tokens=1300,
    )

    cost = completion_cost(
        completion_response=ImageResponse(data=[ImageObject(url="https://example.com/img.png")], usage=usage),
        model=map_model,
        custom_llm_provider="gemini",
        call_type="image_generation",
        custom_pricing=True,
        router_model_id=deployment_id,
        litellm_logging_obj=SimpleNamespace(litellm_params={"metadata": {"model_info": {"id": deployment_id}}}),
    )

    expected: Final = (
        usage.input_tokens * row["input_cost_per_token"] + usage.output_tokens * row["output_cost_per_image_token"]
    )
    assert cost == pytest.approx(expected)


def test_completion_cost_image_generation_ignores_deployment_model_info_without_custom_pricing(
    _local_model_cost_map: None,
) -> None:
    cost = completion_cost(
        completion_response=ImageResponse(data=[ImageObject(url="https://example.com/img.png")]),
        model="fal_ai/openai/gpt-image-2",
        call_type="image_generation",
        custom_pricing=False,
        optional_params={"quality": "high", "image_size": {"width": 1024, "height": 1024}},
        litellm_logging_obj=SimpleNamespace(
            litellm_params={"litellm_metadata": {"model_info": {"output_cost_per_image": 0.5}}}
        ),
    )

    assert cost == pytest.approx(0.211)


async def test_router_image_generation_bills_litellm_params_output_cost_per_image() -> None:
    from litellm import Router

    router = Router(
        model_list=[
            {
                "model_name": "img",
                "litellm_params": {
                    "model": "fal_ai/fal-ai/unlisted-image-model",
                    "api_key": "sk-fake",
                    "output_cost_per_image": 0.08,
                },
            }
        ]
    )

    response = await router.aimage_generation(model="img", prompt="x", mock_response="https://example.com/img.png")

    assert response._hidden_params["response_cost"] == pytest.approx(0.08)


def test_tiered_pricing_only_deployment_completion_cost_is_nonzero():
    """End-to-end: a tier-only deployment must produce the tiered cost, not
    $0. Mirrors the reported dashscope/qwen3.7-plus trace (12 prompt + 377
    completion tokens).
    """
    from litellm import Router
    from litellm.types.utils import Choices, Message

    router = Router(
        model_list=[
            {
                "model_name": "qwen-3.7-plus",
                "litellm_params": {
                    "model": "dashscope/qwen3.7-plus",
                    "api_key": "sk-fake",
                },
                "model_info": {
                    "tiered_pricing": [
                        {
                            "input_cost_per_token": 4e-07,
                            "output_cost_per_token": 1.6e-06,
                            "range": [0, 256000],
                        },
                        {
                            "input_cost_per_token": 1.2e-06,
                            "output_cost_per_token": 4.8e-06,
                            "range": [256000, 1000000],
                        },
                    ],
                },
            },
        ]
    )
    router_model_id = router.model_list[0]["model_info"]["id"]

    response = ModelResponse(
        model="dashscope/qwen3.7-plus",
        choices=[Choices(index=0, message=Message(role="assistant", content="hi"))],
        usage=Usage(prompt_tokens=12, completion_tokens=377, total_tokens=389),
    )
    response._hidden_params = {"custom_llm_provider": "dashscope", "model_id": router_model_id}

    cost = completion_cost(
        completion_response=response,
        model="dashscope/qwen3.7-plus",
        custom_llm_provider="dashscope",
        custom_pricing=True,
        router_model_id=router_model_id,
    )

    expected = 12 * 4e-07 + 377 * 1.6e-06
    assert cost == pytest.approx(expected)
    assert cost > 0


def test_per_query_priced_rerank_deployment_completion_cost_is_nonzero():
    """A rerank deployment priced only via ``input_cost_per_query`` must resolve
    cost against its ``router_model_id`` entry: the shared backend alias has
    custom pricing stripped, so pricing it there bills every search unit as $0.
    """
    from litellm import Router

    router: Final = Router(
        model_list=[
            {
                "model_name": "semantic-ranker-default-004",
                "litellm_params": {
                    "model": "vertex_ai/semantic-ranker-default-004",
                    "vertex_project": "test-project",
                    "vertex_location": "us-east5",
                },
                "model_info": {"input_cost_per_query": 0.001},
            },
        ]
    )
    router_model_id: Final = router.model_list[0]["model_info"]["id"]
    assert litellm.model_cost["vertex_ai/semantic-ranker-default-004"].get("input_cost_per_query") is None

    response: Final = RerankResponse(
        id="vertex_ai_rerank_test",
        results=[{"index": 3, "relevance_score": 0.48}],
        meta={"billed_units": {"search_units": 3}},
    )

    cost: Final = completion_cost(
        completion_response=response,
        model="vertex_ai/semantic-ranker-default-004",
        custom_llm_provider="vertex_ai",
        call_type="arerank",
        custom_pricing=True,
        router_model_id=router_model_id,
    )

    assert cost == pytest.approx(3 * 0.001)


def test_azure_realtime_cost_calculator(_local_model_cost_map):

    cost = handle_realtime_stream_cost_calculation(
        results=[
            {
                "type": "session.created",
                "session": {"model": "gpt-4o-realtime-preview-2024-12-17"},
            },
        ],
        combined_usage_object=Usage(
            prompt_tokens=100,
            completion_tokens=100,
            prompt_tokens_details=PromptTokensDetailsWrapper(text_tokens=10, audio_tokens=90),
        ),
        custom_llm_provider="azure",
        litellm_model_name="my-custom-azure-deployment",
    )

    assert cost > 0


def test_azure_audio_output_cost_calculation(_local_model_cost_map):
    """
    Test that Azure audio models correctly calculate costs for audio output tokens.

    Reproduces issue: https://github.com/BerriAI/litellm/issues/19764
    Audio tokens should be charged at output_cost_per_audio_token rate,
    not at the text token rate (output_cost_per_token).
    """
    from litellm.types.utils import Choices, CompletionTokensDetailsWrapper, Message

    # Scenario from issue #19764:
    # Input: 17 text tokens, 0 audio tokens
    # Output: 110 text tokens, 482 audio tokens
    usage_object = Usage(
        prompt_tokens=17,
        completion_tokens=592,  # 110 text + 482 audio
        total_tokens=609,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=0,
            cached_tokens=0,
            text_tokens=17,
            image_tokens=0,
        ),
        completion_tokens_details=CompletionTokensDetailsWrapper(
            audio_tokens=482,
            reasoning_tokens=0,
            text_tokens=110,
        ),
    )

    completion = ModelResponse(
        id="test-azure-audio-cost",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    content="Test response",
                    role="assistant",
                ),
            )
        ],
        created=1729282652,
        model="azure/gpt-audio-2025-08-28",
        object="chat.completion",
        usage=usage_object,
    )

    cost = completion_cost(completion, model="azure/gpt-audio-2025-08-28")

    model_info = litellm.get_model_info("azure/gpt-audio-2025-08-28")

    # Calculate expected cost
    expected_input_cost = model_info["input_cost_per_token"] * 17  # text tokens
    expected_output_cost = (
        model_info["output_cost_per_token"] * 110  # text tokens
        + model_info["output_cost_per_audio_token"] * 482  # audio tokens
    )
    expected_total_cost = expected_input_cost + expected_output_cost

    # The bug was: all output tokens charged at text rate
    wrong_output_cost = model_info["output_cost_per_token"] * 592
    wrong_total_cost = expected_input_cost + wrong_output_cost

    # Verify audio tokens are NOT charged at text rate (the bug)
    assert abs(cost - wrong_total_cost) > 0.001, "Bug: Audio tokens are being charged at text token rate"

    # Verify cost matches
    assert abs(cost - expected_total_cost) < 0.0000001, f"Expected cost {expected_total_cost}, got {cost}"


def test_default_image_cost_calculator(monkeypatch):
    from litellm.cost_calculator import default_image_cost_calculator

    temp_object = {
        "litellm_provider": "azure",
        "input_cost_per_pixel": 10,
    }

    monkeypatch.setattr(
        litellm,
        "model_cost",
        {"azure/bf9001cd7209f5734ecb4ab937a5a0e2ba5f119708bd68f184db362930f9dc7b": temp_object},
    )

    args = {
        "model": "azure/bf9001cd7209f5734ecb4ab937a5a0e2ba5f119708bd68f184db362930f9dc7b",
        "custom_llm_provider": "azure",
        "quality": "standard",
        "n": 1,
        "size": "1024-x-1024",
        "optional_params": {},
    }
    cost = default_image_cost_calculator(**args)
    assert cost == 10485760


@pytest.mark.parametrize(
    ("model", "quality", "size", "priced_key", "pixels"),
    [
        ("azure/dall-e-3", "standard", "1024x1024", "azure/standard/1024-x-1024/dall-e-3", 1024 * 1024),
        ("azure/dall-e-3", "hd", "1024x1792", "azure/hd/1024-x-1792/dall-e-3", 1024 * 1792),
        ("dall-e-3", "hd", "1024x1792", "azure/hd/1024-x-1792/dall-e-3", 1024 * 1792),
    ],
)
def test_default_image_cost_calculator_matches_provider_first_quality_key(
    monkeypatch, model: str, quality: str, size: str, priced_key: str, pixels: int
):
    from litellm.cost_calculator import default_image_cost_calculator

    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            "azure/standard/1024-x-1024/dall-e-3": {"litellm_provider": "azure", "input_cost_per_pixel": 1e-08},
            "azure/hd/1024-x-1792/dall-e-3": {"litellm_provider": "azure", "input_cost_per_pixel": 3e-08},
        },
    )

    cost = default_image_cost_calculator(
        model=model,
        custom_llm_provider="azure",
        quality=quality,
        n=1,
        size=size,
        optional_params={},
    )

    assert cost == litellm.model_cost[priced_key]["input_cost_per_pixel"] * pixels


def test_cost_calculator_with_cache_creation():
    from litellm import completion_cost
    from litellm.types.utils import Choices, Message, Usage

    litellm_model_response = ModelResponse(
        id="chatcmpl-cc5638bc-fdfe-48e4-8884-57c8f4fb7c63",
        created=1750733889,
        model=None,
        object="chat.completion",
        system_fingerprint=None,
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    content="Hello! How can I help you today?",
                    role="assistant",
                    tool_calls=None,
                    function_call=None,
                    provider_specific_fields=None,
                ),
            )
        ],
        usage=Usage(
            **{
                "total_tokens": 28508,
                "prompt_tokens": 28495,
                "completion_tokens": 13,
                "prompt_tokens_details": {"audio_tokens": None, "cached_tokens": 0},
                "cache_read_input_tokens": 28491,
                "completion_tokens_details": {
                    "audio_tokens": None,
                    "reasoning_tokens": 0,
                    "accepted_prediction_tokens": None,
                    "rejected_prediction_tokens": None,
                },
                "cache_creation_input_tokens": 15,
            }
        ),
    )
    model = "claude-sonnet-4@20250514"

    assert litellm_model_response.usage.prompt_tokens_details.cached_tokens == 28491

    result = completion_cost(
        completion_response=litellm_model_response,
        model=model,
        custom_llm_provider="vertex_ai",
    )

    print(result)


def test_bedrock_cost_calculator_comparison_with_without_cache():
    """Test that Bedrock caching reduces costs compared to non-cached requests"""
    from litellm import completion_cost
    from litellm.types.utils import Choices, Message, Usage

    # Response WITHOUT caching
    response_no_cache = ModelResponse(
        id="msg_no_cache",
        created=1750733889,
        model="anthropic.claude-sonnet-4-20250514-v1:0",
        object="chat.completion",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    content="Response without cache",
                    role="assistant",
                ),
            )
        ],
        usage=Usage(
            total_tokens=28508,
            prompt_tokens=28495,
            completion_tokens=13,
        ),
    )

    # Response WITH caching (same total tokens, but most are cached)
    response_with_cache = ModelResponse(
        id="msg_with_cache",
        created=1750733889,
        model="anthropic.claude-sonnet-4-20250514-v1:0",
        object="chat.completion",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    content="Response with cache",
                    role="assistant",
                ),
            )
        ],
        usage=Usage(
            **{
                "total_tokens": 28508,
                "prompt_tokens": 28495,
                "completion_tokens": 13,
                "prompt_tokens_details": {"audio_tokens": None, "cached_tokens": 0},
                "cache_read_input_tokens": 28491,  # Most tokens are read from cache (cheaper)
                "completion_tokens_details": {
                    "audio_tokens": None,
                    "reasoning_tokens": 0,
                    "accepted_prediction_tokens": None,
                    "rejected_prediction_tokens": None,
                },
                "cache_creation_input_tokens": 15,  # Only 15 new tokens added to cache
            }
        ),
    )

    # Calculate costs
    cost_no_cache = completion_cost(
        completion_response=response_no_cache,
        model="bedrock/anthropic.claude-sonnet-4-20250514-v1:0",
        custom_llm_provider="bedrock",
    )

    cost_with_cache = completion_cost(
        completion_response=response_with_cache,
        model="bedrock/anthropic.claude-sonnet-4-20250514-v1:0",
        custom_llm_provider="bedrock",
    )

    # Verify that cached request is cheaper
    assert cost_with_cache < cost_no_cache
    print(f"Cost without cache: {cost_no_cache}")
    print(f"Cost with cache: {cost_with_cache}")




def test_gemini_25_explicit_caching_cost_direct_usage():
    """
    Test that Gemini 2.5 models correctly calculate costs with explicit caching.

    This test reproduces the issue from #11156 where cached tokens should receive
    a 75% discount.
    """
    from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
    from litellm.types.utils import (
        CompletionTokensDetailsWrapper,
        PromptTokensDetailsWrapper,
        Usage,
    )
    from litellm.utils import get_model_info

    model_info = get_model_info(model="gemini-2.5-pro", custom_llm_provider="gemini")

    usage = Usage(
        completion_tokens=2522,
        prompt_tokens=42001,
        total_tokens=44523,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            accepted_prediction_tokens=None,
            audio_tokens=None,
            reasoning_tokens=1908,
            rejected_prediction_tokens=None,
            text_tokens=614,
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=None, cached_tokens=40938, text_tokens=1063, image_tokens=None
        ),
    )

    input_cost, output_cost = generic_cost_per_token(
        model="gemini/gemini-2.5-pro",
        usage=usage,
        custom_llm_provider="gemini",
    )

    total_cost = input_cost + output_cost

    expected_higher_than_actual_cost = (
        model_info["input_cost_per_token"] * usage.prompt_tokens
        + model_info["output_cost_per_token"] * usage.completion_tokens
    )

    print(f"expected_higher_than_actual_cost: {expected_higher_than_actual_cost}")

    assert expected_higher_than_actual_cost > total_cost

    expected_actual_cost = (
        model_info["input_cost_per_token"] * usage.prompt_tokens_details.text_tokens
        + model_info["cache_read_input_token_cost"] * usage.prompt_tokens_details.cached_tokens
        + model_info["output_cost_per_token"] * usage.completion_tokens
    )

    print(
        f"model_info['input_cost_per_token']: {model_info['input_cost_per_token']}, usage.prompt_tokens_details.text_tokens: {usage.prompt_tokens_details.text_tokens}, model_info['cache_read_input_token_cost']: {model_info['cache_read_input_token_cost']}, model_info['output_cost_per_token']: {model_info['output_cost_per_token']}"
    )

    print(f"Expected actual cost: {expected_actual_cost}")

    assert expected_actual_cost == total_cost


def test_azure_ai_cache_cost_calculation(_local_model_cost_map):
    """
    Test that azure_ai provider correctly calculates cache costs using generic_cost_per_token.

    This verifies that azure_ai models with custom cache pricing in model_info
    will have their cache_creation_input_token_cost and cache_read_input_token_cost
    applied correctly.
    """
    from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
    from litellm.types.utils import PromptTokensDetailsWrapper, Usage

    # Register a custom azure_ai model with cache pricing
    test_model_id = "test-azure-ai-claude-model"
    litellm.register_model(
        model_cost={
            test_model_id: {
                "input_cost_per_token": 5.0e-06,
                "output_cost_per_token": 2.5e-05,
                "cache_creation_input_token_cost": 6.25e-06,
                "cache_read_input_token_cost": 5.0e-07,
                "litellm_provider": "azure_ai",
                "max_tokens": 200000,
            }
        }
    )

    # Create usage with cache tokens
    usage = Usage(
        completion_tokens=100,
        prompt_tokens=1000,
        total_tokens=1100,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=800,  # 800 cache read tokens
            text_tokens=100,  # 100 regular text tokens
        ),
        cache_creation_input_tokens=100,  # 100 cache creation tokens
    )

    input_cost, output_cost = generic_cost_per_token(
        model=test_model_id,
        usage=usage,
        custom_llm_provider="azure_ai",
    )

    total_cost = input_cost + output_cost

    # Calculate expected cost manually
    model_info = litellm.model_cost[test_model_id]
    expected_input_cost = (
        model_info["input_cost_per_token"] * 100  # text tokens
        + model_info["cache_read_input_token_cost"] * 800  # cached tokens
        + model_info["cache_creation_input_token_cost"] * 100  # cache creation tokens
    )
    expected_output_cost = model_info["output_cost_per_token"] * 100

    print(f"Input cost: {input_cost}, Expected: {expected_input_cost}")
    print(f"Output cost: {output_cost}, Expected: {expected_output_cost}")
    print(f"Total cost: {total_cost}")

    assert abs(input_cost - expected_input_cost) < 1e-10, (
        f"Input cost mismatch: got {input_cost}, expected {expected_input_cost}"
    )
    assert abs(output_cost - expected_output_cost) < 1e-10, (
        f"Output cost mismatch: got {output_cost}, expected {expected_output_cost}"
    )


def test_vertex_regional_deployment_costs_uplift_over_global(monkeypatch):
    """
    Regression for https://github.com/BerriAI/litellm/issues/34393: two Vertex
    deployments differing only in vertex_location must not price identically.
    Google bills non-global endpoints at 1.1x for regional-pricing models, so the
    regional request costs 1.1x the global one for the exact same usage, through
    both vertex cost routes (Claude via cost_per_token, Gemini via
    cost_per_character's token fallback).
    """
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))

    usage = Usage(prompt_tokens=15, completion_tokens=5, total_tokens=20)
    for model in ("claude-haiku-4-5@20251001", "gemini-3.5-flash"):
        global_prompt, global_completion = cost_per_token(
            model=model,
            custom_llm_provider="vertex_ai",
            usage_object=usage,
            vertex_location="global",
        )
        regional_prompt, regional_completion = cost_per_token(
            model=model,
            custom_llm_provider="vertex_ai",
            usage_object=usage,
            vertex_location="us-east5",
        )
        global_total = global_prompt + global_completion
        regional_total = regional_prompt + regional_completion
        assert global_total > 0
        assert regional_total == pytest.approx(global_total * 1.10, rel=1e-9), (
            f"{model}: regional Vertex request must cost 1.1x the global one"
        )


def test_vertex_uplift_composes_with_above_128k_pricing(monkeypatch):
    """The regional-endpoint uplift multiplies whatever rate the request priced at,
    including the above-128k dynamic rates, so a synthetic model carrying both keys
    prices regional above-128k usage at 1.1x the above-128k rate."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            **litellm.get_model_cost_map(url=""),
            "vertex_ai/fake-regional-128k-model": {
                "litellm_provider": "vertex_ai",
                "mode": "chat",
                "input_cost_per_token": 1e-06,
                "output_cost_per_token": 2e-06,
                "input_cost_per_token_above_128k_tokens": 2e-06,
                "output_cost_per_token_above_128k_tokens": 4e-06,
                "regional_endpoint_uplift_multiplier": 1.1,
            },
        },
    )

    usage = Usage(prompt_tokens=200_000, completion_tokens=10, total_tokens=200_010)
    global_prompt, global_completion = cost_per_token(
        model="fake-regional-128k-model",
        custom_llm_provider="vertex_ai",
        usage_object=usage,
        vertex_location="global",
    )
    regional_prompt, regional_completion = cost_per_token(
        model="fake-regional-128k-model",
        custom_llm_provider="vertex_ai",
        usage_object=usage,
        vertex_location="europe-west1",
    )

    assert global_prompt == pytest.approx(200_000 * 2e-06, rel=1e-9)
    assert regional_prompt == pytest.approx(global_prompt * 1.10, rel=1e-9)
    assert regional_completion == pytest.approx(global_completion * 1.10, rel=1e-9)


def test_cost_discount_vertex_ai(monkeypatch):
    """
    Test that cost discount is applied correctly for Vertex AI provider
    """
    from litellm import completion_cost
    from litellm.types.utils import Usage

    # Create mock response (use a model that exists in model_prices_and_context_window.json)
    response = ModelResponse(
        id="test-id",
        choices=[],
        created=1234567890,
        model="gemini-3-pro-preview",
        object="chat.completion",
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )

    # Calculate cost without discount
    monkeypatch.setattr(litellm, "cost_discount_config", {})
    cost_without_discount = completion_cost(
        completion_response=response,
        model="vertex_ai/gemini-3-pro-preview",
        custom_llm_provider="vertex_ai",
    )

    # Set 5% discount for vertex_ai
    monkeypatch.setattr(litellm, "cost_discount_config", {"vertex_ai": 0.05})

    # Calculate cost with discount
    cost_with_discount = completion_cost(
        completion_response=response,
        model="vertex_ai/gemini-3-pro-preview",
        custom_llm_provider="vertex_ai",
    )

    # Verify discount is applied (5% off means 95% of original cost)
    expected_cost = cost_without_discount * 0.95
    assert cost_with_discount == pytest.approx(expected_cost, rel=1e-9)

    print("✓ Cost discount test passed:")
    print(f"  - Original cost: ${cost_without_discount:.6f}")
    print(f"  - Discounted cost (5% off): ${cost_with_discount:.6f}")
    print(f"  - Savings: ${cost_without_discount - cost_with_discount:.6f}")


def test_cost_discount_not_applied_to_other_providers(monkeypatch):
    """
    Test that cost discount only applies to configured providers
    """
    from litellm import completion_cost
    from litellm.types.utils import Usage

    # Create mock response for OpenAI
    response = ModelResponse(
        id="test-id",
        choices=[],
        created=1234567890,
        model="gpt-4",
        object="chat.completion",
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )

    # Set discount only for vertex_ai (not openai)
    monkeypatch.setattr(litellm, "cost_discount_config", {"vertex_ai": 0.05})

    # Calculate cost for OpenAI - should NOT have discount applied
    cost_with_selective_discount = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Clear discount config
    monkeypatch.setattr(litellm, "cost_discount_config", {})
    cost_without_discount = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Costs should be the same (no discount applied to OpenAI)
    assert cost_with_selective_discount == cost_without_discount

    print("✓ Selective discount test passed:")
    print(f"  - OpenAI cost (no discount configured): ${cost_without_discount:.6f}")
    print(f"  - Cost remains unchanged: ${cost_with_selective_discount:.6f}")


def test_cost_margin_percentage(monkeypatch):
    """
    Test that percentage-based cost margin is applied correctly
    """
    from litellm import completion_cost
    from litellm.types.utils import Usage

    # Create mock response
    response = ModelResponse(
        id="test-id",
        choices=[],
        created=1234567890,
        model="gpt-4",
        object="chat.completion",
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )

    # Calculate cost without margin
    monkeypatch.setattr(litellm, "cost_margin_config", {})
    cost_without_margin = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Set 10% margin for openai
    monkeypatch.setattr(litellm, "cost_margin_config", {"openai": 0.10})

    # Calculate cost with margin
    cost_with_margin = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Verify margin is applied (10% margin means 110% of original cost)
    expected_cost = cost_without_margin * 1.10
    assert cost_with_margin == pytest.approx(expected_cost, rel=1e-9)

    print("✓ Cost margin percentage test passed:")
    print(f"  - Original cost: ${cost_without_margin:.6f}")
    print(f"  - Cost with margin (10%): ${cost_with_margin:.6f}")
    print(f"  - Margin added: ${cost_with_margin - cost_without_margin:.6f}")


def test_cost_margin_fixed_amount(monkeypatch):
    """
    Test that fixed amount cost margin is applied correctly
    """
    from litellm import completion_cost
    from litellm.types.utils import Usage

    # Create mock response
    response = ModelResponse(
        id="test-id",
        choices=[],
        created=1234567890,
        model="gpt-4",
        object="chat.completion",
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )

    # Calculate cost without margin
    monkeypatch.setattr(litellm, "cost_margin_config", {})
    cost_without_margin = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Set $0.001 fixed margin for openai
    monkeypatch.setattr(litellm, "cost_margin_config", {"openai": {"fixed_amount": 0.001}})

    # Calculate cost with margin
    cost_with_margin = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Verify fixed margin is applied
    expected_cost = cost_without_margin + 0.001
    assert cost_with_margin == pytest.approx(expected_cost, rel=1e-9)

    print("✓ Cost margin fixed amount test passed:")
    print(f"  - Original cost: ${cost_without_margin:.6f}")
    print(f"  - Cost with margin ($0.001): ${cost_with_margin:.6f}")
    print(f"  - Margin added: ${cost_with_margin - cost_without_margin:.6f}")


def test_cost_margin_combined(monkeypatch):
    """
    Test that combined percentage and fixed amount margin is applied correctly
    """
    from litellm import completion_cost
    from litellm.types.utils import Usage

    # Create mock response
    response = ModelResponse(
        id="test-id",
        choices=[],
        created=1234567890,
        model="gpt-4",
        object="chat.completion",
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )

    # Calculate cost without margin
    monkeypatch.setattr(litellm, "cost_margin_config", {})
    cost_without_margin = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Set 8% margin + $0.0005 fixed for openai
    monkeypatch.setattr(litellm, "cost_margin_config", {"openai": {"percentage": 0.08, "fixed_amount": 0.0005}})

    # Calculate cost with margin
    cost_with_margin = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Verify combined margin is applied
    expected_cost = cost_without_margin * 1.08 + 0.0005
    assert cost_with_margin == pytest.approx(expected_cost, rel=1e-9)

    print("✓ Cost margin combined test passed:")
    print(f"  - Original cost: ${cost_without_margin:.6f}")
    print(f"  - Cost with margin (8% + $0.0005): ${cost_with_margin:.6f}")
    print(f"  - Margin added: ${cost_with_margin - cost_without_margin:.6f}")


def test_cost_margin_global(monkeypatch):
    """
    Test that global margin is applied when no provider-specific margin is configured
    """
    from litellm import completion_cost
    from litellm.types.utils import Usage

    # Create mock response
    response = ModelResponse(
        id="test-id",
        choices=[],
        created=1234567890,
        model="gpt-4",
        object="chat.completion",
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )

    # Calculate cost without margin
    monkeypatch.setattr(litellm, "cost_margin_config", {})
    cost_without_margin = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Set 5% global margin (no provider-specific margin)
    monkeypatch.setattr(litellm, "cost_margin_config", {"global": 0.05})

    # Calculate cost with global margin
    cost_with_global_margin = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Verify global margin is applied
    expected_cost = cost_without_margin * 1.05
    assert cost_with_global_margin == pytest.approx(expected_cost, rel=1e-9)

    print("✓ Cost margin global test passed:")
    print(f"  - Original cost: ${cost_without_margin:.6f}")
    print(f"  - Cost with global margin (5%): ${cost_with_global_margin:.6f}")
    print(f"  - Margin added: ${cost_with_global_margin - cost_without_margin:.6f}")


def test_cost_margin_provider_overrides_global(monkeypatch):
    """
    Test that provider-specific margin overrides global margin
    """
    from litellm import completion_cost
    from litellm.types.utils import Usage

    # Create mock response
    response = ModelResponse(
        id="test-id",
        choices=[],
        created=1234567890,
        model="gpt-4",
        object="chat.completion",
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )

    # Calculate cost without margin
    monkeypatch.setattr(litellm, "cost_margin_config", {})
    cost_without_margin = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Set 5% global margin and 10% provider-specific margin
    monkeypatch.setattr(litellm, "cost_margin_config", {"global": 0.05, "openai": 0.10})

    # Calculate cost - should use provider-specific margin (10%), not global (5%)
    cost_with_provider_margin = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Verify provider-specific margin is used (not global)
    expected_cost = cost_without_margin * 1.10  # 10% from provider, not 5% from global
    assert cost_with_provider_margin == pytest.approx(expected_cost, rel=1e-9)

    print("✓ Cost margin provider override test passed:")
    print(f"  - Original cost: ${cost_without_margin:.6f}")
    print(f"  - Cost with provider margin (10%, overrides 5% global): ${cost_with_provider_margin:.6f}")
    print(f"  - Margin added: ${cost_with_provider_margin - cost_without_margin:.6f}")


def test_cost_margin_with_discount(monkeypatch):
    """
    Test that margin is applied after discount (independent calculation)
    """
    from litellm import completion_cost
    from litellm.types.utils import Usage

    # Create mock response
    response = ModelResponse(
        id="test-id",
        choices=[],
        created=1234567890,
        model="gpt-4",
        object="chat.completion",
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )

    # Calculate base cost
    monkeypatch.setattr(litellm, "cost_margin_config", {})
    monkeypatch.setattr(litellm, "cost_discount_config", {})
    base_cost = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Set 5% discount and 10% margin
    monkeypatch.setattr(litellm, "cost_discount_config", {"openai": 0.05})
    monkeypatch.setattr(litellm, "cost_margin_config", {"openai": 0.10})

    # Calculate cost with both discount and margin
    cost_with_both = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Verify: discount applied first, then margin
    # Base cost -> discount: base * 0.95 -> margin: (base * 0.95) * 1.10
    expected_cost = base_cost * 0.95 * 1.10
    assert cost_with_both == pytest.approx(expected_cost, rel=1e-9)

    print("✓ Cost margin with discount test passed:")
    print(f"  - Base cost: ${base_cost:.6f}")
    print(f"  - Cost with 5% discount + 10% margin: ${cost_with_both:.6f}")
    print(f"  - Expected: ${expected_cost:.6f}")




def test_completion_cost_extracts_service_tier_from_response(_local_model_cost_map):
    """Test that completion_cost extracts service_tier from completion_response object."""
    from litellm import completion_cost

    # Test with gpt-5-nano which has flex pricing
    model = "gpt-5-nano"

    # Create usage object
    usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

    # Create ModelResponse with service_tier in the response object
    response_with_service_tier = ModelResponse(
        usage=usage,
        model=model,
    )
    # Set service_tier as an attribute on the response
    setattr(response_with_service_tier, "service_tier", "flex")

    # Test that flex pricing is used when service_tier is in response
    flex_cost = completion_cost(
        completion_response=response_with_service_tier,
        model=model,
        custom_llm_provider="openai",
    )

    # Create ModelResponse without service_tier (should use standard pricing)
    response_without_service_tier = ModelResponse(
        usage=usage,
        model=model,
    )

    # Test that standard pricing is used when service_tier is not in response
    standard_cost = completion_cost(
        completion_response=response_without_service_tier,
        model=model,
        custom_llm_provider="openai",
    )

    # Flex should be approximately 50% of standard
    assert flex_cost > 0, "Flex cost should be greater than 0"
    assert standard_cost > 0, "Standard cost should be greater than 0"
    assert flex_cost < standard_cost, "Flex cost should be less than standard cost"

    flex_ratio = flex_cost / standard_cost
    assert 0.45 <= flex_ratio <= 0.55, f"Flex pricing should be ~50% of standard, got {flex_ratio:.2f}"


def test_completion_cost_extracts_service_tier_from_usage(_local_model_cost_map):
    """Test that completion_cost extracts service_tier from usage object."""
    from litellm import completion_cost

    # Test with gpt-5-nano which has flex pricing
    model = "gpt-5-nano"

    # Create usage object with service_tier
    usage_with_service_tier = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
    # Set service_tier as an attribute on the usage object
    setattr(usage_with_service_tier, "service_tier", "flex")

    # Create ModelResponse with usage containing service_tier
    response = ModelResponse(
        usage=usage_with_service_tier,
        model=model,
    )

    # Test that flex pricing is used when service_tier is in usage
    flex_cost = completion_cost(
        completion_response=response,
        model=model,
        custom_llm_provider="openai",
    )

    # Create usage object without service_tier
    usage_without_service_tier = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

    # Create ModelResponse with usage without service_tier
    response_standard = ModelResponse(
        usage=usage_without_service_tier,
        model=model,
    )

    # Test that standard pricing is used when service_tier is not in usage
    standard_cost = completion_cost(
        completion_response=response_standard,
        model=model,
        custom_llm_provider="openai",
    )

    # Flex should be approximately 50% of standard
    assert flex_cost > 0, "Flex cost should be greater than 0"
    assert standard_cost > 0, "Standard cost should be greater than 0"
    assert flex_cost < standard_cost, "Flex cost should be less than standard cost"

    flex_ratio = flex_cost / standard_cost
    assert 0.45 <= flex_ratio <= 0.55, f"Flex pricing should be ~50% of standard, got {flex_ratio:.2f}"


def test_completion_cost_service_tier_priority(_local_model_cost_map):
    """Test that service_tier extraction follows priority: optional_params > completion_response > usage."""
    from litellm import completion_cost

    # Test with gpt-5-nano which has flex pricing
    model = "gpt-5-nano"

    # Create usage object with service_tier="flex"
    usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
    setattr(usage, "service_tier", "flex")

    # Create response with service_tier="priority"
    response = ModelResponse(
        usage=usage,
        model=model,
    )
    setattr(response, "service_tier", "priority")

    # Test that optional_params takes priority over response and usage
    cost_from_params = completion_cost(
        completion_response=response,
        model=model,
        custom_llm_provider="openai",
        optional_params={"service_tier": "flex"},
    )

    # Test that response takes priority over usage when optional_params is not provided
    completion_cost(
        completion_response=response,
        model=model,
        custom_llm_provider="openai",
    )

    # Test that usage is used when neither optional_params nor response have service_tier
    # Create a new response without service_tier attribute
    response_no_tier = ModelResponse(
        usage=usage,
        model=model,
    )
    # Don't set service_tier on response, so it will fall back to usage

    cost_from_usage = completion_cost(
        completion_response=response_no_tier,
        model=model,
        custom_llm_provider="openai",
    )

    # All should use flex pricing (from different sources)
    assert cost_from_params > 0, "Cost from params should be greater than 0"
    assert cost_from_usage > 0, "Cost from usage should be greater than 0"

    # Costs should be similar (all using flex)
    assert abs(cost_from_params - cost_from_usage) < 1e-6, "Costs from params and usage should be similar (both flex)"


def test_completion_cost_service_tier_for_bedrock(_local_model_cost_map):
    """Test that Bedrock cost calculation applies service_tier-specific pricing."""
    from litellm import completion_cost

    model = "bedrock/us-east-1/test-bedrock-service-tier-cost-model"
    litellm.register_model(
        model_cost={
            model: {
                "input_cost_per_token": 0.001,
                "output_cost_per_token": 0.002,
                "input_cost_per_token_priority": 0.01,
                "output_cost_per_token_priority": 0.02,
                "input_cost_per_token_flex": 0.0005,
                "output_cost_per_token_flex": 0.001,
                "litellm_provider": "bedrock",
                "max_tokens": 8192,
            }
        }
    )

    usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    response = ModelResponse(usage=usage, model=model)

    default_cost = completion_cost(
        completion_response=response,
        model=model,
        custom_llm_provider="bedrock",
    )

    priority_cost = completion_cost(
        completion_response=response,
        model=model,
        custom_llm_provider="bedrock",
        optional_params={"service_tier": "priority"},
    )

    response_with_flex_tier = ModelResponse(usage=usage, model=model)
    setattr(response_with_flex_tier, "service_tier", "flex")
    flex_cost = completion_cost(
        completion_response=response_with_flex_tier,
        model=model,
        custom_llm_provider="bedrock",
    )

    assert priority_cost > default_cost > flex_cost > 0


def test_completion_cost_service_tier_for_anthropic(_local_model_cost_map):
    """
    Anthropic priority-tier requests must be priced at the priority rate.

    Regression for LIT-3771: the Anthropic cost route dropped ``service_tier``,
    so priority requests (whose tier is reported on the response usage) were
    always billed at the standard rate. The tier is captured by the
    transformation and must flow through to ``generic_cost_per_token``.
    """
    from litellm import completion_cost
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    model = "claude-test-service-tier-cost-model"
    litellm.register_model(
        model_cost={
            model: {
                "input_cost_per_token": 3e-6,
                "output_cost_per_token": 15e-6,
                "input_cost_per_token_priority": 6e-6,
                "output_cost_per_token_priority": 30e-6,
                "litellm_provider": "anthropic",
                "max_tokens": 8192,
            }
        }
    )

    def _cost_for_tier(service_tier):
        usage = AnthropicConfig().calculate_usage(
            usage_object={
                "input_tokens": 1000,
                "output_tokens": 500,
                "service_tier": service_tier,
            },
            reasoning_content=None,
        )
        response = ModelResponse(usage=usage, model=model)
        return completion_cost(
            completion_response=response,
            model=model,
            custom_llm_provider="anthropic",
        )

    standard_cost = _cost_for_tier("standard")
    priority_cost = _cost_for_tier("priority")

    expected_standard = 1000 * 3e-6 + 500 * 15e-6
    assert standard_cost == pytest.approx(expected_standard)
    # priority rates are exactly 2x standard for both input and output
    assert priority_cost == pytest.approx(2 * standard_cost)


def test_completion_cost_anthropic_auto_tier_uses_served_priority_rate(_local_model_cost_map):
    """
    Proxy billing path regression for LIT-3771.

    Priority is opted into with ``service_tier="auto"``; Anthropic then serves
    "priority" and reports it on the response usage. The proxy forwards the
    request-level "auto" into ``completion_cost`` (via ``_response_cost_calculator``),
    and that preference must not shadow the served tier, otherwise priority
    requests are silently billed at the standard rate.
    """
    from litellm import completion_cost
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    model = "claude-test-auto-tier-cost-model"
    litellm.register_model(
        model_cost={
            model: {
                "input_cost_per_token": 3e-6,
                "output_cost_per_token": 15e-6,
                "input_cost_per_token_priority": 6e-6,
                "output_cost_per_token_priority": 30e-6,
                "litellm_provider": "anthropic",
                "max_tokens": 8192,
            }
        }
    )

    usage = AnthropicConfig().calculate_usage(
        usage_object={
            "input_tokens": 1000,
            "output_tokens": 500,
            "service_tier": "priority",
        },
        reasoning_content=None,
    )
    response = ModelResponse(usage=usage, model=model)

    cost = completion_cost(
        completion_response=response,
        model=model,
        custom_llm_provider="anthropic",
        service_tier="auto",
        optional_params={"service_tier": "auto"},
    )

    expected_priority = 1000 * 6e-6 + 500 * 30e-6
    assert cost == pytest.approx(expected_priority)


def test_completion_cost_vertex_ai_gemini_flex_traffic_type(_local_model_cost_map):
    """
    Vertex AI flex-tier billing regression for issue #37647.

    Vertex Gemini 3.x models route through ``cost_per_character`` (the
    ``cost_router`` token-path gate only matches "gemini-2"), and its token
    fallbacks dropped ``service_tier``. A response served with
    ``trafficType=ON_DEMAND_FLEX`` must be billed at the flex rate, not the
    standard rate.
    """
    from litellm import completion_cost

    model = "gemini-3-test-flex-tier-cost-model"
    litellm.register_model(
        model_cost={
            model: {
                "input_cost_per_token": 1.5e-6,
                "output_cost_per_token": 9e-6,
                "input_cost_per_token_flex": 7.5e-7,
                "output_cost_per_token_flex": 4.5e-6,
                "litellm_provider": "vertex_ai",
                "max_tokens": 8192,
            }
        }
    )

    def _cost_for_traffic_type(traffic_type):
        usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
        response = ModelResponse(usage=usage, model=model)
        response._hidden_params["provider_specific_fields"] = {"traffic_type": traffic_type}
        return completion_cost(
            completion_response=response,
            model=model,
            custom_llm_provider="vertex_ai",
        )

    standard_cost = _cost_for_traffic_type("ON_DEMAND")
    flex_cost = _cost_for_traffic_type("ON_DEMAND_FLEX")

    assert standard_cost == pytest.approx(1000 * 1.5e-6 + 500 * 9e-6)
    assert flex_cost == pytest.approx(1000 * 7.5e-7 + 500 * 4.5e-6)


def test_completion_cost_non_string_service_tier_defers_to_served_tier(_local_model_cost_map):
    """
    Regression: a non-string request-level ``service_tier`` (reachable via
    ``allowed_openai_params``/``drop_params``) must not crash cost tracking.

    Before the fix, ``completion_cost`` called ``service_tier.lower()`` on the
    request-level value, so a dict raised ``AttributeError``. ``_response_cost_calculator``
    swallowed it and reported ``response_cost=None``, silently dropping the cost.
    The non-string preference must be ignored so pricing defers to the tier the
    provider actually served on the response usage.
    """
    from litellm import completion_cost
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    model = "claude-test-non-string-tier-cost-model"
    litellm.register_model(
        model_cost={
            model: {
                "input_cost_per_token": 3e-6,
                "output_cost_per_token": 15e-6,
                "input_cost_per_token_priority": 6e-6,
                "output_cost_per_token_priority": 30e-6,
                "litellm_provider": "anthropic",
                "max_tokens": 8192,
            }
        }
    )

    usage = AnthropicConfig().calculate_usage(
        usage_object={
            "input_tokens": 1000,
            "output_tokens": 500,
            "service_tier": "priority",
        },
        reasoning_content=None,
    )
    response = ModelResponse(usage=usage, model=model)

    cost = completion_cost(
        completion_response=response,
        model=model,
        custom_llm_provider="anthropic",
        optional_params={"service_tier": {"name": "auto"}},
    )

    expected_priority = 1000 * 6e-6 + 500 * 30e-6
    assert cost == pytest.approx(expected_priority)


def test_completion_cost_non_string_response_service_tier_defers_to_served_tier(_local_model_cost_map):
    """
    Regression: a non-string ``service_tier`` on the response object must not
    crash cost tracking.

    Before the fix ``completion_cost`` read the response-level value verbatim and
    passed it to ``_get_service_tier_cost_key``, which called ``service_tier.lower()``
    on the dict and raised ``AttributeError``. The non-string preference is not a
    billable tier, so pricing defers to the concrete tier the provider served on
    the usage object instead of crashing.
    """
    from litellm import completion_cost
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    model = "claude-test-response-non-string-tier-cost-model"
    litellm.register_model(
        model_cost={
            model: {
                "input_cost_per_token": 3e-6,
                "output_cost_per_token": 15e-6,
                "input_cost_per_token_priority": 6e-6,
                "output_cost_per_token_priority": 30e-6,
                "litellm_provider": "anthropic",
                "max_tokens": 8192,
            }
        }
    )

    usage = AnthropicConfig().calculate_usage(
        usage_object={
            "input_tokens": 1000,
            "output_tokens": 500,
            "service_tier": "priority",
        },
        reasoning_content=None,
    )
    response = ModelResponse(usage=usage, model=model, service_tier={"name": "priority"})

    cost = completion_cost(
        completion_response=response,
        model=model,
        custom_llm_provider="anthropic",
    )

    expected_priority = 1000 * 6e-6 + 500 * 30e-6
    assert cost == pytest.approx(expected_priority)


def test_completion_cost_non_string_usage_service_tier_prices_standard(_local_model_cost_map):
    """
    Regression: a non-string ``service_tier`` on the usage object must not crash
    cost tracking.

    The dict reaches ``completion_cost`` via the usage extraction path with no
    concrete tier to defer to, so pricing falls back to the standard rate instead
    of raising ``AttributeError`` in ``_get_service_tier_cost_key``.
    """
    from litellm import completion_cost

    model = "claude-test-usage-non-string-tier-cost-model"
    litellm.register_model(
        model_cost={
            model: {
                "input_cost_per_token": 3e-6,
                "output_cost_per_token": 15e-6,
                "input_cost_per_token_priority": 6e-6,
                "output_cost_per_token_priority": 30e-6,
                "litellm_provider": "anthropic",
                "max_tokens": 8192,
            }
        }
    )

    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        service_tier={"name": "priority"},
    )
    response = ModelResponse(usage=usage, model=model)

    cost = completion_cost(
        completion_response=response,
        model=model,
        custom_llm_provider="anthropic",
    )

    expected_standard = 1000 * 3e-6 + 500 * 15e-6
    assert cost == pytest.approx(expected_standard)


def test_anthropic_cost_per_token_prices_cache_at_served_tier_with_multiplier(_local_model_cost_map):
    """
    Regression for the cache/tier interaction in the Anthropic geo/speed path.

    When a request is served at "priority" and also carries the ``fast`` speed
    multiplier, the cache portion must be priced at the served tier's cache
    rate and, per Anthropic's fast-mode pricing, scaled by the multiplier like
    every other token type.
    """
    from litellm.llms.anthropic.cost_calculation import (
        cost_per_token as anthropic_cost_per_token,
    )
    from litellm.types.utils import PromptTokensDetailsWrapper, Usage

    model = "claude-test-priority-cache-fast-model"
    litellm.register_model(
        model_cost={
            model: {
                "input_cost_per_token": 3e-6,
                "output_cost_per_token": 15e-6,
                "cache_read_input_token_cost": 0.3e-6,
                "input_cost_per_token_priority": 6e-6,
                "output_cost_per_token_priority": 30e-6,
                "cache_read_input_token_cost_priority": 0.6e-6,
                "litellm_provider": "anthropic",
                "max_tokens": 8192,
                "provider_specific_entry": {"fast": 2.0},
            }
        }
    )

    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=200),
    )
    usage.speed = "fast"

    prompt_cost, completion_cost = anthropic_cost_per_token(model=model, usage=usage, service_tier="priority")

    expected_prompt = ((1000 - 200) * 6e-6 + 200 * 0.6e-6) * 2
    expected_completion = 500 * 30e-6 * 2
    assert prompt_cost == pytest.approx(expected_prompt)
    assert completion_cost == pytest.approx(expected_completion)


def _register_anthropic_geo_cache_model(model: str) -> None:
    litellm.register_model(
        model_cost={
            model: {
                "input_cost_per_token": 5e-6,
                "output_cost_per_token": 25e-6,
                "cache_creation_input_token_cost": 6.25e-6,
                "cache_read_input_token_cost": 0.5e-6,
                "litellm_provider": "anthropic",
                "max_tokens": 8192,
                "provider_specific_entry": {"us": 1.1, "fast": 2.0},
            }
        }
    )


def test_anthropic_geo_multiplier_applies_to_cache_tokens(_local_model_cost_map, monkeypatch):
    """
    Regression: the regional (geo) uplift must scale cache read and cache write
    cost too, not just non-cache input and output.

    Anthropic's regional surcharge applies to every token type, so a cache-heavy
    row (nearly all cache-creation tokens) must still come in 10% above the
    global-priced row. Before the fix the uplift was applied only to the
    non-cache portion, so cache-heavy spend was under-reported by ~10%.
    """
    from litellm.llms.anthropic.cost_calculation import (
        cost_per_token as anthropic_cost_per_token,
    )
    from litellm.types.utils import PromptTokensDetailsWrapper, Usage

    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    model = "claude-test-geo-cache-model"
    _register_anthropic_geo_cache_model(model)

    def make_usage() -> "Usage":
        return Usage(
            prompt_tokens=1_000_000,
            completion_tokens=500,
            total_tokens=1_000_500,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                cached_tokens=200_000,
                cache_creation_tokens=799_800,
            ),
        )

    base_usage = make_usage()
    base_prompt_cost, base_completion_cost = anthropic_cost_per_token(model=model, usage=base_usage)

    geo_usage = make_usage()
    geo_usage.inference_geo = "us"
    geo_prompt_cost, geo_completion_cost = anthropic_cost_per_token(model=model, usage=geo_usage)

    expected_base_prompt = 200 * 5e-6 + 200_000 * 0.5e-6 + 799_800 * 6.25e-6
    assert base_prompt_cost == pytest.approx(expected_base_prompt)
    assert geo_prompt_cost == pytest.approx(expected_base_prompt * 1.1)
    assert geo_completion_cost == pytest.approx(base_completion_cost * 1.1)


def test_anthropic_geo_and_fast_multipliers_compose(_local_model_cost_map, monkeypatch):
    """
    Anthropic's fast-mode pricing doubles every token type, cache reads and
    writes included, and the regional uplift stacks on top, so a fast +
    regional row prices as ``(non_cache + cache) * fast * geo``.
    """
    from litellm.llms.anthropic.cost_calculation import (
        cost_per_token as anthropic_cost_per_token,
    )
    from litellm.types.utils import PromptTokensDetailsWrapper, Usage

    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    model = "claude-test-geo-fast-cache-model"
    _register_anthropic_geo_cache_model(model)

    usage = Usage(
        prompt_tokens=10_000,
        completion_tokens=500,
        total_tokens=10_500,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=2_000,
            cache_creation_tokens=6_000,
        ),
    )
    usage.inference_geo = "us"
    usage.speed = "fast"

    prompt_cost, completion_cost = anthropic_cost_per_token(model=model, usage=usage)

    cache_cost = 2_000 * 0.5e-6 + 6_000 * 6.25e-6
    non_cache_cost = 2_000 * 5e-6
    assert prompt_cost == pytest.approx((non_cache_cost + cache_cost) * 2.0 * 1.1)
    assert completion_cost == pytest.approx(500 * 25e-6 * 2.0 * 1.1)


@pytest.mark.parametrize(
    "model",
    ["claude-sonnet-4-6", "claude-mythos-5", "claude-mythos-preview"],
)
def test_anthropic_us_data_residency_uplift_on_claude_4_6_and_later_models(_local_model_cost_map, monkeypatch, model):
    """
    Anthropic bills every Claude 4.6+ model served with ``inference_geo="us"`` at
    1.1x, and echoes that geo back in the response usage, so each of these real
    cost-map entries has to carry the ``us`` multiplier or US-pinned traffic is
    under-reported by 10%.
    """
    from litellm.llms.anthropic.cost_calculation import (
        cost_per_token as anthropic_cost_per_token,
    )
    from litellm.types.utils import Usage

    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    def make_usage() -> "Usage":
        return Usage(prompt_tokens=1_000, completion_tokens=100, total_tokens=1_100)

    base_prompt_cost, base_completion_cost = anthropic_cost_per_token(model=model, usage=make_usage())

    geo_usage = make_usage()
    geo_usage.inference_geo = "us"
    geo_prompt_cost, geo_completion_cost = anthropic_cost_per_token(model=model, usage=geo_usage)

    assert base_prompt_cost > 0
    assert geo_prompt_cost == pytest.approx(base_prompt_cost * 1.1)
    assert geo_completion_cost == pytest.approx(base_completion_cost * 1.1)


def test_gemini_cache_tokens_details_no_negative_values():
    """
    Test for Issue #18750: Negative text_tokens with Gemini caching

    When using Gemini with explicit caching, the response includes cacheTokensDetails
    which breaks down cached tokens by modality. This test ensures that:
    1. text_tokens is never negative
    2. We correctly subtract cached tokens per modality (not total)
    """
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )

    # Scenario from issue #18750: Image + text with explicit caching
    # Real Gemini response structure when using cached content
    completion_response = {
        "usageMetadata": {
            "promptTokenCount": 9660,
            "candidatesTokenCount": 7,
            "totalTokenCount": 9667,
            "cachedContentTokenCount": 9651,
            # Total tokens by modality (includes cached + non-cached)
            "promptTokensDetails": [
                {"modality": "TEXT", "tokenCount": 9402},
                {"modality": "IMAGE", "tokenCount": 258},
            ],
            # Breakdown of cached tokens by modality
            "cacheTokensDetails": [
                {"modality": "TEXT", "tokenCount": 9393},
                {"modality": "IMAGE", "tokenCount": 258},
            ],
        }
    }

    usage = VertexGeminiConfig._calculate_usage(completion_response)

    # Text tokens should be non-cached text only: 9402 - 9393 = 9
    assert usage.prompt_tokens_details.text_tokens == 9, (
        f"Expected text_tokens=9, got {usage.prompt_tokens_details.text_tokens}"
    )

    # Image tokens should be non-cached image only: 258 - 258 = 0
    assert usage.prompt_tokens_details.image_tokens == 0, (
        f"Expected image_tokens=0, got {usage.prompt_tokens_details.image_tokens}"
    )

    # Total cached should match
    assert usage.prompt_tokens_details.cached_tokens == 9651, (
        f"Expected cached_tokens=9651, got {usage.prompt_tokens_details.cached_tokens}"
    )

    # MOST IMPORTANT: text_tokens should NEVER be negative
    assert usage.prompt_tokens_details.text_tokens >= 0, (
        f"BUG: text_tokens is negative ({usage.prompt_tokens_details.text_tokens})! This was the issue in #18750"
    )

    print("✅ Issue #18750 fix verified: text_tokens is correctly calculated and non-negative")


def test_gemini_without_cache_tokens_details():
    """
    Test Gemini response without cacheTokensDetails (implicit caching or no cache)

    When cacheTokensDetails is not present, we should use promptTokensDetails as-is
    without subtracting anything.
    """
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )

    completion_response = {
        "usageMetadata": {
            "promptTokenCount": 264,
            "candidatesTokenCount": 15,
            "totalTokenCount": 279,
            "promptTokensDetails": [
                {"modality": "TEXT", "tokenCount": 6},
                {"modality": "IMAGE", "tokenCount": 258},
            ],
            # No cacheTokensDetails
        }
    }

    usage = VertexGeminiConfig._calculate_usage(completion_response)

    # Should use promptTokensDetails values directly
    assert usage.prompt_tokens_details.text_tokens == 6
    assert usage.prompt_tokens_details.image_tokens == 258
    assert usage.prompt_tokens_details.text_tokens >= 0

    print("✅ Gemini without cacheTokensDetails works correctly")




def test_additional_costs_only_for_azure_ai(_local_model_cost_map):
    """
    Test that _get_additional_costs is only called for azure_ai provider.

    completion_cost() guards the call with `if custom_llm_provider == "azure_ai"`.
    This test verifies that non-azure_ai providers get additional_costs=None
    (reflected by the absence of "additional_costs" in cost_breakdown),
    while azure_ai providers can include additional costs.
    """
    from litellm.cost_calculator import _get_additional_costs

    # Non-azure_ai providers should return None
    result = _get_additional_costs(
        model="gpt-4o",
        custom_llm_provider="openai",
        prompt_tokens=100,
        completion_tokens=50,
    )
    assert result is None, "Non-azure_ai providers should have no additional costs"

    result = _get_additional_costs(
        model="claude-sonnet-4-20250514",
        custom_llm_provider="anthropic",
        prompt_tokens=100,
        completion_tokens=50,
    )
    assert result is None, "Anthropic should have no additional costs"

    result = _get_additional_costs(
        model="gemini-2.0-flash",
        custom_llm_provider="vertex_ai",
        prompt_tokens=100,
        completion_tokens=50,
    )
    assert result is None, "Vertex AI should have no additional costs"


def test_custom_pricing_applies_cache_read_input_cost():
    """
    Bug 1 reproduction: custom_cost_per_token with cache_read_input_token_cost
    should bill cached prompt tokens at the cache rate, not the full input rate.
    """
    usage = Usage(
        prompt_tokens=6074,
        completion_tokens=285,
        total_tokens=6359,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=3456,
            audio_tokens=0,
        ),
    )

    response = ModelResponse(
        id="test-id",
        created=1234567890,
        model="openai/gpt-5.4",
        object="chat.completion",
        choices=[],
        usage=usage,
    )

    cost = litellm.completion_cost(
        completion_response=response,
        model="openai/gpt-5.4",
        custom_llm_provider="openai",
        custom_cost_per_token={
            "input_cost_per_token": 0.0000025,
            "output_cost_per_token": 0.000015,
            "cache_read_input_token_cost": 0.00000025,
        },
    )

    expected = (6074 - 3456) * 0.0000025 + 3456 * 0.00000025 + 285 * 0.000015

    assert cost == pytest.approx(expected)


def test_custom_pricing_applies_cache_creation_input_cost_via_prompt_details():
    """
    OpenAI-compatible providers report cache-write tokens under
    prompt_tokens_details.cache_creation_tokens. The custom-pricing helper must
    bill those at cache_creation_input_token_cost, not the full input rate.
    """
    pt_details = PromptTokensDetailsWrapper(cached_tokens=1000, audio_tokens=0)
    pt_details.cache_creation_tokens = 500

    usage = Usage(
        prompt_tokens=4000,
        completion_tokens=100,
        total_tokens=4100,
        prompt_tokens_details=pt_details,
    )

    response = ModelResponse(
        id="test-id",
        created=1234567890,
        model="openai/gpt-5.4",
        object="chat.completion",
        choices=[],
        usage=usage,
    )

    cost = litellm.completion_cost(
        completion_response=response,
        model="openai/gpt-5.4",
        custom_llm_provider="openai",
        custom_cost_per_token={
            "input_cost_per_token": 0.0000025,
            "output_cost_per_token": 0.000015,
            "cache_read_input_token_cost": 0.00000025,
            "cache_creation_input_token_cost": 0.000003125,
        },
    )

    expected = (4000 - 1000 - 500) * 0.0000025 + 1000 * 0.00000025 + 500 * 0.000003125 + 100 * 0.000015

    assert cost == pytest.approx(expected)


def test_custom_pricing_applies_cache_creation_input_cost_via_cache_write_tokens_alias():
    """
    Some OpenAI-compatible providers (e.g. kimi-k2) emit cache-write tokens as
    `cache_write_tokens` rather than `cache_creation_tokens`. The cost
    calculator must mirror db_spend_update_writer and accept either name —
    otherwise daily aggregation counts the tokens but the per-request cost
    bills them at the full input rate.

    Drives `cost_per_token` directly with a SimpleNamespace usage stub so the
    `cache_write_tokens` alias survives the call (Pydantic's Usage init
    rebuilds prompt_tokens_details and drops dynamic attributes).
    """
    from types import SimpleNamespace

    from litellm.cost_calculator import cost_per_token

    pt_details = SimpleNamespace(cached_tokens=1000, cache_write_tokens=500)
    usage_stub = SimpleNamespace(
        prompt_tokens=4000,
        completion_tokens=100,
        total_tokens=4100,
        prompt_tokens_details=pt_details,
        cache_read_input_tokens=None,
        cache_creation_input_tokens=None,
    )

    prompt_cost, completion_cost = cost_per_token(
        model="moonshotai/kimi-k2",
        prompt_tokens=4000,
        completion_tokens=100,
        custom_llm_provider="openai",
        usage_object=usage_stub,
        custom_cost_per_token={
            "input_cost_per_token": 0.0000025,
            "output_cost_per_token": 0.000015,
            "cache_read_input_token_cost": 0.00000025,
            "cache_creation_input_token_cost": 0.000003125,
        },
    )

    expected_prompt = (4000 - 1000 - 500) * 0.0000025 + 1000 * 0.00000025 + 500 * 0.000003125
    expected_completion = 100 * 0.000015

    assert prompt_cost == pytest.approx(expected_prompt)
    assert completion_cost == pytest.approx(expected_completion)


# ---------------------------------------------------------------------------
# Bug 2 — db_spend_update_writer cache token extraction helpers.
# ---------------------------------------------------------------------------


def test_extract_cache_read_tokens_anthropic_top_level():
    from litellm.proxy.spend_tracking.savings import extract_cache_read_tokens as _extract_cache_read_tokens

    usage_obj = {
        "prompt_tokens": 100,
        "cache_read_input_tokens": 80,
        "prompt_tokens_details": {"cached_tokens": 80},
    }
    # Anthropic top-level value should win over prompt_tokens_details fallback.
    assert _extract_cache_read_tokens(usage_obj) == 80


def test_extract_cache_read_tokens_openai_compatible_fallback():
    from litellm.proxy.spend_tracking.savings import extract_cache_read_tokens as _extract_cache_read_tokens

    # Anthropic field absent — fall back to prompt_tokens_details.cached_tokens.
    usage_obj = {
        "prompt_tokens": 22583,
        "prompt_tokens_details": {"cached_tokens": 22016},
    }
    assert _extract_cache_read_tokens(usage_obj) == 22016


def test_extract_cache_read_tokens_zero_when_missing():
    from litellm.proxy.spend_tracking.savings import extract_cache_read_tokens as _extract_cache_read_tokens

    assert _extract_cache_read_tokens({}) == 0
    assert _extract_cache_read_tokens({"cache_read_input_tokens": None}) == 0
    assert _extract_cache_read_tokens({"prompt_tokens_details": {"cached_tokens": None}}) == 0


def test_extract_cache_creation_tokens_anthropic_top_level():
    from litellm.proxy.spend_tracking.savings import extract_cache_creation_tokens as _extract_cache_creation_tokens

    usage_obj = {
        "prompt_tokens": 100,
        "cache_creation_input_tokens": 50,
        "prompt_tokens_details": {"cache_write_tokens": 50},
    }
    # Anthropic top-level should short-circuit the fallback.
    assert _extract_cache_creation_tokens(usage_obj) == 50


def test_extract_cache_creation_tokens_openai_cache_write_alias():
    from litellm.proxy.spend_tracking.savings import extract_cache_creation_tokens as _extract_cache_creation_tokens

    # kimi-k2 emits cache_write_tokens.
    usage_obj = {
        "prompt_tokens": 1000,
        "prompt_tokens_details": {"cache_write_tokens": 200},
    }
    assert _extract_cache_creation_tokens(usage_obj) == 200


def test_extract_cache_creation_tokens_openai_cache_creation_alias():
    from litellm.proxy.spend_tracking.savings import extract_cache_creation_tokens as _extract_cache_creation_tokens

    # Other OpenAI-compatible providers emit cache_creation_tokens.
    usage_obj = {
        "prompt_tokens": 1000,
        "prompt_tokens_details": {"cache_creation_tokens": 300},
    }
    assert _extract_cache_creation_tokens(usage_obj) == 300


def test_extract_cache_creation_tokens_zero_when_missing():
    from litellm.proxy.spend_tracking.savings import extract_cache_creation_tokens as _extract_cache_creation_tokens

    assert _extract_cache_creation_tokens({}) == 0
    assert _extract_cache_creation_tokens({"cache_creation_input_tokens": None}) == 0
    assert _extract_cache_creation_tokens({"prompt_tokens_details": {"cache_write_tokens": None}}) == 0


def test_custom_pricing_anthropic_style_cache_tokens_not_double_counted():
    """
    Anthropic providers report cache tokens at the top level of Usage, and
    `prompt_tokens` EXCLUDES them. The helper expects `prompt_tokens` to
    include cache tokens, so cost_per_token must adjust before invoking it —
    otherwise regular_prompt_tokens goes negative and clamps to 0.
    """
    usage = Usage(
        prompt_tokens=2000,
        completion_tokens=100,
        total_tokens=2100,
        cache_read_input_tokens=1500,
        cache_creation_input_tokens=300,
    )

    response = ModelResponse(
        id="test-id",
        created=1234567890,
        model="anthropic/claude-3-5-sonnet",
        object="chat.completion",
        choices=[],
        usage=usage,
    )

    cost = litellm.completion_cost(
        completion_response=response,
        model="anthropic/claude-3-5-sonnet",
        custom_llm_provider="anthropic",
        custom_cost_per_token={
            "input_cost_per_token": 0.000003,
            "output_cost_per_token": 0.000015,
            "cache_read_input_token_cost": 0.0000003,
            "cache_creation_input_token_cost": 0.00000375,
        },
    )

    # Anthropic prompt_tokens=2000 excludes cache. After normalization the
    # helper sees 2000 + 1500 + 300 = 3800, of which 2000 are uncached.
    expected = 2000 * 0.000003 + 1500 * 0.0000003 + 300 * 0.00000375 + 100 * 0.000015

    assert cost == pytest.approx(expected)


def test_custom_pricing_without_cache_keys_preserves_legacy_behavior():
    """
    Backward compatibility: when custom_cost_per_token omits both cache rates,
    cached tokens must be billed at input_cost_per_token (matching the pre-fix
    behavior) so existing callers see no change.
    """
    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=100,
        total_tokens=1100,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=400,
            audio_tokens=0,
        ),
    )

    response = ModelResponse(
        id="test-id",
        created=1234567890,
        model="openai/gpt-5.4",
        object="chat.completion",
        choices=[],
        usage=usage,
    )

    cost = litellm.completion_cost(
        completion_response=response,
        model="openai/gpt-5.4",
        custom_llm_provider="openai",
        custom_cost_per_token={
            "input_cost_per_token": 0.0000025,
            "output_cost_per_token": 0.000015,
        },
    )

    # All 1000 prompt tokens billed at input rate, regardless of cached_tokens.
    expected = 1000 * 0.0000025 + 100 * 0.000015

    assert cost == pytest.approx(expected)


def test_completion_cost_logs_the_rates_it_billed_at(monkeypatch):
    """A caller reporting the cost lines beside their per-token rates reads both off this one call.
    completion_cost infers the provider, and xai's inclusive tier thresholds put a request sitting
    exactly on 200k at the tier rate, which a lookup made without that inferred provider would miss.
    """
    from datetime import datetime

    from litellm.litellm_core_utils.litellm_logging import Logging

    monkeypatch.setitem(
        litellm.model_cost,
        "xai/tiered-model",
        {
            "input_cost_per_token": 3e-6,
            "output_cost_per_token": 15e-6,
            "cache_read_input_token_cost": 3e-7,
            "input_cost_per_token_above_200k_tokens": 6e-6,
            "output_cost_per_token_above_200k_tokens": 3e-5,
            "cache_read_input_token_cost_above_200k_tokens": 6e-7,
            "litellm_provider": "xai",
            "mode": "chat",
        },
    )
    logging_obj = Logging(
        model="xai/tiered-model",
        messages=[{"role": "user", "content": "Hello"}],
        stream=False,
        call_type="completion",
        start_time=datetime.now(),
        litellm_call_id="billed-rates",
        function_id="f",
    )
    usage = Usage(
        prompt_tokens=200_000,
        completion_tokens=1_000,
        total_tokens=201_000,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=100_000),
    )

    litellm.completion_cost(
        completion_response=ModelResponse(model="xai/tiered-model", usage=usage),
        model="xai/tiered-model",
        custom_llm_provider=None,
        litellm_logging_obj=logging_obj,
    )

    rates = logging_obj.billed_token_rates
    assert rates is not None
    assert rates.input_cost_per_token == pytest.approx(6e-6)
    assert rates.cache_read_input_token_cost == pytest.approx(6e-7)
    assert logging_obj.cost_breakdown["cache_read_cost"] == pytest.approx(100_000 * rates.cache_read_input_token_cost)
    assert logging_obj.cost_breakdown["output_cost"] == pytest.approx(1_000 * rates.output_cost_per_token)


def test_completion_cost_logs_cache_and_reasoning_breakdown_for_custom_pricing():
    """
    A custom-priced deployment bills cache tokens at its custom cache rates, but the
    breakdown stored for the spend logs carried no cache or reasoning lines for it.
    """
    from datetime import datetime

    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.types.utils import CompletionTokensDetailsWrapper, CostPerToken

    logging_obj = Logging(
        model="openai/onprem-model",
        messages=[{"role": "user", "content": "Hello"}],
        stream=False,
        call_type="completion",
        start_time=datetime.now(),
        litellm_call_id="custom-pricing-breakdown",
        function_id="f",
    )
    response = ModelResponse(
        model="openai/onprem-model",
        usage=Usage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=800, cache_creation_tokens=100),
            completion_tokens_details=CompletionTokensDetailsWrapper(reasoning_tokens=200),
        ),
    )

    total = completion_cost(
        completion_response=response,
        model="openai/onprem-model",
        custom_llm_provider="openai",
        custom_cost_per_token=CostPerToken(
            input_cost_per_token=1e-6,
            output_cost_per_token=2e-6,
            cache_read_input_token_cost=1e-7,
            cache_creation_input_token_cost=1.25e-6,
        ),
        litellm_logging_obj=logging_obj,
    )

    assert logging_obj.cost_breakdown is not None
    assert logging_obj.cost_breakdown["cache_read_cost"] == pytest.approx(800 * 1e-7)
    assert logging_obj.cost_breakdown["cache_creation_cost"] == pytest.approx(100 * 1.25e-6)
    assert logging_obj.cost_breakdown["reasoning_cost"] == pytest.approx(200 * 2e-6)
    assert total == pytest.approx(100 * 1e-6 + 800 * 1e-7 + 100 * 1.25e-6 + 500 * 2e-6)


@pytest.mark.parametrize("custom_llm_provider", ["together_ai", "openai", "anthropic", "bedrock", "azure"])
def test_cost_per_token_per_second_pricing(monkeypatch, custom_llm_provider: str):
    """
    Models priced by duration (input/output_cost_per_second) with no per-token rates
    must be billed as cost_per_second * response_time_ms / 1000 in cost_per_token,
    whether or not the provider has its own cost calculator.
    """
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))

    model = f"test-per-second-pricing-{custom_llm_provider}"
    litellm.register_model(
        model_cost={
            model: {
                "input_cost_per_second": 0.02,
                "output_cost_per_second": 0.04,
                "litellm_provider": custom_llm_provider,
                "mode": "chat",
            }
        }
    )

    prompt_cost, completion_cost_value = cost_per_token(
        model=model,
        custom_llm_provider=custom_llm_provider,
        prompt_tokens=10,
        completion_tokens=20,
        response_time_ms=1500.0,
    )

    assert prompt_cost == pytest.approx(0.02 * 1.5)
    assert completion_cost_value == pytest.approx(0.04 * 1.5)


def test_cost_per_token_keeps_token_pricing_when_per_second_rates_are_also_set(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))

    model = "test-token-and-per-second-pricing-model"
    litellm.register_model(
        model_cost={
            model: {
                "input_cost_per_token": 1e-6,
                "output_cost_per_token": 2e-6,
                "input_cost_per_second": 0.02,
                "output_cost_per_second": 0.04,
                "litellm_provider": "openai",
                "mode": "chat",
            }
        }
    )

    prompt_cost, completion_cost_value = cost_per_token(
        model=model,
        custom_llm_provider="openai",
        prompt_tokens=10,
        completion_tokens=20,
        response_time_ms=1500.0,
    )

    assert prompt_cost == pytest.approx(10 * 1e-6)
    assert completion_cost_value == pytest.approx(20 * 2e-6)


def _logging_obj_with_call_window(duration_ms: float) -> Logging:
    start_time: Final = datetime.datetime(2026, 9, 21, 12, 0, 0)
    logging_obj: Final = Logging(
        model="gpt-5.4-nano",
        messages=[{"role": "user", "content": "Hello"}],
        stream=False,
        call_type="completion",
        start_time=start_time,
        litellm_call_id="per-second-call-window",
        function_id="f",
    )
    logging_obj.model_call_details["start_time"] = start_time
    logging_obj.model_call_details["end_time"] = start_time + datetime.timedelta(milliseconds=duration_ms)
    return logging_obj


@pytest.mark.parametrize(
    ("stamped_response_ms", "total_time", "logged_duration_ms", "expected_seconds"),
    [(None, 0.0, 1500.0, 1.5), (3000.0, 0.0, 1500.0, 3.0), (None, 2500.0, 1500.0, 2.5), (3000.0, 2500.0, 1500.0, 3.0)],
)
def test_completion_cost_per_second_deployment_bills_the_call_duration(
    monkeypatch,
    stamped_response_ms: float | None,
    total_time: float,
    logged_duration_ms: float,
    expected_seconds: float,
):
    """
    A deployment priced only per second bills the stamped ``_response_ms`` when there is one,
    then the caller's explicit ``total_time``, and the logging object's start/end window otherwise
    (a streamed response is never stamped).
    """
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))

    deployment_id = "per-second-openai-deployment"
    litellm.register_model(
        model_cost={
            deployment_id: {
                "input_cost_per_second": 0.02,
                "output_cost_per_second": 0.04,
                "litellm_provider": "openai",
                "mode": "chat",
            }
        }
    )
    response = ModelResponse(
        model="gpt-5.4-nano",
        usage=Usage(prompt_tokens=11, completion_tokens=7, total_tokens=18),
    )
    response._response_ms = stamped_response_ms

    cost = completion_cost(
        completion_response=response,
        model="openai/gpt-5.4-nano",
        custom_llm_provider="openai",
        custom_pricing=True,
        router_model_id=deployment_id,
        total_time=total_time,
        litellm_logging_obj=_logging_obj_with_call_window(logged_duration_ms),
    )

    assert cost == pytest.approx((0.02 + 0.04) * expected_seconds)


@pytest.mark.parametrize("mode", ["audio_transcription", "audio_speech", "video_generation", "realtime"])
def test_cost_per_token_leaves_media_second_rates_to_their_dedicated_paths(monkeypatch, mode: str):
    """
    A media-mode entry's per-second rates price audio or video seconds, which the dedicated
    transcription, speech, video, and realtime paths bill from the media itself, so a call that
    reaches the generic path with only a wall-clock duration must not bill them.
    """
    model = f"test-media-per-second-{mode}"
    monkeypatch.setitem(
        litellm.model_cost,
        model,
        {"input_cost_per_second": 0.02, "output_cost_per_second": 0.4, "litellm_provider": "openai", "mode": mode},
    )

    assert cost_per_token(model=model, custom_llm_provider="openai", response_time_ms=2000.0) == (0.0, 0.0)


def test_completion_cost_video_status_poll_bills_nothing_on_a_per_second_video_model(monkeypatch):
    """
    Polling a video job returns a ``VideoObject`` with no stamped duration, so the cost path falls
    back to the logging object's call window; on a video model priced per output second that
    window must not be billed, or every status poll would charge for the seconds it took to answer.
    """
    model = "test-veo-per-second-poll"
    monkeypatch.setitem(
        litellm.model_cost,
        model,
        {"output_cost_per_second": 0.4, "litellm_provider": "vertex_ai", "mode": "video_generation"},
    )
    video = VideoObject(id="video_1", object="video", status="completed", model=model, progress=100)

    cost = completion_cost(
        completion_response=video,
        model=model,
        custom_llm_provider="vertex_ai",
        call_type=CallTypes.video_retrieve.value,
        litellm_logging_obj=_logging_obj_with_call_window(2000.0),
    )

    assert cost == 0.0


def _batch_cache_usage() -> Usage:
    return Usage(
        prompt_tokens=11000,
        completion_tokens=200,
        total_tokens=11200,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=8000,
            cache_creation_tokens=2000,
            text_tokens=1000,
        ),
        cache_creation_input_tokens=2000,
        cache_read_input_tokens=8000,
    )


def test_batch_cost_calculator_prices_multimodal_tokens_at_modality_rates():
    from litellm.cost_calculator import batch_cost_calculator

    model_info: ModelInfo = {
        "input_cost_per_token_batches": 1e-7,
        "input_cost_per_audio_token_batches": 3.25e-6,
        "input_cost_per_image_token_batches": 2.25e-7,
        "input_cost_per_video_token_batches": 6e-6,
    }
    usage = Usage(
        prompt_tokens=100,
        completion_tokens=0,
        total_tokens=100,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=64,
            image_tokens=10,
            video_tokens=6,
        ),
    )

    prompt_cost, _ = batch_cost_calculator(
        usage=usage,
        model="gemini-embedding-2",
        custom_llm_provider="vertex_ai",
        model_info=model_info,
    )

    assert prompt_cost == pytest.approx(20 * 1e-7 + 64 * 3.25e-6 + 10 * 2.25e-7 + 6 * 6e-6)


def test_batch_cost_calculator_falls_back_to_text_batch_rate_for_modalities():
    from litellm.cost_calculator import batch_cost_calculator

    model_info: ModelInfo = {"input_cost_per_token_batches": 1e-7}
    usage = Usage(
        prompt_tokens=100,
        completion_tokens=0,
        total_tokens=100,
        prompt_tokens_details=PromptTokensDetailsWrapper(audio_tokens=64),
    )

    prompt_cost, _ = batch_cost_calculator(
        usage=usage,
        model="gemini-embedding-2",
        custom_llm_provider="vertex_ai",
        model_info=model_info,
    )

    assert prompt_cost == pytest.approx(100 * 1e-7)


def test_batch_cost_calculator_prices_cache_creation_tokens_at_cache_write_rate():
    """
    LIT-4008 regression: anthropic batch usage is dominated by cache tokens.
    Cache creation tokens must be priced at cache_creation_input_token_cost / 2,
    not folded into the base input rate, and must not also be billed as base
    input tokens.
    """
    from litellm.cost_calculator import batch_cost_calculator

    model_info: ModelInfo = {
        "supported_openai_params": [],
        "input_cost_per_token": 3e-6,
        "output_cost_per_token": 15e-6,
        "cache_read_input_token_cost": 3e-7,
        "cache_creation_input_token_cost": 3.75e-6,
    }
    prompt_cost, completion_cost_value = batch_cost_calculator(
        usage=_batch_cache_usage(),
        model="claude-sonnet-4-5-20250929",
        custom_llm_provider="anthropic",
        model_info=model_info,
    )

    assert prompt_cost == pytest.approx((1000 * 3e-6 + 8000 * 3e-7 + 2000 * 3.75e-6) / 2)
    assert completion_cost_value == pytest.approx(200 * 15e-6 / 2)


def test_batch_cost_calculator_cache_creation_falls_back_to_input_rate():
    from litellm.cost_calculator import batch_cost_calculator

    model_info: ModelInfo = {
        "supported_openai_params": [],
        "input_cost_per_token": 3e-6,
        "output_cost_per_token": 15e-6,
        "cache_read_input_token_cost": 3e-7,
    }
    prompt_cost, _ = batch_cost_calculator(
        usage=_batch_cache_usage(),
        model="claude-sonnet-4-5-20250929",
        custom_llm_provider="anthropic",
        model_info=model_info,
    )

    assert prompt_cost == pytest.approx((1000 * 3e-6 + 8000 * 3e-7 + 2000 * 3e-6) / 2)


def test_completion_cost_bills_interactions_api_response():
    from litellm.types.interactions import InteractionsAPIResponse

    model_info = litellm.get_model_info(model="gemini-2.5-flash", custom_llm_provider="gemini")
    response = InteractionsAPIResponse(
        id="interactions/abc123",
        model="gemini-2.5-flash",
        status="completed",
        steps=[],
        usage={
            "total_tokens": 175,
            "total_input_tokens": 100,
            "input_tokens_by_modality": [{"modality": "text", "tokens": 100}],
            "total_cached_tokens": 0,
            "total_output_tokens": 50,
            "output_tokens_by_modality": [{"modality": "text", "tokens": 50}],
            "total_tool_use_tokens": 0,
            "total_thought_tokens": 25,
        },
    )

    cost = completion_cost(completion_response=response, custom_llm_provider="gemini")

    reasoning_rate = model_info.get("output_cost_per_reasoning_token") or model_info["output_cost_per_token"]
    expected = 100 * model_info["input_cost_per_token"] + 50 * model_info["output_cost_per_token"] + 25 * reasoning_rate
    assert cost == pytest.approx(expected)
    assert cost > 0


def test_completion_cost_bills_interactions_google_search_per_query():
    from litellm.types.interactions import InteractionsAPIResponse

    model_info = litellm.get_model_info(model="gemini-3-flash-preview", custom_llm_provider="gemini")
    response = InteractionsAPIResponse(
        id="interactions/search123",
        model="gemini-3-flash-preview",
        status="completed",
        steps=[],
        usage={
            "total_tokens": 680,
            "total_input_tokens": 103,
            "input_tokens_by_modality": [{"modality": "text", "tokens": 103}],
            "total_cached_tokens": 0,
            "total_output_tokens": 226,
            "total_tool_use_tokens": 0,
            "total_thought_tokens": 351,
            "grounding_tool_count": [{"type": "google_search", "count": 3}],
        },
    )

    cost = completion_cost(completion_response=response, custom_llm_provider="gemini")

    per_query_cost = model_info["search_context_cost_per_query"]["search_context_size_medium"]
    reasoning_rate = model_info.get("output_cost_per_reasoning_token") or model_info["output_cost_per_token"]
    expected = (
        103 * model_info["input_cost_per_token"]
        + 226 * model_info["output_cost_per_token"]
        + 351 * reasoning_rate
        + 3 * per_query_cost
    )
    assert model_info.get("web_search_billing_unit") == "per_query"
    assert cost == pytest.approx(expected)
    assert cost > 3 * per_query_cost


def test_completion_cost_bills_interactions_video_output_at_video_rate():
    from litellm.types.interactions import InteractionsAPIResponse

    model_info = litellm.get_model_info(model="gemini-omni-flash-preview", custom_llm_provider="gemini")
    video_tokens = 5792 * 8
    response = InteractionsAPIResponse(
        id="interactions/video123",
        model="gemini-omni-flash-preview",
        status="completed",
        steps=[],
        usage={
            "total_tokens": 10 + video_tokens,
            "total_input_tokens": 10,
            "input_tokens_by_modality": [{"modality": "text", "tokens": 10}],
            "total_cached_tokens": 0,
            "total_output_tokens": video_tokens,
            "output_tokens_by_modality": [{"modality": "video", "tokens": video_tokens}],
            "total_tool_use_tokens": 0,
            "total_thought_tokens": 0,
        },
    )

    cost = completion_cost(completion_response=response, custom_llm_provider="gemini")

    expected = 10 * model_info["input_cost_per_token"] + video_tokens * model_info["output_cost_per_video_token"]
    assert model_info["output_cost_per_video_token"] != model_info["output_cost_per_token"]
    assert cost == pytest.approx(expected)


@pytest.mark.parametrize("video_count", [2, 3])
def test_completion_cost_multiplies_video_cost_by_generated_video_count(video_count: int) -> None:
    """Regression for LIT-6896: a Veo request for N samples generates N videos and must be billed N times."""
    from litellm.types.videos.main import VideoObject

    def _video(usage: dict[str, object]) -> VideoObject:
        return VideoObject(id="v", object="video", status="processing", model="veo-3.1-fast-generate-001", usage=usage)

    single_cost = completion_cost(
        completion_response=_video({"duration_seconds": 4.0, "video_resolution": "720p"}),
        model="veo-3.1-fast-generate-001",
        custom_llm_provider="vertex_ai",
        call_type="create_video",
    )
    multi_cost = completion_cost(
        completion_response=_video({"duration_seconds": 4.0, "video_resolution": "720p", "video_count": video_count}),
        model="veo-3.1-fast-generate-001",
        custom_llm_provider="vertex_ai",
        call_type="create_video",
    )

    assert single_cost > 0
    assert multi_cost == pytest.approx(single_cost * video_count)


@pytest.mark.parametrize(
    "batch_rate,expected_prompt,expected_completion",
    [
        (0.0, 0.0, 0.0),
        (1e-6, 1000 * 1e-6, 500 * 1e-6),
        (None, 1000 * 3e-6 / 2, 500 * 15e-6 / 2),
    ],
    ids=["explicit-zero", "explicit-nonzero", "unset"],
)
def test_batch_cost_calculator_honors_an_explicitly_zero_batch_rate(
    batch_rate: float | None,
    expected_prompt: float,
    expected_completion: float,
) -> None:
    """A batch rate configured as 0.0 means free, not unset.

    Gating the batch fields on truthiness read an explicit 0.0 as absent and
    charged half the standard rate for that token direction instead.
    """
    from litellm.cost_calculator import batch_cost_calculator

    base_model_info: ModelInfo = {
        "supported_openai_params": [],
        "input_cost_per_token": 3e-6,
        "output_cost_per_token": 15e-6,
    }
    model_info: ModelInfo = (
        base_model_info
        if batch_rate is None
        else {
            **base_model_info,
            "input_cost_per_token_batches": batch_rate,
            "output_cost_per_token_batches": batch_rate,
        }
    )

    prompt_cost, completion_cost_value = batch_cost_calculator(
        usage=Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
        model="claude-sonnet-4-5-20250929",
        custom_llm_provider="anthropic",
        model_info=model_info,
    )

    assert prompt_cost == pytest.approx(expected_prompt)
    assert completion_cost_value == pytest.approx(expected_completion)


def test_combine_usage_objects_sums_mirrored_cache_write_fields_once():
    """
    cache_write_tokens and cache_creation_tokens mirror each other on
    PromptTokensDetailsWrapper, so field-iterating aggregation must sum the pair
    once: a single 50-token usage stays 50 and two combine to 100, not double.
    """
    single = Usage(
        prompt_tokens=100,
        completion_tokens=10,
        total_tokens=110,
        prompt_tokens_details=PromptTokensDetailsWrapper(cache_write_tokens=50),
    )
    combined = BaseTokenUsageProcessor.combine_usage_objects([single])
    assert combined.prompt_tokens_details is not None
    assert combined.prompt_tokens_details.cache_write_tokens == 50
    assert combined.prompt_tokens_details.cache_creation_tokens == 50

    anthropic_style = Usage(
        prompt_tokens=100,
        completion_tokens=10,
        total_tokens=110,
        cache_creation_input_tokens=50,
    )
    combined_pair = BaseTokenUsageProcessor.combine_usage_objects([anthropic_style, anthropic_style])
    assert combined_pair.prompt_tokens_details is not None
    assert combined_pair.prompt_tokens_details.cache_write_tokens == 100
    assert combined_pair.prompt_tokens_details.cache_creation_tokens == 100


def test_select_model_name_strips_unregistered_alias_prefix(_local_model_cost_map):
    """A router-facing model_name alias containing "/" whose leading segment is NOT a
    registered provider must not be double-prefixed into a non-existent cost key.

    Regression test for #38069: alias "vertex/claude-opus-5" (real deployment
    "vertex_ai/claude-opus-5") was re-prefixed into "vertex_ai/vertex/claude-opus-5",
    silently pricing every streamed request at $0.
    """

    from litellm.cost_calculator import _select_model_name_for_cost_calc

    response = litellm.ModelResponse(
        id="x",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
        model="vertex/claude-opus-5",
    )
    response._hidden_params = {}

    selected = _select_model_name_for_cost_calc(
        model=None,
        completion_response=response,
        custom_llm_provider="vertex_ai",
    )

    assert selected == "vertex_ai/claude-opus-5"


def test_select_model_name_strips_duplicated_region_segment(_local_model_cost_map):
    """A "region/model" alias whose leading segment repeats the request's region must
    resolve to the region-priced cost key instead of keeping the region segment twice."""

    from litellm.cost_calculator import _select_model_name_for_cost_calc

    response = litellm.ModelResponse(
        id="x",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
        model="us-east-1/anthropic.claude-v2:1",
    )
    response._hidden_params = {"region_name": "us-east-1"}

    selected = _select_model_name_for_cost_calc(
        model=None,
        completion_response=response,
        custom_llm_provider="bedrock",
    )

    assert selected == "bedrock/us-east-1/anthropic.claude-v2:1"


def _bedrock_response_with_private_model(model: str, region_name: str) -> litellm.ModelResponse:
    response = litellm.ModelResponse(
        id="x",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
        model=model,
    )
    response._hidden_params = {"provider_response_model": model, "region_name": region_name}
    return response


def test_select_model_name_applies_region_to_private_provider_response_model(_local_model_cost_map):
    """A Bedrock stream carries its requested model as the private provider model and must keep the
    request's region in the cost key, exactly as the same request does without streaming."""

    from litellm.cost_calculator import _select_model_name_for_cost_calc

    selected = _select_model_name_for_cost_calc(
        model=None,
        completion_response=_bedrock_response_with_private_model("anthropic.claude-v2:1", "us-east-1"),
        custom_llm_provider="bedrock",
    )

    assert selected == "bedrock/us-east-1/anthropic.claude-v2:1"


def test_completion_cost_region_name_prices_mantle_on_the_regional_row(_local_model_cost_map):
    """completion_cost(region_name=...) must price a Bedrock Mantle call from the
    bedrock_mantle/<region>/<model> row when one exists, for the bare and the provider-prefixed
    model alike, and keep the flat row for regions without their own row."""

    response = litellm.ModelResponse(
        id="x",
        choices=[{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        model="xai.grok-4.3",
        usage={"prompt_tokens": 38, "completion_tokens": 20, "total_tokens": 58},
    )
    gov = litellm.model_cost["bedrock_mantle/us-gov-west-1/xai.grok-4.3"]
    flat = litellm.model_cost["bedrock_mantle/xai.grok-4.3"]
    expected_gov = 38 * gov["input_cost_per_token"] + 20 * gov["output_cost_per_token"]
    expected_flat = 38 * flat["input_cost_per_token"] + 20 * flat["output_cost_per_token"]
    assert expected_gov != expected_flat

    for model in ("xai.grok-4.3", "bedrock_mantle/xai.grok-4.3"):
        assert litellm.completion_cost(
            completion_response=response,
            model=model,
            custom_llm_provider="bedrock_mantle",
            region_name="us-gov-west-1",
        ) == pytest.approx(expected_gov)
        assert litellm.completion_cost(
            completion_response=response,
            model=model,
            custom_llm_provider="bedrock_mantle",
            region_name="eu-west-1",
        ) == pytest.approx(expected_flat)
    assert litellm.completion_cost(
        completion_response=response, model="xai.grok-4.3", custom_llm_provider="bedrock_mantle"
    ) == pytest.approx(expected_flat)


def test_cost_per_token_region_name_applies_to_provider_prefixed_model(_local_model_cost_map):
    """A provider-prefixed model must still find its bedrock_mantle/<region>/<model> row instead of
    composing the region key with the provider segment twice."""

    prompt_cost, completion_cost = litellm.cost_per_token(
        model="bedrock_mantle/xai.grok-4.3",
        prompt_tokens=38,
        completion_tokens=20,
        custom_llm_provider="bedrock_mantle",
        region_name="us-gov-west-1",
    )
    gov = litellm.model_cost["bedrock_mantle/us-gov-west-1/xai.grok-4.3"]

    assert prompt_cost + completion_cost == pytest.approx(
        38 * gov["input_cost_per_token"] + 20 * gov["output_cost_per_token"]
    )


def test_completion_cost_mantle_native_messages_prices_claude_from_the_bedrock_row(_local_model_cost_map):
    """Mantle's native Messages API answers with Anthropic's canonical model name and the proxy
    resolves a Mantle region for every call, so the first cost candidate is
    bedrock_mantle/<region>/claude-sonnet-5. That name has no row of its own and must fall through to
    the deployment's bare Bedrock row instead of stopping on an unpriced capability rule at $0."""

    response = litellm.ModelResponse(
        id="msg_x",
        choices=[{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        model="claude-sonnet-5",
        usage={"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
    )
    row = litellm.model_cost["anthropic.claude-sonnet-5"]
    expected = 100 * row["input_cost_per_token"] + 10 * row["output_cost_per_token"]
    assert expected > 0

    for region_name in ("us-east-1", None):
        assert litellm.completion_cost(
            completion_response=response,
            model="bedrock_mantle/anthropic.claude-sonnet-5",
            custom_llm_provider="bedrock_mantle",
            region_name=region_name,
        ) == pytest.approx(expected)


def test_completion_cost_mantle_native_messages_prices_haiku_from_the_mantle_row(_local_model_cost_map):
    """Mantle serves Anthropic's un-versioned haiku id, which has no bare Bedrock row (Bedrock's carries
    the -20251001-v1:0 suffix), and Claude Code sends every small-fast-model call to it. Both the plain
    and the region-prefixed deployment names must price from bedrock_mantle/anthropic.claude-haiku-4-5
    instead of billing $0."""

    response = litellm.ModelResponse(
        id="msg_x",
        choices=[{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        model="claude-haiku-4-5",
        usage={"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
    )
    row = litellm.model_cost["bedrock_mantle/anthropic.claude-haiku-4-5"]
    expected = 100 * row["input_cost_per_token"] + 10 * row["output_cost_per_token"]
    assert expected > 0

    for model in (
        "bedrock_mantle/anthropic.claude-haiku-4-5",
        "bedrock_mantle/us-east-2/anthropic.claude-haiku-4-5",
    ):
        assert litellm.completion_cost(
            completion_response=response,
            model=model,
            custom_llm_provider="bedrock_mantle",
        ) == pytest.approx(expected), model


def test_completion_cost_legacy_mantle_route_prices_after_router_registration(local_model_cost_map):
    """The proxy registers every deployment under its provider-prefixed key at boot. A
    bedrock/mantle/<model> deployment must resolve to the bare Bedrock row there, otherwise the boot
    entry is a cost-less capability rule that shadows the priced row and every call on the deployment,
    /v1/chat/completions and /v1/messages alike, bills $0."""
    from litellm import Router

    Router(
        model_list=[
            {
                "model_name": "claude-sonnet-5",
                "litellm_params": {
                    "model": "bedrock/mantle/anthropic.claude-sonnet-5",
                    "aws_region_name": "us-east-1",
                },
            }
        ]
    )
    assert "bedrock/mantle/anthropic.claude-sonnet-5" not in litellm.model_cost

    response = litellm.ModelResponse(
        id="msg_x",
        choices=[{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        model="claude-sonnet-5",
        usage={"prompt_tokens": 16, "completion_tokens": 4, "total_tokens": 20},
    )
    row = litellm.model_cost["anthropic.claude-sonnet-5"]
    expected = 16 * row["input_cost_per_token"] + 4 * row["output_cost_per_token"]
    assert expected > 0

    for call_type in ("completion", "anthropic_messages"):
        assert litellm.completion_cost(
            completion_response=response,
            model="mantle/anthropic.claude-sonnet-5",
            custom_llm_provider="bedrock",
            call_type=call_type,
        ) == pytest.approx(expected), call_type


def test_select_model_name_keeps_base_model_free_of_region(_local_model_cost_map):
    """An explicit base_model keeps pricing on that model's own key even when the request carries a
    region with different regional rates, so the private provider model never widens region pricing."""

    from litellm.cost_calculator import _select_model_name_for_cost_calc

    selected = _select_model_name_for_cost_calc(
        model="my-bedrock-deployment",
        completion_response=_bedrock_response_with_private_model("moonshotai.kimi-k2.5", "ap-northeast-1"),
        base_model="moonshotai.kimi-k2.5",
        custom_llm_provider="bedrock",
    )

    assert selected == "bedrock/moonshotai.kimi-k2.5"


def test_completion_cost_base_model_ignores_regional_row(_local_model_cost_map):
    """A deployment with base_model set is priced from that model's own row even when the response
    carries a region whose regional row charges different rates."""

    response = litellm.ModelResponse(
        id="x",
        choices=[{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        model="my-bedrock-deployment",
        usage={"prompt_tokens": 1000, "completion_tokens": 0, "total_tokens": 1000},
    )
    response._hidden_params = {"custom_llm_provider": "bedrock", "region_name": "eu-central-1"}
    flat = litellm.model_cost["anthropic.claude-instant-v1"]
    regional = litellm.model_cost["bedrock/eu-central-1/anthropic.claude-instant-v1"]
    assert flat["input_cost_per_token"] != regional["input_cost_per_token"]

    assert litellm.completion_cost(
        completion_response=response,
        model="my-bedrock-deployment",
        custom_llm_provider="bedrock",
        base_model="anthropic.claude-instant-v1",
    ) == pytest.approx(1000 * flat["input_cost_per_token"])


def test_select_model_name_unresolvable_alias_unchanged(_local_model_cost_map):
    """An alias that resolves to no known cost key keeps the legacy double-prefixed name."""

    from litellm.cost_calculator import _select_model_name_for_cost_calc

    response = litellm.ModelResponse(
        id="x",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
        model="team/nonsense-model",
    )
    response._hidden_params = {}

    selected = _select_model_name_for_cost_calc(
        model=None,
        completion_response=response,
        custom_llm_provider="vertex_ai",
    )

    assert selected == "vertex_ai/team/nonsense-model"


def test_completion_cost_keeps_custom_priced_slash_router_id(_local_model_cost_map):
    """A custom-priced router id containing "/" keeps its custom pricing instead of being
    rewritten to the built-in key its suffix happens to match."""

    from litellm.cost_calculator import _select_model_name_for_cost_calc

    litellm.register_model(
        model_cost={
            "vertex/claude-opus-5": {
                "input_cost_per_token": 7e-6,
                "output_cost_per_token": 8e-6,
                "litellm_provider": "vertex_ai",
            }
        }
    )

    selected = _select_model_name_for_cost_calc(
        model="vertex_ai/claude-opus-5",
        completion_response=None,
        custom_pricing=True,
        custom_llm_provider="vertex_ai",
        router_model_id="vertex/claude-opus-5",
    )
    assert selected == "vertex_ai/vertex/claude-opus-5"

    response = litellm.ModelResponse(
        id="x",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
        model="vertex/claude-opus-5",
    )
    response._hidden_params = {"custom_llm_provider": "vertex_ai"}
    response.usage = litellm.Usage(prompt_tokens=100, completion_tokens=50)

    cost = litellm.completion_cost(
        completion_response=response,
        custom_llm_provider="vertex_ai",
        custom_pricing=True,
        router_model_id="vertex/claude-opus-5",
    )
    assert cost == pytest.approx(100 * 7e-6 + 50 * 8e-6, rel=1e-9)


@pytest.mark.parametrize(
    "priceless_entry",
    [
        {"litellm_provider": "vertex_ai", "mode": "realtime"},
        {
            "litellm_provider": "vertex_ai",
            "mode": "realtime",
            "input_cost_per_token": None,
            "output_cost_per_token": None,
            "input_cost_per_audio_token": None,
        },
    ],
    ids=["registered_without_price_fields", "registered_with_none_valued_price_fields"],
)
def test_realtime_priceless_deployment_entry_falls_through_to_priced_model(
    _local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch, priceless_entry: dict
) -> None:
    """Regression for https://github.com/BerriAI/litellm/issues/31087 (router-registered priceless entries)."""
    monkeypatch.setitem(
        litellm.model_cost,
        "vertex_ai/some-unmapped-live-model",
        priceless_entry,
    )
    priced_model = "vertex_ai/gemini-live-2.5-flash-preview-native-audio-09-2025"
    priced_entry = litellm.model_cost["gemini-live-2.5-flash-preview-native-audio-09-2025"]

    results: OpenAIRealtimeStreamList = [
        {"type": "session.created", "session": {"model": "some-unmapped-live-model"}},
    ]
    combined_usage_object = Usage(prompt_tokens=8, completion_tokens=25, total_tokens=33)

    cost = handle_realtime_stream_cost_calculation(
        results=results,
        combined_usage_object=combined_usage_object,
        custom_llm_provider="vertex_ai",
        litellm_model_name=priced_model,
    )

    expected_cost = 8 * priced_entry["input_cost_per_token"] + 25 * priced_entry["output_cost_per_token"]
    assert cost == pytest.approx(expected_cost, rel=1e-9)
    assert cost > 0


def test_realtime_explicitly_free_session_model_still_bills_zero(
    _local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        "vertex_ai/free-live-model",
        {
            "litellm_provider": "vertex_ai",
            "mode": "realtime",
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        },
    )

    results: OpenAIRealtimeStreamList = [
        {"type": "session.created", "session": {"model": "free-live-model"}},
    ]
    combined_usage_object = Usage(prompt_tokens=8, completion_tokens=25, total_tokens=33)

    cost = handle_realtime_stream_cost_calculation(
        results=results,
        combined_usage_object=combined_usage_object,
        custom_llm_provider="vertex_ai",
        litellm_model_name="vertex_ai/gemini-live-2.5-flash-preview-native-audio-09-2025",
    )

    assert cost == 0.0


def test_completion_cost_prefers_private_provider_response_model(
    _local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        "openai/selected-cost-model",
        {
            "input_cost_per_token": 0.000002,
            "output_cost_per_token": 0.000004,
            "litellm_provider": "openai",
        },
    )
    response = litellm.ModelResponse(
        id="x",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
        model="requested-route",
    )
    response._hidden_params = {
        "custom_llm_provider": "openai",
        "provider_response_model": "selected-cost-model",
    }
    response.usage = litellm.Usage(prompt_tokens=100, completion_tokens=50)

    cost = litellm.completion_cost(
        completion_response=response,
        custom_llm_provider="openai",
    )

    assert response.model == "requested-route"
    assert cost == pytest.approx(100 * 0.000002 + 50 * 0.000004)


@pytest.mark.parametrize(
    ("base_model", "custom_pricing", "expected"),
    [
        ("openai/base-model", False, "openai/base-model"),
        (None, True, "openai/requested-route"),
    ],
)
def test_explicit_pricing_precedes_private_provider_response_model(
    base_model: str | None,
    custom_pricing: bool,
    expected: str,
) -> None:
    from litellm.cost_calculator import _select_model_name_for_cost_calc

    response = litellm.ModelResponse(
        id="x",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
        model="requested-route",
    )
    response._hidden_params = {"provider_response_model": "selected-cost-model"}

    selected = _select_model_name_for_cost_calc(
        model="requested-route",
        completion_response=response,
        base_model=base_model,
        custom_pricing=custom_pricing,
        custom_llm_provider="openai",
    )

    assert selected == expected


def test_collect_and_combine_realtime_usage_stores_partitioned_text_tokens() -> None:
    """The combined usage that lands in spend logs keeps reasoning out of text_tokens for every turn."""
    results: OpenAIRealtimeStreamList = [
        {"type": "session.created", "session": {"model": "gpt-realtime-2.1-mini"}},
        {
            "type": "response.done",
            "response": {
                "usage": {
                    "total_tokens": 307,
                    "input_tokens": 237,
                    "output_tokens": 70,
                    "input_token_details": {
                        "text_tokens": 43,
                        "audio_tokens": 0,
                        "image_tokens": 194,
                        "cached_tokens": 0,
                    },
                    "output_token_details": {"text_tokens": 70, "audio_tokens": 0, "reasoning_tokens": 52},
                }
            },
        },
        {
            "type": "response.done",
            "response": {
                "usage": {
                    "total_tokens": 363,
                    "input_tokens": 300,
                    "output_tokens": 63,
                    "input_token_details": {
                        "text_tokens": 106,
                        "audio_tokens": 0,
                        "image_tokens": 194,
                        "cached_tokens": 0,
                    },
                    "output_token_details": {"text_tokens": 63, "audio_tokens": 0, "reasoning_tokens": 43},
                }
            },
        },
    ]

    combined = RealtimeAPITokenUsageProcessor.collect_and_combine_usage_from_realtime_stream_results(results=results)

    assert combined.completion_tokens == 133
    assert combined.completion_tokens_details is not None
    assert combined.completion_tokens_details.reasoning_tokens == 95
    assert combined.completion_tokens_details.text_tokens == 38
    assert combined.completion_tokens_details.audio_tokens == 0


def test_realtime_combine_sums_nested_cached_tokens_details():
    results: OpenAIRealtimeStreamList = [
        {
            "type": "response.done",
            "response": {
                "usage": {
                    "input_tokens": 283,
                    "output_tokens": 0,
                    "total_tokens": 283,
                    "input_token_details": {
                        "text_tokens": 116,
                        "audio_tokens": 167,
                        "cached_tokens": 192,
                        "cached_tokens_details": {"text_tokens": 64, "audio_tokens": 128},
                    },
                }
            },
        },
        {
            "type": "response.done",
            "response": {
                "usage": {
                    "input_tokens": 150,
                    "output_tokens": 0,
                    "total_tokens": 150,
                    "input_token_details": {
                        "text_tokens": 50,
                        "audio_tokens": 100,
                        "cached_tokens": 100,
                        "cached_tokens_details": {"audio_tokens": 100},
                    },
                }
            },
        },
    ]

    combined = RealtimeAPITokenUsageProcessor.collect_and_combine_usage_from_realtime_stream_results(
        results=results,
    )

    assert combined.prompt_tokens_details is not None
    assert combined.prompt_tokens_details.cached_tokens == 292
    assert combined.prompt_tokens_details.cached_tokens_details is not None
    assert combined.prompt_tokens_details.cached_tokens_details.audio_tokens == 228
    assert combined.prompt_tokens_details.cached_tokens_details.text_tokens == 64
    assert combined.prompt_tokens_details.cached_tokens_details.image_tokens is None


@pytest.mark.parametrize("details_first", [True, False])
def test_realtime_combine_keeps_cached_split_when_only_one_usage_has_details(details_first: bool):
    with_details: Final = {
        "type": "response.done",
        "response": {
            "usage": {
                "input_tokens": 283,
                "output_tokens": 0,
                "total_tokens": 283,
                "input_token_details": {
                    "text_tokens": 116,
                    "audio_tokens": 167,
                    "cached_tokens": 192,
                    "cached_tokens_details": {"text_tokens": 64, "audio_tokens": 128},
                },
            }
        },
    }
    without_details: Final = {
        "type": "response.done",
        "response": {
            "usage": {
                "input_tokens": 150,
                "output_tokens": 0,
                "total_tokens": 150,
                "input_token_details": {"text_tokens": 50, "audio_tokens": 100, "cached_tokens": 100},
            }
        },
    }
    results: OpenAIRealtimeStreamList = (
        [with_details, without_details] if details_first else [without_details, with_details]
    )

    combined = RealtimeAPITokenUsageProcessor.collect_and_combine_usage_from_realtime_stream_results(
        results=results,
    )

    assert combined.prompt_tokens_details is not None
    assert combined.prompt_tokens_details.cached_tokens == 292
    assert combined.prompt_tokens_details.cached_tokens_details == CachedTokensDetails(text_tokens=64, audio_tokens=128)


def test_usage_without_cached_tokens_details_omits_key():
    usage = Usage(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        prompt_tokens_details=PromptTokensDetailsWrapper(text_tokens=10),
    )

    dumped = usage.prompt_tokens_details.model_dump()
    assert "cached_tokens_details" not in dumped
    assert "cached_tokens_details" not in usage.prompt_tokens_details.model_dump_json()


UNMAPPED_OCR_MODEL: Final = "azure_ai/some-unmapped-ocr-model-for-testing"
MAPPED_OCR_MODEL: Final = "mistral/mistral-ocr-4-0"


def _ocr_response(model: str, pages_processed: int, credits: float | None = None) -> OCRResponse:
    return OCRResponse(
        pages=[OCRPage(index=index, markdown=f"page {index}") for index in range(pages_processed)],
        model=model,
        usage_info=OCRUsageInfo(pages_processed=pages_processed, credits=credits),
    )


def _ocr_logging_obj(litellm_params: dict[str, object]) -> Logging:
    logging_obj: Final = Logging(
        model=UNMAPPED_OCR_MODEL,
        messages=[],
        stream=False,
        call_type="ocr",
        start_time=None,
        litellm_call_id="test-ocr-custom-pricing",
        function_id="1234",
    )
    logging_obj.update_environment_variables(litellm_params=litellm_params, optional_params={})
    return logging_obj


@pytest.mark.parametrize("pages_processed", [1, 3, 10])
def test_ocr_cost_uses_deployment_per_page_pricing_for_unmapped_model(pages_processed: int):
    from litellm.cost_calculator import ocr_cost

    assert UNMAPPED_OCR_MODEL not in litellm.model_cost
    cost, _ = ocr_cost(
        model=UNMAPPED_OCR_MODEL,
        custom_llm_provider="azure_ai",
        response=_ocr_response(UNMAPPED_OCR_MODEL, pages_processed=pages_processed),
        model_info={"ocr_cost_per_page": 0.004},
    )
    assert cost == pytest.approx(0.004 * pages_processed)


def test_ocr_cost_uses_deployment_annotation_only_pricing_for_unmapped_model():
    from litellm.cost_calculator import ocr_cost

    assert UNMAPPED_OCR_MODEL not in litellm.model_cost
    response: Final = OCRResponse(
        pages=[OCRPage(index=index, markdown=f"page {index}") for index in range(3)],
        model=UNMAPPED_OCR_MODEL,
        usage_info=OCRUsageInfo(pages_processed=3, pages_processed_annotation=2),
    )
    cost, _ = ocr_cost(
        model=UNMAPPED_OCR_MODEL,
        custom_llm_provider="azure_ai",
        response=response,
        model_info={"annotation_cost_per_page": 0.01},
    )
    assert cost == pytest.approx(0.01 * 2)


def test_ocr_cost_annotation_only_override_keeps_mapped_per_page_rate():
    from litellm.cost_calculator import ocr_cost

    map_price: Final = litellm.model_cost[MAPPED_OCR_MODEL]["ocr_cost_per_page"]
    response: Final = OCRResponse(
        pages=[OCRPage(index=index, markdown=f"page {index}") for index in range(3)],
        model=MAPPED_OCR_MODEL,
        usage_info=OCRUsageInfo(pages_processed=3, pages_processed_annotation=2),
    )
    cost, _ = ocr_cost(
        model=MAPPED_OCR_MODEL,
        custom_llm_provider="mistral",
        response=response,
        model_info={"annotation_cost_per_page": 0.01},
    )
    assert cost == pytest.approx(map_price * 3 + 0.01 * 2)


def test_ocr_cost_uses_deployment_per_credit_pricing_for_unmapped_model():
    from litellm.cost_calculator import ocr_cost

    cost, _ = ocr_cost(
        model=UNMAPPED_OCR_MODEL,
        custom_llm_provider="azure_ai",
        response=_ocr_response(UNMAPPED_OCR_MODEL, pages_processed=2, credits=4),
        model_info={"ocr_cost_per_credit": 0.25},
    )
    assert cost == pytest.approx(0.25 * 4)


def test_ocr_cost_unmapped_model_without_deployment_pricing_bills_zero():
    from litellm.cost_calculator import ocr_cost

    cost, _ = ocr_cost(
        model=UNMAPPED_OCR_MODEL,
        custom_llm_provider="azure_ai",
        response=_ocr_response(UNMAPPED_OCR_MODEL, pages_processed=5),
        model_info={"id": "some-deployment-id"},
    )
    assert cost == 0.0


@pytest.mark.usefixtures("_local_model_cost_map")
def test_ocr_cost_deployment_pricing_overrides_cost_map_for_mapped_model():
    from litellm.cost_calculator import ocr_cost

    map_price: Final = litellm.get_model_info(MAPPED_OCR_MODEL)["ocr_cost_per_page"]
    assert map_price is not None
    override_price: Final = map_price * 10

    cost, _ = ocr_cost(
        model=MAPPED_OCR_MODEL,
        custom_llm_provider="mistral",
        response=_ocr_response(MAPPED_OCR_MODEL, pages_processed=2),
        model_info={"ocr_cost_per_page": override_price},
    )
    assert cost == pytest.approx(override_price * 2)


@pytest.mark.usefixtures("_local_model_cost_map")
def test_ocr_cost_falls_through_to_cost_map_when_deployment_has_no_ocr_pricing():
    from litellm.cost_calculator import ocr_cost

    map_price: Final = litellm.get_model_info(MAPPED_OCR_MODEL)["ocr_cost_per_page"]
    assert map_price is not None

    cost, _ = ocr_cost(
        model=MAPPED_OCR_MODEL,
        custom_llm_provider="mistral",
        response=_ocr_response(MAPPED_OCR_MODEL, pages_processed=2),
        model_info={"id": "some-deployment-id"},
    )
    assert cost == pytest.approx(map_price * 2)


@pytest.mark.usefixtures("_local_model_cost_map")
def test_ocr_cost_ignores_deployment_credit_pricing_when_response_reports_no_credits():
    from litellm.cost_calculator import ocr_cost

    map_price: Final = litellm.get_model_info(MAPPED_OCR_MODEL)["ocr_cost_per_page"]
    assert map_price is not None

    cost, _ = ocr_cost(
        model=MAPPED_OCR_MODEL,
        custom_llm_provider="mistral",
        response=_ocr_response(MAPPED_OCR_MODEL, pages_processed=2),
        model_info={"ocr_cost_per_credit": 0.5},
    )
    assert cost == pytest.approx(map_price * 2)


@pytest.mark.parametrize("metadata_key", ["metadata", "litellm_metadata"])
def test_completion_cost_ocr_reads_deployment_pricing_from_logging_metadata(metadata_key: str):
    logging_obj = _ocr_logging_obj({metadata_key: {"model_info": {"ocr_cost_per_page": 0.004}}})

    cost = completion_cost(
        completion_response=_ocr_response(UNMAPPED_OCR_MODEL, pages_processed=3),
        model=UNMAPPED_OCR_MODEL,
        custom_llm_provider="azure_ai",
        call_type="ocr",
        custom_pricing=True,
        litellm_logging_obj=logging_obj,
    )
    assert cost == pytest.approx(0.004 * 3)


def test_completion_cost_ocr_prefers_pricing_registered_under_router_model_id(monkeypatch: pytest.MonkeyPatch):
    deployment_id: Final = "ocr-deployment-priced-through-litellm-params"
    monkeypatch.setitem(
        litellm.model_cost, deployment_id, {"mode": "ocr", "litellm_provider": "azure_ai", "ocr_cost_per_page": 0.05}
    )
    logging_obj = _ocr_logging_obj({"metadata": {"model_info": {"mode": "ocr"}}})

    cost = completion_cost(
        completion_response=_ocr_response(UNMAPPED_OCR_MODEL, pages_processed=3),
        model=UNMAPPED_OCR_MODEL,
        custom_llm_provider="azure_ai",
        call_type="ocr",
        custom_pricing=True,
        router_model_id=deployment_id,
        litellm_logging_obj=logging_obj,
    )
    assert cost == pytest.approx(0.05 * 3)


def test_completion_cost_ocr_bills_request_level_pricing_for_direct_sdk_call():
    logging_obj = _ocr_logging_obj({"ocr_cost_per_page": 0.05})

    cost = completion_cost(
        completion_response=_ocr_response(UNMAPPED_OCR_MODEL, pages_processed=3),
        model=UNMAPPED_OCR_MODEL,
        custom_llm_provider="azure_ai",
        call_type="ocr",
        custom_pricing=True,
        litellm_logging_obj=logging_obj,
    )
    assert cost == pytest.approx(0.05 * 3)


def test_completion_cost_ocr_request_level_pricing_fills_in_deployment_model_info_without_ocr_pricing():
    logging_obj = _ocr_logging_obj({"ocr_cost_per_page": 0.05, "metadata": {"model_info": {"mode": "ocr"}}})

    cost = completion_cost(
        completion_response=_ocr_response(UNMAPPED_OCR_MODEL, pages_processed=3),
        model=UNMAPPED_OCR_MODEL,
        custom_llm_provider="azure_ai",
        call_type="ocr",
        custom_pricing=True,
        litellm_logging_obj=logging_obj,
    )
    assert cost == pytest.approx(0.05 * 3)


def test_completion_cost_ocr_ignores_deployment_pricing_without_custom_pricing_flag():
    logging_obj = _ocr_logging_obj({"metadata": {"model_info": {"ocr_cost_per_page": 0.004}}})

    cost = completion_cost(
        completion_response=_ocr_response(UNMAPPED_OCR_MODEL, pages_processed=3),
        model=UNMAPPED_OCR_MODEL,
        custom_llm_provider="azure_ai",
        call_type="ocr",
        custom_pricing=False,
        litellm_logging_obj=logging_obj,
    )
    assert cost == 0.0


def test_completion_cost_prices_responses_websocket_turns_per_service_tier():
    """Issue #41299: a session mixing default and priority turns must price each turn at
    its own returned service_tier, not the summed usage at a single tier."""
    events = [
        {"type": "response.created", "response": {}},
        {
            "type": "response.completed",
            "response": {
                "service_tier": "default",
                "usage": {"input_tokens": 100, "output_tokens": 40, "total_tokens": 140},
            },
        },
        {"type": "rate_limits.updated", "rate_limits": {}},
        {
            "type": "response.completed",
            "response": {
                "service_tier": "priority",
                "usage": {"input_tokens": 60, "output_tokens": 10, "total_tokens": 70},
            },
        },
        {"type": "response.failed", "response": {"usage": None}},
    ]

    partition = ResponsesWebSocketTokenUsageProcessor.partition_results_by_service_tier(events)
    assert tuple(partition.keys()) == ("default", "priority")
    assert len(partition["default"]) == 1
    assert len(partition["priority"]) == 1

    logging_obj = Logging(
        model="gpt-5.4",
        messages=[],
        stream=False,
        call_type=CallTypes.aresponses_websocket.value,
        start_time=time.time(),
        litellm_call_id="responses-ws-tier-test",
        function_id="responses-ws-tier-test",
    )
    normalized = logging_obj.normalize_logging_result(result=events)
    assert isinstance(normalized, LiteLLMRealtimeStreamLoggingObject)
    assert normalized.service_tier is None

    def _http_cost(input_tokens: int, output_tokens: int, service_tier: str) -> float:
        return completion_cost(
            completion_response=ResponsesAPIResponse(
                id=f"resp-{service_tier}",
                created_at=1700000000,
                output=[],
                service_tier=service_tier,
                usage=ResponseAPIUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                ),
            ),
            model="gpt-5.4",
            call_type=CallTypes.aresponses.value,
            custom_llm_provider="openai",
        )

    ws_cost = completion_cost(
        completion_response=normalized,
        model="gpt-5.4",
        call_type=CallTypes.aresponses_websocket.value,
        custom_llm_provider="openai",
    )

    assert ws_cost == pytest.approx(_http_cost(100, 40, "default") + _http_cost(60, 10, "priority"))
    assert ws_cost != pytest.approx(_http_cost(160, 50, "default"))
    assert ws_cost != pytest.approx(_http_cost(160, 50, "priority"))


_TIERED_BATCH_MODEL: Final = "lit-tiered-batch-model"
_FLAT_CACHE_BATCH_MODEL: Final = "lit-tiered-batch-model-without-cache-batch-rates"
_TIERED_BATCH_ENTRY: Final = MappingProxyType(
    {
        "litellm_provider": "openai",
        "mode": "chat",
        "input_cost_per_token": 2e-6,
        "output_cost_per_token": 8e-6,
        "cache_read_input_token_cost": 2e-7,
        "cache_creation_input_token_cost": 2.5e-6,
        "input_cost_per_token_above_272k_tokens": 4e-6,
        "output_cost_per_token_above_272k_tokens": 1.2e-5,
        "input_cost_per_token_batches": 1e-6,
        "output_cost_per_token_batches": 4e-6,
        "cache_read_input_token_cost_batches": 1e-7,
        "cache_creation_input_token_cost_batches": 1.25e-6,
        "input_cost_per_token_above_272k_tokens_batches": 3e-6,
        "output_cost_per_token_above_272k_tokens_batches": 7e-6,
        "cache_read_input_token_cost_above_272k_tokens_batches": 3e-7,
        "cache_creation_input_token_cost_above_272k_tokens_batches": 3.75e-6,
    }
)
_BATCH_RATE_PREFIXES: Final = (
    "input_cost_per_token",
    "output_cost_per_token",
    "cache_read_input_token_cost",
    "cache_creation_input_token_cost",
)


@pytest.fixture
def _tiered_batch_models(_local_model_cost_map: None) -> None:
    litellm.register_model(
        model_cost={
            _TIERED_BATCH_MODEL: {**_TIERED_BATCH_ENTRY},
            _FLAT_CACHE_BATCH_MODEL: {
                key: rate
                for key, rate in _TIERED_BATCH_ENTRY.items()
                if not (key.startswith("cache_") and key.endswith("_batches"))
            },
        },
        persist_across_reloads=False,
    )


def test_batch_cost_calculator_bills_the_long_context_batch_tier_above_272k(_tiered_batch_models: None) -> None:
    from litellm.cost_calculator import batch_cost_calculator

    usage: Final = Usage(prompt_tokens=300_035, completion_tokens=64, total_tokens=300_099)

    prompt_cost, completion_cost_value = batch_cost_calculator(
        usage=usage, model=_TIERED_BATCH_MODEL, custom_llm_provider="openai"
    )

    assert prompt_cost == pytest.approx(300_035 * 3e-6)
    assert completion_cost_value == pytest.approx(64 * 7e-6)


@pytest.mark.parametrize("prompt_tokens", [272_000, 1_000])
def test_batch_cost_calculator_bills_the_flat_batch_rate_at_or_below_272k(
    _tiered_batch_models: None, prompt_tokens: int
) -> None:
    from litellm.cost_calculator import batch_cost_calculator

    usage: Final = Usage(prompt_tokens=prompt_tokens, completion_tokens=64, total_tokens=prompt_tokens + 64)

    prompt_cost, completion_cost_value = batch_cost_calculator(
        usage=usage, model=_TIERED_BATCH_MODEL, custom_llm_provider="openai"
    )

    assert prompt_cost == pytest.approx(prompt_tokens * 1e-6)
    assert completion_cost_value == pytest.approx(64 * 4e-6)


def test_get_model_info_exposes_every_registered_batch_rate(_tiered_batch_models: None) -> None:
    info: Final = litellm.get_model_info(_TIERED_BATCH_MODEL, custom_llm_provider="openai")
    batch_keys: Final = tuple(key for key in _TIERED_BATCH_ENTRY if key.endswith("_batches"))

    assert len(batch_keys) == 8
    assert {key: info[key] for key in batch_keys} == {key: _TIERED_BATCH_ENTRY[key] for key in batch_keys}


def test_regular_path_never_bills_the_batch_tier_keys(_local_model_cost_map, monkeypatch):
    monkeypatch.setitem(
        litellm.model_cost,
        "lit-batch-tier-guard",
        {
            "litellm_provider": "openai",
            "mode": "chat",
            "input_cost_per_token": 2e-6,
            "output_cost_per_token": 8e-6,
            "input_cost_per_token_batches": 1e-6,
            "output_cost_per_token_batches": 4e-6,
            "input_cost_per_token_above_272k_tokens_batches": 5e-6,
            "output_cost_per_token_above_272k_tokens_batches": 9e-6,
        },
    )

    prompt_cost, completion_cost = litellm.cost_per_token(
        model="lit-batch-tier-guard", custom_llm_provider="openai", prompt_tokens=300_035, completion_tokens=64
    )

    assert prompt_cost == pytest.approx(300_035 * 2e-6)
    assert completion_cost == pytest.approx(64 * 8e-6)


@pytest.mark.parametrize("prefix", _BATCH_RATE_PREFIXES)
def test_every_openai_entry_with_a_long_context_rate_and_a_batch_rate_declares_the_batch_tier(
    _local_model_cost_map: None, prefix: str
) -> None:
    undeclared: Final = [
        name
        for name, entry in litellm.model_cost.items()
        if isinstance(entry, dict)
        and entry.get("litellm_provider") == "openai"
        and entry.get(f"{prefix}_above_272k_tokens") is not None
        and entry.get(f"{prefix}_batches") is not None
        and entry.get(f"{prefix}_above_272k_tokens_batches") is None
    ]

    assert undeclared == []


def test_batch_cost_calculator_ignores_malformed_batch_tier_keys():
    from litellm.cost_calculator import batch_cost_calculator

    usage = Usage(prompt_tokens=300_035, completion_tokens=64, total_tokens=300_099)
    model_info = cast(
        ModelInfo,
        {
            "input_cost_per_token": 2e-6,
            "output_cost_per_token": 8e-6,
            "input_cost_per_token_batches": 1e-6,
            "output_cost_per_token_batches": 4e-6,
            "input_cost_per_token_above_272k_tokens_batches": 2e-6,
            "output_cost_per_token_above_272k_tokens_batches": 6e-6,
            "input_cost_per_token_above_lots_tokens_batches": 1.0,
        },
    )

    prompt_cost, completion_cost = batch_cost_calculator(
        usage=usage, model=_TIERED_BATCH_MODEL, custom_llm_provider="openai", model_info=model_info
    )

    assert prompt_cost == pytest.approx(300_035 * 2e-6)
    assert completion_cost == pytest.approx(64 * 6e-6)


def _cached_usage(prompt_tokens: int, cached_tokens: int, completion_tokens: int) -> Usage:
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=cached_tokens),
    )


def test_batch_cost_calculator_bills_cached_tokens_at_the_long_context_batch_cached_rate(
    _tiered_batch_models: None,
) -> None:
    from litellm.cost_calculator import batch_cost_calculator

    prompt_cost, completion_cost_value = batch_cost_calculator(
        usage=_cached_usage(300_048, 300_045, 11), model=_TIERED_BATCH_MODEL, custom_llm_provider="openai"
    )

    assert prompt_cost == pytest.approx(3 * 3e-6 + 300_045 * 3e-7)
    assert completion_cost_value == pytest.approx(11 * 7e-6)


def test_batch_cost_calculator_bills_cached_tokens_at_the_flat_batch_cached_rate_at_or_below_272k(
    _tiered_batch_models: None,
) -> None:
    from litellm.cost_calculator import batch_cost_calculator

    prompt_cost, _ = batch_cost_calculator(
        usage=_cached_usage(1_000, 900, 4), model=_TIERED_BATCH_MODEL, custom_llm_provider="openai"
    )

    assert prompt_cost == pytest.approx(100 * 1e-6 + 900 * 1e-7)


def test_batch_cost_calculator_bills_cached_tokens_at_the_batch_input_rate_without_a_cached_batch_rate(
    _tiered_batch_models: None,
) -> None:
    from litellm.cost_calculator import batch_cost_calculator

    prompt_cost, _ = batch_cost_calculator(
        usage=_cached_usage(300_048, 300_045, 11), model=_FLAT_CACHE_BATCH_MODEL, custom_llm_provider="openai"
    )

    assert prompt_cost == pytest.approx(300_048 * 3e-6)


def _cache_write_usage(prompt_tokens: int, cache_write_tokens: int, completion_tokens: int) -> Usage:
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=0, cache_write_tokens=cache_write_tokens),
    )


def test_batch_cost_calculator_bills_cache_write_tokens_at_the_long_context_batch_cache_write_rate(
    _tiered_batch_models: None,
) -> None:
    from litellm.cost_calculator import batch_cost_calculator

    prompt_cost, completion_cost_value = batch_cost_calculator(
        usage=_cache_write_usage(300_048, 300_045, 4), model=_TIERED_BATCH_MODEL, custom_llm_provider="openai"
    )

    assert prompt_cost == pytest.approx(3 * 3e-6 + 300_045 * 3.75e-6)
    assert completion_cost_value == pytest.approx(4 * 7e-6)


def test_batch_cost_calculator_bills_cache_write_tokens_at_the_flat_batch_cache_write_rate_at_or_below_272k(
    _tiered_batch_models: None,
) -> None:
    from litellm.cost_calculator import batch_cost_calculator

    prompt_cost, _ = batch_cost_calculator(
        usage=_cache_write_usage(1_000, 900, 4), model=_TIERED_BATCH_MODEL, custom_llm_provider="openai"
    )

    assert prompt_cost == pytest.approx(100 * 1e-6 + 900 * 1.25e-6)


def test_batch_cost_calculator_bills_cache_write_tokens_at_the_batch_input_rate_without_a_cache_write_batch_rate(
    _tiered_batch_models: None,
) -> None:
    from litellm.cost_calculator import batch_cost_calculator

    prompt_cost, _ = batch_cost_calculator(
        usage=_cache_write_usage(300_048, 300_045, 4), model=_FLAT_CACHE_BATCH_MODEL, custom_llm_provider="openai"
    )

    assert prompt_cost == pytest.approx(300_048 * 3e-6)


def test_batch_cost_calculator_prices_modalities_and_cached_tokens_together_in_the_crossed_tier() -> None:
    from litellm.cost_calculator import batch_cost_calculator

    model_info: Final = cast(
        ModelInfo,
        {
            "input_cost_per_token_batches": 1e-6,
            "input_cost_per_token_above_272k_tokens_batches": 3e-6,
            "input_cost_per_audio_token_batches": 5e-6,
            "cache_read_input_token_cost_batches": 1e-7,
            "cache_read_input_token_cost_above_272k_tokens_batches": 3e-7,
        },
    )
    usage: Final = Usage(
        prompt_tokens=300_000,
        completion_tokens=0,
        total_tokens=300_000,
        prompt_tokens_details=PromptTokensDetailsWrapper(audio_tokens=64, image_tokens=10, cached_tokens=1_000),
    )

    prompt_cost, _ = batch_cost_calculator(
        usage=usage, model=_TIERED_BATCH_MODEL, custom_llm_provider="openai", model_info=model_info
    )

    assert prompt_cost == pytest.approx(298_926 * 3e-6 + 64 * 5e-6 + 10 * 3e-6 + 1_000 * 3e-7)


QWEN3_NEXT_REGIONS: Final = ("ap-northeast-1", "ap-south-1", "ap-southeast-2", "eu-west-1", "eu-west-2", "sa-east-1")


@pytest.mark.parametrize("region", QWEN3_NEXT_REGIONS)
def test_cost_per_token_bedrock_qwen3_next_uses_regional_entry_not_us_rate(
    monkeypatch: pytest.MonkeyPatch, region: str
) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))

    regional: Final = litellm.model_cost[f"bedrock/{region}/qwen.qwen3-next-80b-a3b"]
    us: Final = litellm.model_cost["qwen.qwen3-next-80b-a3b"]
    assert regional["input_cost_per_token"] != us["input_cost_per_token"]
    assert regional["output_cost_per_token"] != us["output_cost_per_token"]

    prompt_tokens, completion_tokens = 1000, 500
    prompt_usd, completion_usd = cost_per_token(
        model=f"bedrock/{region}/qwen.qwen3-next-80b-a3b",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        custom_llm_provider="bedrock",
    )

    assert prompt_usd == pytest.approx(prompt_tokens * regional["input_cost_per_token"])
    assert completion_usd == pytest.approx(completion_tokens * regional["output_cost_per_token"])


def test_cost_per_token_bedrock_nemotron_super_3_uses_eu_west_2_entry_not_us_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))

    regional_key: Final = "bedrock/eu-west-2/nvidia.nemotron-super-3-120b"
    regional: Final = litellm.model_cost[regional_key]
    us: Final = litellm.model_cost["nvidia.nemotron-super-3-120b"]
    assert regional["input_cost_per_token"] != us["input_cost_per_token"]
    assert regional["output_cost_per_token"] != us["output_cost_per_token"]

    prompt_tokens, completion_tokens = 1000, 500
    prompt_usd, completion_usd = cost_per_token(
        model=regional_key,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        custom_llm_provider="bedrock",
    )

    assert prompt_usd == pytest.approx(prompt_tokens * regional["input_cost_per_token"])
    assert completion_usd == pytest.approx(completion_tokens * regional["output_cost_per_token"])


GPT_REALTIME_2_FAMILY: Final = (
    "azure/gpt-realtime-2.1",
    "azure/gpt-realtime-2.1-mini",
    "gpt-realtime-2",
    "gpt-realtime-2.1",
    "gpt-realtime-2.1-mini",
)


def test_gpt_realtime_2_family_prices_audio_cache_writes_and_reads_alike(_local_model_cost_map: None) -> None:
    audio_cache_rates: Final = {
        model: (
            litellm.model_cost[model].get("cache_read_input_audio_token_cost"),
            litellm.model_cost[model].get("cache_creation_input_audio_token_cost"),
        )
        for model in GPT_REALTIME_2_FAMILY
    }

    # Azure publishes one cached-audio meter per gpt-realtime-2 deployment,
    # https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/, checked 2026-09-23
    assert all(read is not None and write == read for read, write in audio_cache_rates.values()), audio_cache_rates
    assert len(audio_cache_rates) == len(GPT_REALTIME_2_FAMILY)


GEMINI_LIVE_NATIVE_AUDIO_CASES: Final = (
    ("gemini-live-2.5-flash-native-audio", "vertex_ai"),
    ("gemini-live-2.5-flash-preview-native-audio-09-2025", "vertex_ai"),
    ("gemini/gemini-live-2.5-flash-preview-native-audio-09-2025", "gemini"),
)


@pytest.mark.parametrize(("model", "provider"), GEMINI_LIVE_NATIVE_AUDIO_CASES)
def test_gemini_live_native_audio_carries_no_cached_input_rate(
    _local_model_cost_map: None, model: str, provider: str
) -> None:
    # the Vertex pricing table prints N/A for cached input on every Live row,
    # https://cloud.google.com/vertex-ai/generative-ai/pricing, checked 2026-09-23
    assert litellm.get_model_info(model, custom_llm_provider=provider)["cache_read_input_token_cost"] is None

    prompt_usd, _ = cost_per_token(
        model=model,
        prompt_tokens=101_000,
        completion_tokens=0,
        custom_llm_provider=provider,
        usage_object=Usage(
            prompt_tokens=101_000,
            completion_tokens=0,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=100_000),
        ),
    )
    fresh_usd, _ = cost_per_token(
        model=model,
        prompt_tokens=101_000,
        completion_tokens=0,
        custom_llm_provider=provider,
        usage_object=Usage(prompt_tokens=101_000, completion_tokens=0),
    )

    assert prompt_usd == pytest.approx(fresh_usd), (
        "with no cached rate the cached tokens bill at the input rate, so a phantom discount cannot appear"
    )
    assert prompt_usd > 0


@pytest.mark.parametrize(("model", "provider"), GEMINI_LIVE_NATIVE_AUDIO_CASES)
def test_gemini_live_native_audio_declares_prompt_caching_unsupported(
    _local_model_cost_map: None, model: str, provider: str
) -> None:
    # the Vertex context-caching supported-model lists contain no Live model while 2.5 Flash is listed,
    # https://cloud.google.com/vertex-ai/generative-ai/docs/context-cache/context-cache-overview, checked 2026-09-23
    assert litellm.get_model_info(model, custom_llm_provider=provider)["supports_prompt_caching"] is False
    assert supports_prompt_caching(model=model, custom_llm_provider=provider) is False
    assert supports_prompt_caching(model="gemini-2.5-flash", custom_llm_provider="vertex_ai") is True, (
        "control: the helper swallows a lookup error into False, so without this a broken lookup reads as a pass"
    )


@pytest.mark.parametrize(
    "model",
    ["gemini-live-2.5-flash-native-audio", "vertex_ai/gemini-live-2.5-flash-native-audio"],
)
def test_gemini_live_native_audio_limits_and_capabilities_match_vendor_model_card(
    _local_model_cost_map: None, model: str
) -> None:
    info = litellm.get_model_info(model)

    # the Vertex model card for gemini-live-2.5-flash-native-audio publishes these limits and flags,
    # https://cloud.google.com/vertex-ai/generative-ai/docs/models, checked 2026-09-23
    assert info["max_input_tokens"] == 131072
    assert info["max_output_tokens"] == 65536
    assert info["max_tokens"] == 65536
    assert info["supports_response_schema"] is False
    assert info["supports_url_context"] is False
    assert info["supports_pdf_input"] is False


def test_baseten_glm_5_3_fast_is_priced_from_registry(_local_model_cost_map: None) -> None:
    model: Final = "baseten/zai-org/GLM-5.3-Fast"
    prompt_tokens: Final = 1000
    completion_tokens: Final = 500

    prompt_usd, completion_usd = litellm.cost_per_token(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )

    entry: Final = litellm.model_cost[model]
    assert prompt_usd == pytest.approx(prompt_tokens * entry["input_cost_per_token"])
    assert completion_usd == pytest.approx(completion_tokens * entry["output_cost_per_token"])
    assert prompt_usd > 0
    assert completion_usd > 0


def test_completion_cost_charges_explicit_per_token_rates_over_registered_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        "smoke-priced-model",
        {"input_cost_per_token": 0.01, "output_cost_per_token": 0.02, "litellm_provider": "openai", "mode": "chat"},
    )
    response: Final = ModelResponse(
        model="smoke-priced-model",
        choices=[],
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )

    cost: Final = completion_cost(
        completion_response=response,
        model="smoke-priced-model",
        custom_llm_provider="openai",
        custom_cost_per_token={"input_cost_per_token": 0.001, "output_cost_per_token": 0.002},
    )

    assert cost == pytest.approx(100 * 0.001 + 50 * 0.002)


def test_completion_cost_is_zero_when_explicit_rates_are_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        "smoke-priced-model",
        {"input_cost_per_token": 0.01, "output_cost_per_token": 0.02, "litellm_provider": "openai", "mode": "chat"},
    )
    response: Final = ModelResponse(
        model="smoke-priced-model",
        choices=[],
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )

    cost: Final = completion_cost(
        completion_response=response,
        model="smoke-priced-model",
        custom_llm_provider="openai",
        custom_cost_per_token={"input_cost_per_token": 0.0, "output_cost_per_token": 0.0},
    )

    assert cost == 0.0
