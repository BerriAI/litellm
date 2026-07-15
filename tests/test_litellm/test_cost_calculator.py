import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


from pydantic import BaseModel

import litellm
from litellm.cost_calculator import (
    RealtimeAPITokenUsageProcessor,
    completion_cost,
    cost_per_token,
    handle_realtime_stream_cost_calculation,
    response_cost_calculator,
)
from litellm.types.llms.openai import OpenAIRealtimeStreamList
from litellm.types.utils import ModelResponse, PromptTokensDetailsWrapper, Usage
from litellm.utils import TranscriptionResponse


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


def test_completion_cost_uses_response_model_for_dynamic_routing():
    """
    Test that completion_cost uses the model from the response object
    when the input model (e.g., azure-model-router) is not in model_cost.
    This supports Azure Model Router and similar dynamic routing scenarios.
    """
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

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


def test_cost_calculator_with_response_cost_in_additional_headers():
    class MockResponse(BaseModel):
        _hidden_params = {
            "additional_headers": {"llm_provider-x-litellm-response-cost": 1000}
        }

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


def test_baseten_model_api_pricing_entries():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    expected_pricing = {
        "baseten/nvidia/Nemotron-120B-A12B": (3e-07, 7.5e-07),
        "baseten/MiniMaxAI/MiniMax-M2.5": (3e-07, 1.2e-06),
        "baseten/zai-org/GLM-5": (9.5e-07, 3.15e-06),
        "baseten/zai-org/GLM-4.7": (6e-07, 2.2e-06),
        "baseten/zai-org/GLM-4.6": (6e-07, 2.2e-06),
        "baseten/moonshotai/Kimi-K2.5": (6e-07, 3e-06),
        "baseten/moonshotai/Kimi-K2-Thinking": (6e-07, 2.5e-06),
        "baseten/moonshotai/Kimi-K2-Instruct-0905": (6e-07, 2.5e-06),
        "baseten/openai/gpt-oss-120b": (1e-07, 5e-07),
        "baseten/deepseek-ai/DeepSeek-V3.1": (5e-07, 1.5e-06),
        "baseten/deepseek-ai/DeepSeek-V3-0324": (7.7e-07, 7.7e-07),
    }

    for model_name, (input_cost, output_cost) in expected_pricing.items():
        model_info = litellm.model_cost.get(model_name)
        assert model_info is not None, f"Missing model pricing entry: {model_name}"
        assert model_info["litellm_provider"] == "baseten"
        assert model_info["input_cost_per_token"] == input_cost
        assert model_info["output_cost_per_token"] == output_cost


def test_wandb_model_api_pricing_entries():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    expected_pricing = {
        "wandb/moonshotai/Kimi-K2.5": (6e-07, 3e-06),
        "wandb/MiniMaxAI/MiniMax-M2.5": (3e-07, 1.2e-06),
    }

    for model_name, (input_cost, output_cost) in expected_pricing.items():
        model_info = litellm.model_cost.get(model_name)
        assert model_info is not None, f"Missing model pricing entry: {model_name}"
        assert model_info["litellm_provider"] == "wandb"
        assert model_info["input_cost_per_token"] == input_cost
        assert model_info["output_cost_per_token"] == output_cost


def test_openrouter_qwen36_plus_model_info():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    model_info = litellm.model_cost.get("openrouter/qwen/qwen3.6-plus")

    assert model_info is not None
    assert model_info["litellm_provider"] == "openrouter"
    assert model_info["mode"] == "chat"
    assert model_info["max_input_tokens"] == 1000000
    assert model_info["max_output_tokens"] == 65536
    assert model_info["input_cost_per_token"] == 3.25e-07
    assert model_info["output_cost_per_token"] == 1.95e-06
    assert model_info["supports_function_calling"] is True
    assert model_info["supports_tool_choice"] is True
    assert model_info["supports_reasoning"] is True
    assert model_info["supports_vision"] is True


@pytest.mark.parametrize(
    "model",
    [
        "github_copilot/mai-code-1-flash",
        "github_copilot/mai-code-1-flash-internal",
    ],
)
def test_github_copilot_mai_code_1_flash_pricing(model):
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    model_info = litellm.model_cost.get(model)

    assert model_info is not None, f"Missing model pricing entry: {model}"
    assert model_info["litellm_provider"] == "github_copilot"
    assert model_info["mode"] == "chat"
    assert model_info["input_cost_per_token"] == 7.5e-07
    assert model_info["cache_read_input_token_cost"] == 7.5e-08
    assert model_info["output_cost_per_token"] == 4.5e-06
    assert model_info["supported_endpoints"] == ["/v1/chat/completions"]

    prompt_usd, completion_usd = cost_per_token(
        model=model,
        prompt_tokens=1000,
        completion_tokens=500,
        custom_llm_provider="github_copilot",
        usage_object=Usage(
            prompt_tokens=1000,
            completion_tokens=500,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=200),
        ),
    )

    assert prompt_usd == pytest.approx((800 * 7.5e-07) + (200 * 7.5e-08))
    assert completion_usd == pytest.approx(500 * 4.5e-06)


def test_cost_calculator_with_usage(monkeypatch):
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    usage = Usage(
        prompt_tokens=120,
        completion_tokens=100,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            text_tokens=10,
            audio_tokens=90,
            image_tokens=20,
        ),
    )
    mr = ModelResponse(usage=usage, model="gemini-2.0-flash-001")

    result = response_cost_calculator(
        response_object=mr,
        model="",
        custom_llm_provider="vertex_ai",
        call_type="acompletion",
        optional_params={},
        cache_hit=None,
        base_model=None,
    )

    model_info = litellm.model_cost["gemini-2.0-flash-001"]

    # Step 1: Test a model where input_cost_per_image_token is not set.
    # In this case the calculation should use input_cost_per_token as fallback.
    assert (
        model_info.get("input_cost_per_image_token") is None
    ), "Test case expects that input_cost_per_image_token is not set"

    expected_cost = (
        usage.prompt_tokens_details.audio_tokens
        * model_info["input_cost_per_audio_token"]
        + usage.prompt_tokens_details.text_tokens * model_info["input_cost_per_token"]
        + usage.prompt_tokens_details.image_tokens * model_info["input_cost_per_token"]
        + usage.completion_tokens * model_info["output_cost_per_token"]
    )

    assert result == expected_cost, f"Got {result}, Expected {expected_cost}"

    # Step 2: Set input_cost_per_image_token.
    # In this case the explicit cost information should be used.
    temp_model_info_object = dict(model_info)
    temp_model_info_object["input_cost_per_image_token"] = 0.5

    monkeypatch.setattr(
        litellm,
        "model_cost",
        {"gemini-2.0-flash-001": temp_model_info_object},
    )

    # Invalidate caches after modifying litellm.model_cost
    from litellm.utils import _invalidate_model_cost_lowercase_map

    _invalidate_model_cost_lowercase_map()

    result = response_cost_calculator(
        response_object=mr,
        model="",
        custom_llm_provider="vertex_ai",
        call_type="acompletion",
        optional_params={},
        cache_hit=None,
        base_model=None,
    )

    expected_cost = (
        usage.prompt_tokens_details.audio_tokens
        * temp_model_info_object["input_cost_per_audio_token"]
        + usage.prompt_tokens_details.text_tokens
        * temp_model_info_object["input_cost_per_token"]
        + usage.prompt_tokens_details.image_tokens
        * temp_model_info_object["input_cost_per_image_token"]
        + usage.completion_tokens * temp_model_info_object["output_cost_per_token"]
    )

    assert result == expected_cost, f"Got {result}, Expected {expected_cost}"


def test_transcription_cost_uses_token_pricing():
    from litellm import completion_cost

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    usage = Usage(
        prompt_tokens=14,
        completion_tokens=45,
        total_tokens=59,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            text_tokens=0, audio_tokens=14
        ),
    )
    response = TranscriptionResponse(text="demo text")
    response.usage = usage

    cost = completion_cost(
        completion_response=response,
        model="gpt-4o-transcribe",
        custom_llm_provider="openai",
        call_type="atranscription",
    )

    expected_cost = (14 * 2.5e-06) + (45 * 1e-05)
    assert pytest.approx(cost, rel=1e-6) == expected_cost


def test_transcription_cost_falls_back_to_duration():
    from litellm import completion_cost

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    response = TranscriptionResponse(text="demo text")
    response.duration = 10.0

    cost = completion_cost(
        completion_response=response,
        model="whisper-1",
        custom_llm_provider="openai",
        call_type="atranscription",
    )

    expected_cost = 10.0 * 0.0001
    assert pytest.approx(cost, rel=1e-6) == expected_cost


def test_vertex_chirp_3_transcription_cost_from_duration():
    """Regression: the chirp_3 cost map entry shipped with output_cost_per_second 0.0,
    and cost_per_second prefers output_cost_per_second whenever it is not None, so
    every transcription priced to $0.00 instead of using input_cost_per_second."""
    from litellm import completion_cost

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    response = TranscriptionResponse(text="demo text")
    response.duration = 18.0

    cost = completion_cost(
        completion_response=response,
        model="vertex_ai/chirp_3",
        custom_llm_provider="vertex_ai",
        call_type="atranscription",
    )

    expected_cost = 18.0 * 0.00026667
    assert cost > 0
    assert pytest.approx(cost, rel=1e-6) == expected_cost


def test_handle_realtime_stream_cost_calculation():
    from litellm.cost_calculator import RealtimeAPITokenUsageProcessor

    # Setup test data
    results: OpenAIRealtimeStreamList = [
        {"type": "session.created", "session": {"model": "gpt-3.5-turbo"}},
        {
            "type": "response.done",
            "response": {
                "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
            },
        },
        {
            "type": "response.done",
            "response": {
                "usage": {
                    "input_tokens": 200,
                    "output_tokens": 100,
                    "total_tokens": 300,
                }
            },
        },
    ]

    combined_usage_object = RealtimeAPITokenUsageProcessor.collect_and_combine_usage_from_realtime_stream_results(
        results=results,
    )

    # Test with explicit model name
    cost = handle_realtime_stream_cost_calculation(
        results=results,
        combined_usage_object=combined_usage_object,
        custom_llm_provider="openai",
        litellm_model_name="gpt-3.5-turbo",
    )

    # Calculate expected cost
    # gpt-3.5-turbo costs: $0.0015/1K tokens input, $0.002/1K tokens output
    expected_cost = (300 * 0.0015 / 1000) + (  # input tokens (100 + 200)
        150 * 0.002 / 1000
    )  # output tokens (50 + 100)
    assert (
        abs(cost - expected_cost) <= 0.00075
    )  # Allow small floating point differences

    # Test with different model name in session
    results[0]["session"]["model"] = "gpt-4"

    cost = handle_realtime_stream_cost_calculation(
        results=results,
        combined_usage_object=combined_usage_object,
        custom_llm_provider="openai",
        litellm_model_name="gpt-3.5-turbo",
    )

    # Calculate expected cost using gpt-4 rates
    # gpt-4 costs: $0.03/1K tokens input, $0.06/1K tokens output
    expected_cost = (300 * 0.03 / 1000) + (  # input tokens
        150 * 0.06 / 1000
    )  # output tokens
    assert abs(cost - expected_cost) < 0.00076

    # Test with no response.done events
    results = [{"type": "session.created", "session": {"model": "gpt-3.5-turbo"}}]
    combined_usage_object = RealtimeAPITokenUsageProcessor.collect_and_combine_usage_from_realtime_stream_results(
        results=results,
    )
    cost = handle_realtime_stream_cost_calculation(
        results=results,
        combined_usage_object=combined_usage_object,
        custom_llm_provider="openai",
        litellm_model_name="gpt-3.5-turbo",
    )
    assert cost == 0.0  # No usage, no cost


def test_handle_realtime_stream_cost_calculation_stores_cost_breakdown():
    """Regression: realtime cost must populate logging_obj.cost_breakdown so the
    spend logs / UI show input vs output cost (issue: cost_breakdown was None for
    /v1/realtime even though a total spend was computed)."""
    from datetime import datetime

    from litellm.litellm_core_utils.litellm_logging import Logging

    results: OpenAIRealtimeStreamList = [
        {"type": "session.created", "session": {"model": "gpt-4o-realtime-preview"}},
        {
            "type": "response.done",
            "response": {
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "total_tokens": 150,
                }
            },
        },
    ]
    combined_usage_object = RealtimeAPITokenUsageProcessor.collect_and_combine_usage_from_realtime_stream_results(
        results=results,
    )

    logging_obj = Logging(
        model="gpt-4o-realtime-preview",
        messages=[],
        stream=False,
        call_type="_arealtime",
        start_time=datetime.now(),
        litellm_call_id="realtime-cost-breakdown-test",
        function_id="realtime-cost-breakdown-test",
    )

    total_cost = handle_realtime_stream_cost_calculation(
        results=results,
        combined_usage_object=combined_usage_object,
        custom_llm_provider="openai",
        litellm_model_name="gpt-4o-realtime-preview",
        litellm_logging_obj=logging_obj,
    )

    assert total_cost > 0
    assert logging_obj.cost_breakdown is not None
    assert logging_obj.cost_breakdown["input_cost"] > 0
    assert logging_obj.cost_breakdown["output_cost"] > 0
    assert (
        abs(
            logging_obj.cost_breakdown["input_cost"]
            + logging_obj.cost_breakdown["output_cost"]
            - total_cost
        )
        < 1e-9
    )
    assert abs(logging_obj.cost_breakdown["total_cost"] - total_cost) < 1e-9


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

    usage = RealtimeAPITokenUsageProcessor.collect_and_combine_usage_from_realtime_stream_results(
        results=results
    )
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

    usage = RealtimeAPITokenUsageProcessor.collect_and_combine_usage_from_realtime_stream_results(
        results=results
    )
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
        if r["type"]
        in ("rate_limits.updated", "response.function_call_arguments.delta")
    }
    assert unknown_types == {
        "rate_limits.updated",
        "response.function_call_arguments.delta",
    }

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dumped = logging_result.model_dump()
    assert len(dumped["results"]) == len(results)


def test_realtime_transcription_duration_cost(monkeypatch):
    """
    gpt-realtime-whisper transcription sessions are billed by input audio duration
    ($0.017/min). The .completed events carry usage {type: duration, seconds: N};
    cost must equal total_seconds * input_cost_per_second.
    """
    from datetime import datetime

    from litellm.litellm_core_utils.litellm_logging import Logging

    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))

    from litellm.cost_calculator import RealtimeAPITokenUsageProcessor

    results: OpenAIRealtimeStreamList = [
        {
            "type": "session.created",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {"transcription": {"model": "gpt-realtime-whisper"}}
                },
            },
        },
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "hello",
            "usage": {"type": "duration", "seconds": 60.0},
        },
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "world",
            "usage": {"type": "duration", "seconds": 30.0},
        },
    ]

    combined = RealtimeAPITokenUsageProcessor.collect_and_combine_usage_from_realtime_stream_results(
        results=results
    )
    logging_obj = Logging(
        model="gpt-realtime-whisper",
        messages=[],
        stream=False,
        call_type="_arealtime",
        start_time=datetime.now(),
        litellm_call_id="realtime-transcription-cost-breakdown-test",
        function_id="realtime-transcription-cost-breakdown-test",
    )
    cost = handle_realtime_stream_cost_calculation(
        results=results,
        combined_usage_object=combined,
        custom_llm_provider="openai",
        litellm_model_name="gpt-realtime-whisper",
        litellm_logging_obj=logging_obj,
    )

    # 90 seconds at $0.017/minute.
    expected = 90.0 * (0.017 / 60)
    assert abs(cost - expected) < 1e-9
    assert cost > 0  # guards against the duration branch being dropped
    assert logging_obj.cost_breakdown is not None
    assert abs(logging_obj.cost_breakdown["total_cost"] - cost) < 1e-9

    # The transcription cost must be attributed in the breakdown, not just folded
    # into total_cost, or input_cost + output_cost + additional_costs won't sum to total_cost.
    additional_costs = logging_obj.cost_breakdown.get("additional_costs")
    assert additional_costs is not None
    assert abs(additional_costs["transcription_cost"] - expected) < 1e-9
    attributed_total = (
        logging_obj.cost_breakdown["input_cost"]
        + logging_obj.cost_breakdown["output_cost"]
        + additional_costs["transcription_cost"]
    )
    assert abs(attributed_total - logging_obj.cost_breakdown["total_cost"]) < 1e-9


def test_realtime_transcription_duration_cost_resolves_model_from_litellm_name(
    monkeypatch,
):
    """When no session event carries the ASR model, the litellm_model_name is used."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))

    results: OpenAIRealtimeStreamList = [
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "usage": {"type": "duration", "seconds": 120.0},
        },
    ]
    cost = handle_realtime_stream_cost_calculation(
        results=results,
        combined_usage_object=Usage(),
        custom_llm_provider="azure",
        litellm_model_name="azure/gpt-realtime-whisper",
    )
    assert abs(cost - 120.0 * (0.017 / 60)) < 1e-9


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


def test_realtime_transcription_token_billed_fallback(monkeypatch):
    """
    Token-billed transcription models price by audio/text tokens. Verify the
    fallback path multiplies audio tokens by the model's audio token cost.
    """
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))

    from litellm.cost_calculator import _transcription_usage_cost

    # gpt-4o-transcribe: input_cost_per_audio_token = 2.5e-06, input_cost_per_token = 2.5e-06,
    # output_cost_per_token = 1e-05
    model_info = litellm.get_model_info(
        model="gpt-4o-transcribe", custom_llm_provider="openai"
    )
    usage = {
        "type": "tokens",
        "input_tokens": 40,
        "output_tokens": 10,
        "total_tokens": 50,
        "input_token_details": {"audio_tokens": 30, "text_tokens": 10},
    }
    cost = _transcription_usage_cost(usage, model_info)
    expected = (
        30 * 2.5e-06  # audio tokens
        + 10 * 2.5e-06  # text tokens
        + 10 * 1e-05  # output tokens
    )
    assert abs(cost - expected) < 1e-12


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

    assert (
        result._hidden_params["response_cost"]
        > result_2._hidden_params["response_cost"]
    )

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
    assert litellm.model_cost["dashscope/qwen3.7-plus"].get("tiered_pricing") is None

    selected = _select_model_name_for_cost_calc(
        model="dashscope/qwen3.7-plus",
        completion_response=None,
        custom_pricing=True,
        custom_llm_provider="dashscope",
        router_model_id=router_model_id,
    )
    assert selected is not None
    assert router_model_id in selected


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


def test_azure_realtime_cost_calculator():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

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
            prompt_tokens_details=PromptTokensDetailsWrapper(
                text_tokens=10, audio_tokens=90
            ),
        ),
        custom_llm_provider="azure",
        litellm_model_name="my-custom-azure-deployment",
    )

    assert cost > 0


def test_azure_audio_output_cost_calculation():
    """
    Test that Azure audio models correctly calculate costs for audio output tokens.

    Reproduces issue: https://github.com/BerriAI/litellm/issues/19764
    Audio tokens should be charged at output_cost_per_audio_token rate,
    not at the text token rate (output_cost_per_token).
    """
    from litellm.types.utils import Choices, CompletionTokensDetailsWrapper, Message

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

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
    assert (
        abs(cost - wrong_total_cost) > 0.001
    ), "Bug: Audio tokens are being charged at text token rate"

    # Verify cost matches
    assert (
        abs(cost - expected_total_cost) < 0.0000001
    ), f"Expected cost {expected_total_cost}, got {cost}"


def test_default_image_cost_calculator(monkeypatch):
    from litellm.cost_calculator import default_image_cost_calculator

    temp_object = {
        "litellm_provider": "azure",
        "input_cost_per_pixel": 10,
    }

    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            "azure/bf9001cd7209f5734ecb4ab937a5a0e2ba5f119708bd68f184db362930f9dc7b": temp_object
        },
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


def test_gemini_25_implicit_caching_cost():
    """
    Test that Gemini 2.5 models correctly calculate costs with implicit caching.

    This test reproduces the issue from #11156 where cached tokens should receive
    a 75% discount.
    """
    from litellm import completion_cost
    from litellm.types.utils import (
        Choices,
        Message,
        ModelResponse,
        PromptTokensDetailsWrapper,
        Usage,
    )

    # Create a mock response similar to the one in the issue
    litellm_model_response = ModelResponse(
        id="test-response",
        created=1750733889,
        model="gemini/gemini-2.5-flash",
        object="chat.completion",
        system_fingerprint=None,
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    content="Understood. This is a test message to check the response from the Gemini model.",
                    role="assistant",
                    tool_calls=None,
                    function_call=None,
                ),
            )
        ],
        usage=Usage(
            total_tokens=15050,
            prompt_tokens=15033,
            completion_tokens=17,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                audio_tokens=None,
                cached_tokens=14316,  # This is cachedContentTokenCount from Gemini
            ),
            completion_tokens_details=None,
        ),
    )

    # Calculate the cost
    result = completion_cost(
        completion_response=litellm_model_response,
        model="gemini/gemini-2.5-flash",
    )

    # Current pricing for gemini/gemini-2.5-flash:
    # input: $0.30 / 1M tokens (3e-07 per token)
    # cache_read: $0.03 / 1M tokens (3e-08 per token)
    # output: $2.50 / 1M tokens (2.5e-06 per token)

    # Breakdown:
    # - Cached tokens: 14316 * 3e-08 = 0.00042948
    # - Non-cached tokens: (15033-14316) * 3e-07 = 717 * 3e-07 = 0.00021510
    # - Output tokens: 17 * 2.5e-06 = 0.00004250
    # Total: 0.00042948 + 0.00021510 + 0.00004250 = 0.00068708

    expected_cost = 0.00068708

    # Allow for small floating point differences
    assert (
        abs(result - expected_cost) < 1e-8
    ), f"Expected cost {expected_cost}, but got {result}"

    print(f"✓ Gemini 2.5 implicit caching cost calculation is correct: ${result:.8f}")


def test_log_context_cost_calculation():
    """
    Test that log context cost calculation works correctly with tiered pricing.

    This test verifies that when using extended context (above 200k tokens),
    the log context costs are calculated using the appropriate tiered rates.
    """
    from litellm import completion_cost
    from litellm.types.utils import (
        Choices,
        Message,
        ModelResponse,
        PromptTokensDetailsWrapper,
        Usage,
    )

    # Create a mock response with extended context usage
    extended_context_response = ModelResponse(
        id="test-extended-context-response",
        created=1750733889,
        model="claude-4-sonnet-20250514",
        object="chat.completion",
        system_fingerprint=None,
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    content="This is a test response for extended context cost calculation.",
                    role="assistant",
                    tool_calls=None,
                    function_call=None,
                ),
            )
        ],
        usage=Usage(
            total_tokens=350000,  # Above 200k threshold
            prompt_tokens=301000,  # Above 200k threshold
            completion_tokens=50000,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                text_tokens=300000,
                cached_tokens=0,  # No cache hits
                audio_tokens=None,
                image_tokens=None,
                character_count=None,
                video_length_seconds=None,
                cache_creation_tokens=1000,
            ),
            completion_tokens_details=None,
            _cache_creation_input_tokens=1000,  # Some tokens added to cache
        ),
    )

    # Calculate the cost using the extended context model
    result = completion_cost(
        completion_response=extended_context_response,
        model="claude-4-sonnet-20250514",
        custom_llm_provider="anthropic",
    )

    # Debug: Print the actual result
    print(f"DEBUG: Actual cost result: ${result:.6f}")

    # Get model info to understand the pricing
    from litellm import get_model_info

    model_info = get_model_info(
        model="claude-4-sonnet-20250514", custom_llm_provider="anthropic"
    )

    # Calculate expected cost based on actual model pricing
    input_cost_per_token = model_info.get("input_cost_per_token", 0)
    output_cost_per_token = model_info.get("output_cost_per_token", 0)
    cache_creation_cost_per_token = model_info.get("cache_creation_input_token_cost", 0)

    # Check if tiered pricing is applied
    input_cost_above_200k = model_info.get(
        "input_cost_per_token_above_200k_tokens", input_cost_per_token
    )
    output_cost_above_200k = model_info.get(
        "output_cost_per_token_above_200k_tokens", output_cost_per_token
    )
    cache_creation_above_200k = model_info.get(
        "cache_creation_input_token_cost_above_200k_tokens",
        cache_creation_cost_per_token,
    )

    print(f"DEBUG: Base input cost per token: ${input_cost_per_token:.2e}")
    print(f"DEBUG: Base output cost per token: ${output_cost_per_token:.2e}")
    print(
        f"DEBUG: Base cache creation cost per token: ${cache_creation_cost_per_token:.2e}"
    )

    # Handle tiered pricing - if not available, use base pricing
    if input_cost_above_200k is not None:
        print(
            f"DEBUG: Tiered input cost per token (>200k): ${input_cost_above_200k:.2e}"
        )
    else:
        print("DEBUG: No tiered input pricing available, using base pricing")
        input_cost_above_200k = input_cost_per_token

    if output_cost_above_200k is not None:
        print(
            f"DEBUG: Tiered output cost per token (>200k): ${output_cost_above_200k:.2e}"
        )
    else:
        print("DEBUG: No tiered output pricing available, using base pricing")
        output_cost_above_200k = output_cost_per_token

    if cache_creation_above_200k is not None:
        print(
            f"DEBUG: Tiered cache creation cost per token (>200k): ${cache_creation_above_200k:.2e}"
        )
    else:
        print("DEBUG: No tiered cache creation pricing available, using base pricing")
        cache_creation_above_200k = cache_creation_cost_per_token

    # Since we're above 200k tokens, we should use tiered pricing if available
    expected_input_cost = 300000 * input_cost_above_200k
    expected_output_cost = 50000 * output_cost_above_200k
    expected_cache_cost = 1000 * cache_creation_above_200k
    expected_total = expected_input_cost + expected_output_cost + expected_cache_cost

    print(f"DEBUG: Expected total: ${expected_total:.6f}")

    # Allow for small floating point differences
    assert (
        abs(result - expected_total) < 1e-6
    ), f"Expected cost ${expected_total:.6f}, but got ${result:.6f}"

    print(
        f"✓ Log context cost calculation with tiered pricing is correct: ${result:.6f}"
    )
    print(f"  - Input tokens (300k): ${expected_input_cost:.6f}")
    print(f"  - Output tokens (50k): ${expected_output_cost:.6f}")
    print(f"  - Cache creation (1k): ${expected_cache_cost:.6f}")
    print(f"  - Total: ${result:.6f}")


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
        + model_info["cache_read_input_token_cost"]
        * usage.prompt_tokens_details.cached_tokens
        + model_info["output_cost_per_token"] * usage.completion_tokens
    )

    print(
        f"model_info['input_cost_per_token']: {model_info['input_cost_per_token']}, usage.prompt_tokens_details.text_tokens: {usage.prompt_tokens_details.text_tokens}, model_info['cache_read_input_token_cost']: {model_info['cache_read_input_token_cost']}, model_info['output_cost_per_token']: {model_info['output_cost_per_token']}"
    )

    print(f"Expected actual cost: {expected_actual_cost}")

    assert expected_actual_cost == total_cost


def test_azure_ai_cache_cost_calculation():
    """
    Test that azure_ai provider correctly calculates cache costs using generic_cost_per_token.

    This verifies that azure_ai models with custom cache pricing in model_info
    will have their cache_creation_input_token_cost and cache_read_input_token_cost
    applied correctly.
    """
    from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
    from litellm.types.utils import PromptTokensDetailsWrapper, Usage

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

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

    assert (
        abs(input_cost - expected_input_cost) < 1e-10
    ), f"Input cost mismatch: got {input_cost}, expected {expected_input_cost}"
    assert (
        abs(output_cost - expected_output_cost) < 1e-10
    ), f"Output cost mismatch: got {output_cost}, expected {expected_output_cost}"


def test_cost_discount_vertex_ai():
    """
    Test that cost discount is applied correctly for Vertex AI provider
    """
    from litellm import completion_cost
    from litellm.types.utils import Usage

    # Save original config
    original_discount_config = litellm.cost_discount_config.copy()

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
    litellm.cost_discount_config = {}
    cost_without_discount = completion_cost(
        completion_response=response,
        model="vertex_ai/gemini-3-pro-preview",
        custom_llm_provider="vertex_ai",
    )

    # Set 5% discount for vertex_ai
    litellm.cost_discount_config = {"vertex_ai": 0.05}

    # Calculate cost with discount
    cost_with_discount = completion_cost(
        completion_response=response,
        model="vertex_ai/gemini-3-pro-preview",
        custom_llm_provider="vertex_ai",
    )

    # Restore original config
    litellm.cost_discount_config = original_discount_config

    # Verify discount is applied (5% off means 95% of original cost)
    expected_cost = cost_without_discount * 0.95
    assert cost_with_discount == pytest.approx(expected_cost, rel=1e-9)

    print("✓ Cost discount test passed:")
    print(f"  - Original cost: ${cost_without_discount:.6f}")
    print(f"  - Discounted cost (5% off): ${cost_with_discount:.6f}")
    print(f"  - Savings: ${cost_without_discount - cost_with_discount:.6f}")


def test_cost_discount_not_applied_to_other_providers():
    """
    Test that cost discount only applies to configured providers
    """
    from litellm import completion_cost
    from litellm.types.utils import Usage

    # Save original config
    original_discount_config = litellm.cost_discount_config.copy()

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
    litellm.cost_discount_config = {"vertex_ai": 0.05}

    # Calculate cost for OpenAI - should NOT have discount applied
    cost_with_selective_discount = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Clear discount config
    litellm.cost_discount_config = {}
    cost_without_discount = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Restore original config
    litellm.cost_discount_config = original_discount_config

    # Costs should be the same (no discount applied to OpenAI)
    assert cost_with_selective_discount == cost_without_discount

    print("✓ Selective discount test passed:")
    print(f"  - OpenAI cost (no discount configured): ${cost_without_discount:.6f}")
    print(f"  - Cost remains unchanged: ${cost_with_selective_discount:.6f}")


def test_cost_margin_percentage():
    """
    Test that percentage-based cost margin is applied correctly
    """
    from litellm import completion_cost
    from litellm.types.utils import Usage

    # Save original config
    original_margin_config = litellm.cost_margin_config.copy()

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
    litellm.cost_margin_config = {}
    cost_without_margin = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Set 10% margin for openai
    litellm.cost_margin_config = {"openai": 0.10}

    # Calculate cost with margin
    cost_with_margin = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Restore original config
    litellm.cost_margin_config = original_margin_config

    # Verify margin is applied (10% margin means 110% of original cost)
    expected_cost = cost_without_margin * 1.10
    assert cost_with_margin == pytest.approx(expected_cost, rel=1e-9)

    print("✓ Cost margin percentage test passed:")
    print(f"  - Original cost: ${cost_without_margin:.6f}")
    print(f"  - Cost with margin (10%): ${cost_with_margin:.6f}")
    print(f"  - Margin added: ${cost_with_margin - cost_without_margin:.6f}")


def test_cost_margin_fixed_amount():
    """
    Test that fixed amount cost margin is applied correctly
    """
    from litellm import completion_cost
    from litellm.types.utils import Usage

    # Save original config
    original_margin_config = litellm.cost_margin_config.copy()

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
    litellm.cost_margin_config = {}
    cost_without_margin = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Set $0.001 fixed margin for openai
    litellm.cost_margin_config = {"openai": {"fixed_amount": 0.001}}

    # Calculate cost with margin
    cost_with_margin = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Restore original config
    litellm.cost_margin_config = original_margin_config

    # Verify fixed margin is applied
    expected_cost = cost_without_margin + 0.001
    assert cost_with_margin == pytest.approx(expected_cost, rel=1e-9)

    print("✓ Cost margin fixed amount test passed:")
    print(f"  - Original cost: ${cost_without_margin:.6f}")
    print(f"  - Cost with margin ($0.001): ${cost_with_margin:.6f}")
    print(f"  - Margin added: ${cost_with_margin - cost_without_margin:.6f}")


def test_cost_margin_combined():
    """
    Test that combined percentage and fixed amount margin is applied correctly
    """
    from litellm import completion_cost
    from litellm.types.utils import Usage

    # Save original config
    original_margin_config = litellm.cost_margin_config.copy()

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
    litellm.cost_margin_config = {}
    cost_without_margin = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Set 8% margin + $0.0005 fixed for openai
    litellm.cost_margin_config = {
        "openai": {"percentage": 0.08, "fixed_amount": 0.0005}
    }

    # Calculate cost with margin
    cost_with_margin = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Restore original config
    litellm.cost_margin_config = original_margin_config

    # Verify combined margin is applied
    expected_cost = cost_without_margin * 1.08 + 0.0005
    assert cost_with_margin == pytest.approx(expected_cost, rel=1e-9)

    print("✓ Cost margin combined test passed:")
    print(f"  - Original cost: ${cost_without_margin:.6f}")
    print(f"  - Cost with margin (8% + $0.0005): ${cost_with_margin:.6f}")
    print(f"  - Margin added: ${cost_with_margin - cost_without_margin:.6f}")


def test_cost_margin_global():
    """
    Test that global margin is applied when no provider-specific margin is configured
    """
    from litellm import completion_cost
    from litellm.types.utils import Usage

    # Save original config
    original_margin_config = litellm.cost_margin_config.copy()

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
    litellm.cost_margin_config = {}
    cost_without_margin = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Set 5% global margin (no provider-specific margin)
    litellm.cost_margin_config = {"global": 0.05}

    # Calculate cost with global margin
    cost_with_global_margin = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Restore original config
    litellm.cost_margin_config = original_margin_config

    # Verify global margin is applied
    expected_cost = cost_without_margin * 1.05
    assert cost_with_global_margin == pytest.approx(expected_cost, rel=1e-9)

    print("✓ Cost margin global test passed:")
    print(f"  - Original cost: ${cost_without_margin:.6f}")
    print(f"  - Cost with global margin (5%): ${cost_with_global_margin:.6f}")
    print(f"  - Margin added: ${cost_with_global_margin - cost_without_margin:.6f}")


def test_cost_margin_provider_overrides_global():
    """
    Test that provider-specific margin overrides global margin
    """
    from litellm import completion_cost
    from litellm.types.utils import Usage

    # Save original config
    original_margin_config = litellm.cost_margin_config.copy()

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
    litellm.cost_margin_config = {}
    cost_without_margin = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Set 5% global margin and 10% provider-specific margin
    litellm.cost_margin_config = {"global": 0.05, "openai": 0.10}

    # Calculate cost - should use provider-specific margin (10%), not global (5%)
    cost_with_provider_margin = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Restore original config
    litellm.cost_margin_config = original_margin_config

    # Verify provider-specific margin is used (not global)
    expected_cost = cost_without_margin * 1.10  # 10% from provider, not 5% from global
    assert cost_with_provider_margin == pytest.approx(expected_cost, rel=1e-9)

    print("✓ Cost margin provider override test passed:")
    print(f"  - Original cost: ${cost_without_margin:.6f}")
    print(
        f"  - Cost with provider margin (10%, overrides 5% global): ${cost_with_provider_margin:.6f}"
    )
    print(f"  - Margin added: ${cost_with_provider_margin - cost_without_margin:.6f}")


def test_cost_margin_with_discount():
    """
    Test that margin is applied after discount (independent calculation)
    """
    from litellm import completion_cost
    from litellm.types.utils import Usage

    # Save original configs
    original_margin_config = litellm.cost_margin_config.copy()
    original_discount_config = litellm.cost_discount_config.copy()

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
    litellm.cost_margin_config = {}
    litellm.cost_discount_config = {}
    base_cost = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Set 5% discount and 10% margin
    litellm.cost_discount_config = {"openai": 0.05}
    litellm.cost_margin_config = {"openai": 0.10}

    # Calculate cost with both discount and margin
    cost_with_both = completion_cost(
        completion_response=response,
        model="gpt-4",
        custom_llm_provider="openai",
    )

    # Restore original configs
    litellm.cost_margin_config = original_margin_config
    litellm.cost_discount_config = original_discount_config

    # Verify: discount applied first, then margin
    # Base cost -> discount: base * 0.95 -> margin: (base * 0.95) * 1.10
    expected_cost = base_cost * 0.95 * 1.10
    assert cost_with_both == pytest.approx(expected_cost, rel=1e-9)

    print("✓ Cost margin with discount test passed:")
    print(f"  - Base cost: ${base_cost:.6f}")
    print(f"  - Cost with 5% discount + 10% margin: ${cost_with_both:.6f}")
    print(f"  - Expected: ${expected_cost:.6f}")


def test_azure_image_generation_cost_calculator():
    from unittest.mock import MagicMock

    from litellm.types.utils import (
        ImageObject,
        ImageResponse,
        ImageUsage,
        ImageUsageInputTokensDetails,
    )

    response_cost_calculator_kwargs = {
        "response_object": ImageResponse(
            created=1761785270,
            background=None,
            data=[
                ImageObject(
                    b64_json=None,
                    revised_prompt="A futuristic, techno-inspired green duck wearing cool modern sunglasses. The duck has a sleek, metallic appearance with glowing neon green accents, standing on a high-tech urban background with holographic billboards and illuminated city lights in the distance. The duck's feathers have a glossy, high-tech sheen, resembling a robotic design but still maintaining its avian features. The scene has a vibrant, cyberpunk aesthetic with a neon color palette.",
                    url="test-azure-blob-url-with-sas-token",
                )
            ],
            output_format=None,
            quality="hd",
            size=None,
            usage=ImageUsage(
                input_tokens=0,
                input_tokens_details=ImageUsageInputTokensDetails(
                    image_tokens=0, text_tokens=0
                ),
                output_tokens=0,
                total_tokens=0,
            ),
        ),
        "model": "azure/dall-e-3",
        "cache_hit": False,
        "custom_llm_provider": "azure",
        "base_model": "azure/dall-e-3",
        "call_type": "aimage_generation",
        "optional_params": {},
        "custom_pricing": False,
        "prompt": "",
        "standard_built_in_tools_params": {
            "web_search_options": None,
            "file_search": None,
        },
        "router_model_id": "6738c432ffc9b733597c6b86613ca20dc5f49bde591fd3d03e7cd6aa25bb241e",
        "litellm_logging_obj": MagicMock(),
        "service_tier": None,
    }

    cost = response_cost_calculator(**response_cost_calculator_kwargs)
    assert cost > 0.079


def test_completion_cost_extracts_service_tier_from_response():
    """Test that completion_cost extracts service_tier from completion_response object."""
    from litellm import completion_cost

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

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
    assert (
        0.45 <= flex_ratio <= 0.55
    ), f"Flex pricing should be ~50% of standard, got {flex_ratio:.2f}"


def test_completion_cost_extracts_service_tier_from_usage():
    """Test that completion_cost extracts service_tier from usage object."""
    from litellm import completion_cost

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    # Test with gpt-5-nano which has flex pricing
    model = "gpt-5-nano"

    # Create usage object with service_tier
    usage_with_service_tier = Usage(
        prompt_tokens=1000, completion_tokens=500, total_tokens=1500
    )
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
    usage_without_service_tier = Usage(
        prompt_tokens=1000, completion_tokens=500, total_tokens=1500
    )

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
    assert (
        0.45 <= flex_ratio <= 0.55
    ), f"Flex pricing should be ~50% of standard, got {flex_ratio:.2f}"


def test_completion_cost_service_tier_priority():
    """Test that service_tier extraction follows priority: optional_params > completion_response > usage."""
    from litellm import completion_cost

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

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
    assert (
        abs(cost_from_params - cost_from_usage) < 1e-6
    ), "Costs from params and usage should be similar (both flex)"


def test_completion_cost_service_tier_for_bedrock():
    """Test that Bedrock cost calculation applies service_tier-specific pricing."""
    from litellm import completion_cost

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

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


def test_completion_cost_service_tier_for_anthropic():
    """
    Anthropic priority-tier requests must be priced at the priority rate.

    Regression for LIT-3771: the Anthropic cost route dropped ``service_tier``,
    so priority requests (whose tier is reported on the response usage) were
    always billed at the standard rate. The tier is captured by the
    transformation and must flow through to ``generic_cost_per_token``.
    """
    from litellm import completion_cost
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

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


def test_completion_cost_anthropic_auto_tier_uses_served_priority_rate():
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

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

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


def test_completion_cost_non_string_service_tier_defers_to_served_tier():
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

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

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


def test_completion_cost_non_string_response_service_tier_defers_to_served_tier():
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

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

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
    response = ModelResponse(
        usage=usage, model=model, service_tier={"name": "priority"}
    )

    cost = completion_cost(
        completion_response=response,
        model=model,
        custom_llm_provider="anthropic",
    )

    expected_priority = 1000 * 6e-6 + 500 * 30e-6
    assert cost == pytest.approx(expected_priority)


def test_completion_cost_non_string_usage_service_tier_prices_standard():
    """
    Regression: a non-string ``service_tier`` on the usage object must not crash
    cost tracking.

    The dict reaches ``completion_cost`` via the usage extraction path with no
    concrete tier to defer to, so pricing falls back to the standard rate instead
    of raising ``AttributeError`` in ``_get_service_tier_cost_key``.
    """
    from litellm import completion_cost

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

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


def test_anthropic_cost_per_token_prices_cache_at_served_tier_with_multiplier():
    """
    Regression for the cache/tier interaction in the Anthropic geo/speed path.

    When a request is served at "priority" and also carries a geo/speed
    multiplier (here ``speed="fast"``), the cache portion is held out of the
    multiplier so it is not scaled. That held-out cache cost must use the
    served tier's cache rate; pricing it at the standard rate while the cache
    embedded in ``prompt_cost`` is priced at the priority rate leaves a
    ``(cache_priority - cache_standard)(multiplier - 1)`` billing error.
    """
    from litellm.llms.anthropic.cost_calculation import (
        cost_per_token as anthropic_cost_per_token,
    )
    from litellm.types.utils import PromptTokensDetailsWrapper, Usage

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

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

    prompt_cost, completion_cost = anthropic_cost_per_token(
        model=model, usage=usage, service_tier="priority"
    )

    # non-cache input priced at the priority rate and scaled by the fast
    # multiplier; the 200 cache-hit tokens priced at the priority cache rate
    # and held out of the multiplier
    expected_prompt = (1000 - 200) * 6e-6 * 2 + 200 * 0.6e-6
    expected_completion = 500 * 30e-6 * 2
    assert prompt_cost == pytest.approx(expected_prompt)
    assert completion_cost == pytest.approx(expected_completion)


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
    assert (
        usage.prompt_tokens_details.text_tokens == 9
    ), f"Expected text_tokens=9, got {usage.prompt_tokens_details.text_tokens}"

    # Image tokens should be non-cached image only: 258 - 258 = 0
    assert (
        usage.prompt_tokens_details.image_tokens == 0
    ), f"Expected image_tokens=0, got {usage.prompt_tokens_details.image_tokens}"

    # Total cached should match
    assert (
        usage.prompt_tokens_details.cached_tokens == 9651
    ), f"Expected cached_tokens=9651, got {usage.prompt_tokens_details.cached_tokens}"

    # MOST IMPORTANT: text_tokens should NEVER be negative
    assert (
        usage.prompt_tokens_details.text_tokens >= 0
    ), f"BUG: text_tokens is negative ({usage.prompt_tokens_details.text_tokens})! This was the issue in #18750"

    print(
        "✅ Issue #18750 fix verified: text_tokens is correctly calculated and non-negative"
    )


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


def test_gemini_implicit_caching_cost_calculation():
    """
    Test for Issue #16341: Gemini implicit cached tokens not counted in spend log

    When Gemini uses implicit caching, it returns cachedContentTokenCount but NOT
    cacheTokensDetails. In this case, we should subtract cachedContentTokenCount
    from text_tokens to correctly calculate costs.

    See: https://github.com/BerriAI/litellm/issues/16341
    """
    from litellm import completion_cost
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )
    from litellm.types.utils import Choices, Message, ModelResponse

    # Simulate Gemini response with implicit caching (cachedContentTokenCount only)
    completion_response = {
        "usageMetadata": {
            "promptTokenCount": 10000,
            "candidatesTokenCount": 5,
            "totalTokenCount": 10005,
            "cachedContentTokenCount": 8000,  # Implicit caching - no cacheTokensDetails
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 10000}],
            "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 5}],
        }
    }

    usage = VertexGeminiConfig._calculate_usage(completion_response)

    # Verify parsing
    assert (
        usage.cache_read_input_tokens == 8000
    ), f"cache_read_input_tokens should be 8000, got {usage.cache_read_input_tokens}"
    assert (
        usage.prompt_tokens_details.cached_tokens == 8000
    ), f"cached_tokens should be 8000, got {usage.prompt_tokens_details.cached_tokens}"

    # CRITICAL: text_tokens should be (10000 - 8000) = 2000, NOT 10000
    # This is the fix for issue #16341
    assert (
        usage.prompt_tokens_details.text_tokens == 2000
    ), f"text_tokens should be 2000 (10000 - 8000), got {usage.prompt_tokens_details.text_tokens}"

    # Verify cost calculation uses cached token pricing
    response = ModelResponse(
        id="mock-id",
        model="gemini-2.0-flash",
        choices=[
            Choices(
                index=0,
                message=Message(role="assistant", content="Hello!"),
                finish_reason="stop",
            )
        ],
        usage=usage,
    )

    cost = completion_cost(
        completion_response=response,
        model="gemini-2.0-flash",
        custom_llm_provider="gemini",
    )

    # Get model pricing for verification
    import litellm

    model_info = litellm.get_model_info("gemini/gemini-2.0-flash")
    input_cost = model_info.get("input_cost_per_token", 0)
    cache_read_cost = model_info.get("cache_read_input_token_cost", input_cost)
    output_cost = model_info.get("output_cost_per_token", 0)

    # Expected cost: (2000 * input) + (8000 * cache_read) + (5 * output)
    expected_cost = (2000 * input_cost) + (8000 * cache_read_cost) + (5 * output_cost)

    assert abs(cost - expected_cost) < 1e-9, (
        f"Cost calculation is wrong. Got ${cost:.6f}, expected ${expected_cost:.6f}. "
        f"Cached tokens may not be using reduced pricing."
    )

    print(
        "✅ Issue #16341 fix verified: Gemini implicit caching cost calculated correctly"
    )


def test_additional_costs_only_for_azure_ai():
    """
    Test that _get_additional_costs is only called for azure_ai provider.

    completion_cost() guards the call with `if custom_llm_provider == "azure_ai"`.
    This test verifies that non-azure_ai providers get additional_costs=None
    (reflected by the absence of "additional_costs" in cost_breakdown),
    while azure_ai providers can include additional costs.
    """
    from litellm.cost_calculator import _get_additional_costs

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

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


def test_openrouter_gemini_3_1_flash_lite_preview_pricing():
    """
    Test that openrouter/google/gemini-3.1-flash-lite-preview has a pricing entry.

    Regression test for https://github.com/BerriAI/litellm/issues/25604

    The model exists and is callable via OpenRouter, but was missing from
    model_prices_and_context_window.json when other Gemini 3.x variants were present.
    This caused ValueError: This model isn't mapped yet during router pre-call checks.
    """
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    model_name = "openrouter/google/gemini-3.1-flash-lite-preview"
    model_info = litellm.model_cost.get(model_name)

    assert model_info is not None, f"Missing model pricing entry: {model_name}"
    assert model_info["litellm_provider"] == "openrouter"
    assert model_info["input_cost_per_token"] == 2.5e-07
    assert model_info["output_cost_per_token"] == 1.5e-06
    assert model_info["max_input_tokens"] == 1048576
    assert model_info["max_output_tokens"] == 65536


def test_gemini_3_1_flash_lite_pricing():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    for model_name in (
        "gemini-3.1-flash-lite",
        "gemini/gemini-3.1-flash-lite",
        "vertex_ai/gemini-3.1-flash-lite",
    ):
        model_info = litellm.model_cost.get(model_name)
        assert model_info is not None, f"Missing model pricing entry: {model_name}"
        assert model_info["input_cost_per_token"] == 2.5e-07
        assert model_info["input_cost_per_audio_token"] == 5e-07
        assert model_info["output_cost_per_token"] == 1.5e-06
        assert model_info["output_cost_per_reasoning_token"] == 1.5e-06
        assert model_info["cache_read_input_token_cost"] == 2.5e-08
        assert model_info["max_input_tokens"] == 1048576


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

    expected = (
        (4000 - 1000 - 500) * 0.0000025
        + 1000 * 0.00000025
        + 500 * 0.000003125
        + 100 * 0.000015
    )

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

    expected_prompt = (
        (4000 - 1000 - 500) * 0.0000025 + 1000 * 0.00000025 + 500 * 0.000003125
    )
    expected_completion = 100 * 0.000015

    assert prompt_cost == pytest.approx(expected_prompt)
    assert completion_cost == pytest.approx(expected_completion)


# ---------------------------------------------------------------------------
# Bug 2 — db_spend_update_writer cache token extraction helpers.
# ---------------------------------------------------------------------------


def test_extract_cache_read_tokens_anthropic_top_level():
    from litellm.proxy.db.db_spend_update_writer import _extract_cache_read_tokens

    usage_obj = {
        "prompt_tokens": 100,
        "cache_read_input_tokens": 80,
        "prompt_tokens_details": {"cached_tokens": 80},
    }
    # Anthropic top-level value should win over prompt_tokens_details fallback.
    assert _extract_cache_read_tokens(usage_obj) == 80


def test_extract_cache_read_tokens_openai_compatible_fallback():
    from litellm.proxy.db.db_spend_update_writer import _extract_cache_read_tokens

    # Anthropic field absent — fall back to prompt_tokens_details.cached_tokens.
    usage_obj = {
        "prompt_tokens": 22583,
        "prompt_tokens_details": {"cached_tokens": 22016},
    }
    assert _extract_cache_read_tokens(usage_obj) == 22016


def test_extract_cache_read_tokens_zero_when_missing():
    from litellm.proxy.db.db_spend_update_writer import _extract_cache_read_tokens

    assert _extract_cache_read_tokens({}) == 0
    assert _extract_cache_read_tokens({"cache_read_input_tokens": None}) == 0
    assert (
        _extract_cache_read_tokens({"prompt_tokens_details": {"cached_tokens": None}})
        == 0
    )


def test_extract_cache_creation_tokens_anthropic_top_level():
    from litellm.proxy.db.db_spend_update_writer import (
        _extract_cache_creation_tokens,
    )

    usage_obj = {
        "prompt_tokens": 100,
        "cache_creation_input_tokens": 50,
        "prompt_tokens_details": {"cache_write_tokens": 50},
    }
    # Anthropic top-level should short-circuit the fallback.
    assert _extract_cache_creation_tokens(usage_obj) == 50


def test_extract_cache_creation_tokens_openai_cache_write_alias():
    from litellm.proxy.db.db_spend_update_writer import (
        _extract_cache_creation_tokens,
    )

    # kimi-k2 emits cache_write_tokens.
    usage_obj = {
        "prompt_tokens": 1000,
        "prompt_tokens_details": {"cache_write_tokens": 200},
    }
    assert _extract_cache_creation_tokens(usage_obj) == 200


def test_extract_cache_creation_tokens_openai_cache_creation_alias():
    from litellm.proxy.db.db_spend_update_writer import (
        _extract_cache_creation_tokens,
    )

    # Other OpenAI-compatible providers emit cache_creation_tokens.
    usage_obj = {
        "prompt_tokens": 1000,
        "prompt_tokens_details": {"cache_creation_tokens": 300},
    }
    assert _extract_cache_creation_tokens(usage_obj) == 300


def test_extract_cache_creation_tokens_zero_when_missing():
    from litellm.proxy.db.db_spend_update_writer import (
        _extract_cache_creation_tokens,
    )

    assert _extract_cache_creation_tokens({}) == 0
    assert _extract_cache_creation_tokens({"cache_creation_input_tokens": None}) == 0
    assert (
        _extract_cache_creation_tokens(
            {"prompt_tokens_details": {"cache_write_tokens": None}}
        )
        == 0
    )


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


def test_openrouter_gemini_3_1_flash_lite_stable_pricing():
    """
    Test that openrouter/google/gemini-3.1-flash-lite (stable, no -preview suffix)
    has a pricing entry.

    Google promoted gemini-3.1-flash-lite to GA on 2026-05-07. PR #27933 added the
    stable pricing for the bare, gemini/, and vertex_ai/ prefixes but missed the
    openrouter/google/ variant — every other Gemini family in the file has an
    openrouter/google/ sibling (2.0-flash-001, 2.5-flash, 2.5-pro, 3-flash-preview,
    3-pro-preview, 3.1-flash-lite-preview, 3.1-pro-preview), so the gap is a
    consistency issue, not a design choice. Same shape as the preview-variant gap
    fixed in PR #25610.

    Pricing matches the existing -preview entry one-for-one (input $0.25/M, output
    $1.50/M, cache-read $0.025/M) — Google did not change costs at the GA cutover.
    """
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    model_name = "openrouter/google/gemini-3.1-flash-lite"
    model_info = litellm.model_cost.get(model_name)

    assert model_info is not None, f"Missing model pricing entry: {model_name}"
    assert model_info["litellm_provider"] == "openrouter"
    assert model_info["input_cost_per_token"] == 2.5e-07
    assert model_info["output_cost_per_token"] == 1.5e-06
    assert model_info["cache_read_input_token_cost"] == 2.5e-08
    assert model_info["max_input_tokens"] == 1048576
    assert model_info["max_output_tokens"] == 65536


def test_completion_cost_logs_reasoning_and_cache_breakdown():
    """
    completion_cost must surface explicit reasoning and cache-read costs into the
    cost_breakdown stored on the logging object, so they end up in the spend logs
    rather than being silently folded into the output/input totals.
    """
    from datetime import datetime

    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.types.utils import Choices, CompletionTokensDetailsWrapper, Message

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    logging_obj = Logging(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "Hello"}],
        stream=False,
        call_type="completion",
        start_time=datetime.now(),
        litellm_call_id="reasoning-cache-breakdown",
        function_id="f",
    )

    response = ModelResponse(
        id="x",
        created=1,
        model="gemini-2.5-flash",
        object="chat.completion",
        choices=[
            Choices(
                index=0,
                message=Message(role="assistant", content="hi"),
                finish_reason="length",
            )
        ],
        usage=Usage(
            prompt_tokens=209,
            completion_tokens=3996,
            total_tokens=4205,
            completion_tokens_details=CompletionTokensDetailsWrapper(
                reasoning_tokens=3114, text_tokens=882
            ),
            prompt_tokens_details=PromptTokensDetailsWrapper(
                cached_tokens=100, text_tokens=109
            ),
        ),
    )

    litellm.completion_cost(
        completion_response=response,
        model="gemini-2.5-flash",
        custom_llm_provider="vertex_ai",
        litellm_logging_obj=logging_obj,
    )

    assert logging_obj.cost_breakdown is not None
    assert logging_obj.cost_breakdown["reasoning_cost"] == pytest.approx(3114 * 2.5e-06)
    assert logging_obj.cost_breakdown["cache_read_cost"] == pytest.approx(100 * 3e-08)


def test_cost_per_token_per_second_pricing(monkeypatch):
    """
    Models priced by duration (input/output_cost_per_second) with no per-token rates
    must be billed as cost_per_second * response_time_ms / 1000 in cost_per_token.
    """
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))

    model = "test-per-second-pricing-model"
    litellm.register_model(
        model_cost={
            model: {
                "input_cost_per_second": 0.02,
                "output_cost_per_second": 0.04,
                "litellm_provider": "together_ai",
                "mode": "chat",
            }
        }
    )

    prompt_cost, completion_cost_value = cost_per_token(
        model=model,
        custom_llm_provider="together_ai",
        prompt_tokens=10,
        completion_tokens=20,
        response_time_ms=1500.0,
    )

    assert prompt_cost == pytest.approx(0.02 * 1.5)
    assert completion_cost_value == pytest.approx(0.04 * 1.5)


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


def test_batch_cost_calculator_prices_cache_creation_tokens_at_cache_write_rate():
    """
    LIT-4008 regression: anthropic batch usage is dominated by cache tokens.
    Cache creation tokens must be priced at cache_creation_input_token_cost / 2,
    not folded into the base input rate, and must not also be billed as base
    input tokens.
    """
    from litellm.cost_calculator import batch_cost_calculator

    prompt_cost, completion_cost_value = batch_cost_calculator(
        usage=_batch_cache_usage(),
        model="claude-sonnet-4-5-20250929",
        custom_llm_provider="anthropic",
        model_info={  # type: ignore[arg-type]
            "input_cost_per_token": 3e-6,
            "output_cost_per_token": 15e-6,
            "cache_read_input_token_cost": 3e-7,
            "cache_creation_input_token_cost": 3.75e-6,
        },
    )

    assert prompt_cost == pytest.approx((1000 * 3e-6 + 8000 * 3e-7 + 2000 * 3.75e-6) / 2)
    assert completion_cost_value == pytest.approx(200 * 15e-6 / 2)


def test_batch_cost_calculator_cache_creation_falls_back_to_input_rate():
    from litellm.cost_calculator import batch_cost_calculator

    prompt_cost, _ = batch_cost_calculator(
        usage=_batch_cache_usage(),
        model="claude-sonnet-4-5-20250929",
        custom_llm_provider="anthropic",
        model_info={  # type: ignore[arg-type]
            "input_cost_per_token": 3e-6,
            "output_cost_per_token": 15e-6,
            "cache_read_input_token_cost": 3e-7,
        },
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
    expected = (
        100 * model_info["input_cost_per_token"]
        + 50 * model_info["output_cost_per_token"]
        + 25 * reasoning_rate
    )
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
