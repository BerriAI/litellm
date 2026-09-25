import json

import pytest

import litellm
import litellm.batches.batch_utils as bu
from litellm.types.llms.openai import Batch

GROUNDED_USAGE_METADATA = {
    "promptTokenCount": 19,
    "candidatesTokenCount": 59,
    "thoughtsTokenCount": 406,
    "toolUsePromptTokenCount": 73,
    "totalTokenCount": 557,
    "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 19}],
    "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 59}],
    "toolUsePromptTokensDetails": [{"modality": "TEXT", "tokenCount": 73}],
    "trafficType": "ON_DEMAND",
}
PASSTHROUGH_OUTPUT_URI = (
    "gs://litellm-bucket/litellm-vertex-files/passthrough/publishers/google/models/gemini-2.5-flash/u/"
    "predictions.jsonl"
)
UNGROUNDED_USAGE_METADATA = {
    "promptTokenCount": 20,
    "candidatesTokenCount": 48,
    "thoughtsTokenCount": 195,
    "toolUsePromptTokenCount": 73,
    "totalTokenCount": 336,
    "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 20}],
    "trafficType": "ON_DEMAND",
}


def _batch(output_file_id: str) -> Batch:
    return Batch(
        id="b",
        completion_window="24h",
        created_at=1,
        endpoint="/v1/chat/completions",
        input_file_id="f",
        object="batch",
        status="completed",
        output_file_id=output_file_id,
    )


def _vertex_jsonl(rows: list[dict]) -> bytes:
    return "\n".join(json.dumps(row) for row in rows).encode()


def _vertex_openai_row(custom_id: str, model: str, prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "id": f"batch_req_{custom_id}",
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "request_id": custom_id,
            "body": {
                "id": f"chatcmpl-{custom_id}",
                "object": "chat.completion",
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            },
        },
        "error": None,
    }


def _native_vertex_row(usage_metadata: dict, *, grounded: bool, model_version: str | None = "gemini-2.5-flash"):
    candidate = {"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}
    grounding = {"groundingMetadata": {"webSearchQueries": ["q"]}} if grounded else {}
    response = {"candidates": [{**candidate, **grounding}], "usageMetadata": usage_metadata}
    return {
        "request": {"contents": [{"role": "user", "parts": [{"text": "q"}]}], "tools": [{"googleSearch": {}}]},
        "status": "",
        "response": {**response, **({"modelVersion": model_version} if model_version else {})},
        "processed_time": "2026-09-23T19:02:00.000+00:00",
    }


def _capture_cost_calls(monkeypatch, prompt_cost=0.5, completion_cost=0.25) -> list:
    import litellm.cost_calculator as cc

    calls: list = []

    def _calc(**kw):
        calls.append(kw)
        return (prompt_cost, completion_cost)

    monkeypatch.setattr(cc, "batch_cost_calculator", _calc)
    return calls


def test_vertex_native_cost_bills_embedding_rows(monkeypatch):
    monkeypatch.setitem(litellm.model_cost, "vertex_ai/gemini-embedding-2", {"input_cost_per_token_batches": 1e-7})
    rows = [
        {
            "key": "id_1",
            "status": "",
            "request": {"content": {"parts": [{"text": "hello world"}]}},
            "response": {"embedding": {"values": [0.1, 0.2]}, "usageMetadata": {"promptTokenCount": 2}},
        },
        {
            "key": "id_2",
            "status": "",
            "request": {"content": {"parts": [{"text": "hello"}]}},
            "response": {"embedding": {"values": [0.3]}, "tokenCount": "3"},
        },
        {"key": "id_3", "status": "INVALID_ARGUMENT", "request": {"content": {"parts": [{"text": ""}]}}},
    ]

    result = bu.calculate_vertex_ai_batch_cost_and_usage(rows, "gemini-embedding-2")

    assert (result.successful_requests, result.failed_requests) == (2, 1)
    assert (result.usage.prompt_tokens, result.usage.completion_tokens, result.usage.total_tokens) == (5, 0, 5)
    assert result.cost == pytest.approx(5 * 1e-7)
    assert result.models == ["gemini-embedding-2"]


@pytest.mark.asyncio
async def test_native_vertex_rows_route_to_vertex_cost_path_without_flag(monkeypatch):
    monkeypatch.setattr(litellm, "disable_vertex_batch_output_transformation", False, raising=False)
    monkeypatch.setattr(
        bu, "_aggregate_batch_cost_usage_models", lambda **kw: pytest.fail("generic path should not run")
    )
    calls = _capture_cost_calls(monkeypatch)
    rows = [
        _native_vertex_row(GROUNDED_USAGE_METADATA, grounded=True),
        _native_vertex_row(UNGROUNDED_USAGE_METADATA, grounded=False),
    ]

    result = await bu.calculate_batch_cost_and_usage(
        file_content_dictionary=rows, custom_llm_provider="vertex_ai", model_name="gemini-2.5-flash"
    )

    assert result.cost == pytest.approx(1.5)
    assert (result.successful_requests, result.failed_requests) == (2, 0)
    assert result.models == ["gemini-2.5-flash"]
    assert {(call["model"], call["custom_llm_provider"]) for call in calls} == {("gemini-2.5-flash", "vertex_ai")}


@pytest.mark.asyncio
async def test_openai_shaped_vertex_rows_keep_the_generic_path_without_flag(monkeypatch):
    monkeypatch.setattr(litellm, "disable_vertex_batch_output_transformation", False, raising=False)
    monkeypatch.setattr(
        bu, "calculate_vertex_ai_batch_cost_and_usage", lambda *a, **kw: pytest.fail("native path should not run")
    )
    _capture_cost_calls(monkeypatch)
    rows = [_vertex_openai_row("request-1", "gemini-2.5-flash", 10, 5)]

    result = await bu.calculate_batch_cost_and_usage(
        file_content_dictionary=rows, custom_llm_provider="vertex_ai", model_name="gemini-2.5-flash"
    )

    assert result.successful_requests == 1


@pytest.mark.asyncio
async def test_native_vertex_rows_on_another_provider_keep_the_generic_path(monkeypatch):
    monkeypatch.setattr(
        bu, "calculate_vertex_ai_batch_cost_and_usage", lambda *a, **kw: pytest.fail("native path should not run")
    )
    _capture_cost_calls(monkeypatch)

    result = await bu.calculate_batch_cost_and_usage(
        file_content_dictionary=[_native_vertex_row(GROUNDED_USAGE_METADATA, grounded=True)],
        custom_llm_provider="openai",
    )

    assert result.successful_requests == 0


@pytest.mark.asyncio
async def test_handle_completed_batch_routes_native_rows_without_flag(monkeypatch):
    monkeypatch.setattr(litellm, "disable_vertex_batch_output_transformation", False, raising=False)
    raw_rows = [_native_vertex_row(GROUNDED_USAGE_METADATA, grounded=True)]

    async def fake_fetch(batch, custom_llm_provider, litellm_params=None):
        return _vertex_jsonl(raw_rows)

    monkeypatch.setattr(bu, "_fetch_batch_output_file_content", fake_fetch)
    monkeypatch.setattr(
        bu, "_aggregate_batch_cost_usage_models", lambda **kw: pytest.fail("generic path should not run")
    )
    calls = _capture_cost_calls(monkeypatch, prompt_cost=0.7, completion_cost=0.3)
    deployment_model_info = {"input_cost_per_token_batches": 1e-6, "output_cost_per_token_batches": 2e-6}

    result = await bu._handle_completed_batch(
        _batch(PASSTHROUGH_OUTPUT_URI),
        custom_llm_provider="vertex_ai",
        model_name="gemini-2.5-flash",
        model_info=deployment_model_info,
    )

    assert result.cost == pytest.approx(1.0)
    assert result.usage.total_tokens == 557
    assert [call["model_info"] for call in calls] == [deployment_model_info]


def test_native_vertex_usage_is_billed_like_the_online_path(monkeypatch):
    calls = _capture_cost_calls(monkeypatch)
    grounded = _native_vertex_row(GROUNDED_USAGE_METADATA, grounded=True)
    ungrounded = _native_vertex_row(UNGROUNDED_USAGE_METADATA, grounded=False)

    result = bu.calculate_vertex_ai_batch_cost_and_usage([grounded, ungrounded], "gemini-2.5-flash")

    grounded_usage, ungrounded_usage = (call["usage"] for call in calls)
    assert grounded_usage.prompt_tokens == 19
    assert grounded_usage.completion_tokens == 59 + 406
    assert grounded_usage.completion_tokens_details.reasoning_tokens == 406
    assert ungrounded_usage.prompt_tokens == 20 + 73
    assert ungrounded_usage.completion_tokens == 48 + 195
    assert (result.usage.prompt_tokens, result.usage.completion_tokens, result.usage.total_tokens) == (
        19 + 93,
        465 + 243,
        557 + 336,
    )


def test_native_vertex_rows_are_priced_by_model_version_without_a_model_name(monkeypatch):
    calls = _capture_cost_calls(monkeypatch)
    rows = [
        _native_vertex_row(GROUNDED_USAGE_METADATA, grounded=True, model_version="gemini-2.5-flash"),
        _native_vertex_row(UNGROUNDED_USAGE_METADATA, grounded=False, model_version="gemini-2.5-pro"),
        _native_vertex_row(UNGROUNDED_USAGE_METADATA, grounded=False, model_version=None),
    ]

    result = bu.calculate_vertex_ai_batch_cost_and_usage(rows)

    assert [call["model"] for call in calls] == ["gemini-2.5-flash", "gemini-2.5-pro"]
    assert result.models == ["gemini-2.5-flash", "gemini-2.5-pro"]
    assert result.cost == pytest.approx(1.5)
    assert result.successful_requests == 3
    assert result.usage.total_tokens == 557 + 336 + 336


def test_native_vertex_rows_without_usage_metadata_count_as_failed(monkeypatch):
    _capture_cost_calls(monkeypatch)
    rows = [
        {"request": {"contents": []}, "status": "Error: bad request", "processed_time": "t"},
        {"request": {"contents": []}, "response": {"candidates": []}},
        _native_vertex_row(GROUNDED_USAGE_METADATA, grounded=True),
    ]

    result = bu.calculate_vertex_ai_batch_cost_and_usage(rows, "gemini-2.5-flash")

    assert (result.successful_requests, result.failed_requests) == (1, 2)
    assert result.usage.total_tokens == 557


def test_native_vertex_batch_whose_rows_all_failed_still_names_the_deployment_model(monkeypatch):
    calls = _capture_cost_calls(monkeypatch)
    rows = [{"request": {"contents": []}, "status": "Error: quota exceeded", "processed_time": "t"}] * 2

    result = bu.calculate_vertex_ai_batch_cost_and_usage(rows, "gemini-2.5-flash")

    assert result.models == ["gemini-2.5-flash"]
    assert (result.successful_requests, result.failed_requests, result.cost) == (0, 2, 0.0)
    assert calls == []


def test_native_vertex_rows_are_priced_with_the_deployment_model_info(monkeypatch):
    calls = _capture_cost_calls(monkeypatch)
    deployment_model_info = {"input_cost_per_token_batches": 1e-6, "output_cost_per_token_batches": 2e-6}

    bu.calculate_vertex_ai_batch_cost_and_usage(
        [_native_vertex_row(GROUNDED_USAGE_METADATA, grounded=True)],
        "gemini-2.5-flash",
        model_info=deployment_model_info,
    )

    assert [call["model_info"] for call in calls] == [deployment_model_info]


@pytest.mark.asyncio
async def test_native_vertex_rows_keep_the_deployment_model_info_through_the_batch_entrypoint(monkeypatch):
    calls = _capture_cost_calls(monkeypatch)
    deployment_model_info = {"input_cost_per_token_batches": 1e-6}

    await bu.calculate_batch_cost_and_usage(
        file_content_dictionary=[_native_vertex_row(GROUNDED_USAGE_METADATA, grounded=True)],
        custom_llm_provider="vertex_ai",
        model_name="gemini-2.5-flash",
        model_info=deployment_model_info,
    )

    assert [call["model_info"] for call in calls] == [deployment_model_info]


def test_native_vertex_rows_are_priced_by_the_deployment_model_over_model_version(monkeypatch):
    calls = _capture_cost_calls(monkeypatch)
    rows = [_native_vertex_row(GROUNDED_USAGE_METADATA, grounded=True, model_version="gemini-2.5-pro")]

    result = bu.calculate_vertex_ai_batch_cost_and_usage(rows, "gemini-2.5-flash")

    assert [call["model"] for call in calls] == ["gemini-2.5-flash"]
    assert result.models == ["gemini-2.5-flash"]


def test_native_vertex_rows_that_fail_response_validation_count_as_failed(monkeypatch):
    calls = _capture_cost_calls(monkeypatch)
    rows = [
        {"request": {"contents": []}, "response": {"candidates": "nope", "usageMetadata": GROUNDED_USAGE_METADATA}},
        _native_vertex_row(GROUNDED_USAGE_METADATA, grounded=True),
    ]

    result = bu.calculate_vertex_ai_batch_cost_and_usage(rows, "gemini-2.5-flash")

    assert (result.successful_requests, result.failed_requests) == (1, 1)
    assert result.usage.total_tokens == 557
    assert len(calls) == 1


@pytest.mark.parametrize("wildcard_model", ["*", "vertex_ai/*"])
def test_native_vertex_rows_under_a_wildcard_deployment_are_priced_by_model_version(monkeypatch, wildcard_model):
    calls = _capture_cost_calls(monkeypatch)
    rows = [
        _native_vertex_row(GROUNDED_USAGE_METADATA, grounded=True, model_version="gemini-2.5-flash"),
        _native_vertex_row(UNGROUNDED_USAGE_METADATA, grounded=False, model_version=None),
    ]

    result = bu.calculate_vertex_ai_batch_cost_and_usage(rows, wildcard_model)

    assert [call["model"] for call in calls] == ["gemini-2.5-flash", wildcard_model]
    assert result.cost == pytest.approx(1.5)
    assert (result.successful_requests, result.failed_requests) == (2, 0)
    assert result.usage.total_tokens == 557 + 336


def test_native_vertex_row_without_model_version_under_a_wildcard_deployment_bills_its_explicit_prices():
    deployment_model_info = {"input_cost_per_token_batches": 1e-6, "output_cost_per_token_batches": 2e-6}
    with_version = _native_vertex_row(GROUNDED_USAGE_METADATA, grounded=True, model_version="gemini-2.5-flash")
    without_version = _native_vertex_row(GROUNDED_USAGE_METADATA, grounded=True, model_version=None)

    twin = bu.calculate_vertex_ai_batch_cost_and_usage([with_version], "vertex_ai/*", model_info=deployment_model_info)
    both = bu.calculate_vertex_ai_batch_cost_and_usage(
        [with_version, without_version], "vertex_ai/*", model_info=deployment_model_info
    )

    assert twin.cost > 0
    assert both.cost == pytest.approx(2 * twin.cost)
    assert (both.successful_requests, both.failed_requests) == (2, 0)


def test_native_vertex_row_the_cost_map_cannot_price_is_billed_at_zero_and_the_rest_still_bills(monkeypatch):
    import litellm.cost_calculator as cc

    def _calc(**kw):
        if kw["model"] == "gemini-unpriced":
            raise ValueError("no pricing")
        return (0.5, 0.25)

    monkeypatch.setattr(cc, "batch_cost_calculator", _calc)
    rows = [
        _native_vertex_row(GROUNDED_USAGE_METADATA, grounded=True, model_version="gemini-unpriced"),
        _native_vertex_row(UNGROUNDED_USAGE_METADATA, grounded=False, model_version="gemini-2.5-flash"),
    ]

    result = bu.calculate_vertex_ai_batch_cost_and_usage(rows)

    assert result.cost == pytest.approx(0.75)
    assert (result.successful_requests, result.failed_requests) == (2, 0)
    assert result.usage.total_tokens == 557 + 336
    assert result.models == ["gemini-unpriced", "gemini-2.5-flash"]


@pytest.mark.asyncio
async def test_flag_sends_every_vertex_row_down_the_native_path_when_a_model_is_known(monkeypatch):
    monkeypatch.setattr(litellm, "disable_vertex_batch_output_transformation", True, raising=False)
    monkeypatch.setattr(
        bu, "_aggregate_batch_cost_usage_models", lambda **kw: pytest.fail("generic path should not run")
    )
    calls = _capture_cost_calls(monkeypatch)
    rows = [_vertex_openai_row("request-1", "gemini-2.5-flash", 10, 5)]

    result = await bu.calculate_batch_cost_and_usage(
        file_content_dictionary=rows, custom_llm_provider="vertex_ai", model_name="gemini-2.5-flash"
    )

    assert calls == []
    assert (result.successful_requests, result.failed_requests) == (0, 1)
