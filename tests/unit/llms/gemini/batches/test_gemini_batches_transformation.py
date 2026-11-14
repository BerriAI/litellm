"""
Regression tests for ``GeminiBatchesConfig``, the BaseBatchesConfig implementation behind
``custom_llm_provider="gemini"`` on /v1/batches.

Locks the ``models/{model}:batchGenerateContent`` create URL, the ``input_config.file_name``
request shape, Gemini -> OpenAI state mapping, ``batchStats`` -> ``request_counts``, bare ids at
the LiteLLM boundary (``batches/<id>`` and ``files/<id>`` stripped) and ``x-goog-api-key`` auth.
Everything runs for real against canned httpx responses; only the API key env var is set.
"""

import json

import httpx
import pytest

from litellm.llms.gemini.batches.transformation import (
    GeminiBatchesConfig,
    is_gemini_batch_usage_shape,
    is_native_gemini_batch_output_row,
    usage_from_batch_output,
)
from litellm.llms.gemini.common_utils import GeminiError
from litellm.types.utils import LiteLLMBatch, LlmProviders

STATUS_MAP = {
    "BATCH_STATE_UNSPECIFIED": "validating",
    "BATCH_STATE_PENDING": "validating",
    "BATCH_STATE_RUNNING": "in_progress",
    "BATCH_STATE_SUCCEEDED": "completed",
    "BATCH_STATE_FAILED": "failed",
    "BATCH_STATE_CANCELLED": "cancelled",
    "BATCH_STATE_EXPIRED": "expired",
}


def _operation(**metadata_overrides):
    metadata = {
        "@type": "type.googleapis.com/google.ai.generativelanguage.v1main.GenerateContentBatch",
        "model": "models/gemini-3.8-flash",
        "displayName": "nightly-eval",
        "inputConfig": {"fileName": "files/haf0y9b1kydd"},
        "output": {"responsesFile": "files/batch-la6uw1zzqrect7ugge7te7fueeslt1ms9xiw"},
        "createTime": "2026-11-10T21:18:34.714998843Z",
        "endTime": "2026-11-10T21:21:01.499150334Z",
        "updateTime": "2026-11-10T21:21:01.499150354Z",
        "batchStats": {"requestCount": "3", "successfulRequestCount": "2", "failedRequestCount": "1"},
        "state": "BATCH_STATE_SUCCEEDED",
        "name": "batches/la6uw1zzqrect7ugge7te7fueeslt1ms9xiw",
    }
    return {
        "name": "batches/la6uw1zzqrect7ugge7te7fueeslt1ms9xiw",
        "metadata": {**metadata, **metadata_overrides},
        "done": True,
    }


def _response(payload: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        content=json.dumps(payload).encode(),
        request=httpx.Request("GET", "https://generativelanguage.googleapis.com/v1beta/batches/x"),
    )


@pytest.fixture
def config() -> GeminiBatchesConfig:
    return GeminiBatchesConfig()


@pytest.fixture
def api_key(monkeypatch) -> str:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    return "AIza-test"


def test_custom_llm_provider(config):
    assert config.custom_llm_provider == LlmProviders.GEMINI


@pytest.mark.parametrize(
    "input_file_id",
    ["haf0y9b1kydd", "files/haf0y9b1kydd", "https://generativelanguage.googleapis.com/v1beta/files/haf0y9b1kydd"],
)
def test_create_request_names_the_uploaded_file_in_input_config(config, input_file_id):
    body = config.transform_create_batch_request(
        model="gemini-3.8-flash",
        create_batch_data={
            "input_file_id": input_file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
        optional_params={},
        litellm_params={},
    )
    assert body == {
        "batch": {"display_name": "litellm-haf0y9b1kydd", "input_config": {"file_name": "files/haf0y9b1kydd"}}
    }


def test_create_request_uses_display_name_from_metadata(config):
    body = config.transform_create_batch_request(
        model="gemini-3.8-flash",
        create_batch_data={
            "input_file_id": "files/abc",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
            "metadata": {"display_name": "nightly-eval"},
        },
        optional_params={},
        litellm_params={},
    )
    assert body["batch"]["display_name"] == "nightly-eval"


def test_create_request_requires_input_file(config):
    with pytest.raises(ValueError, match="input_file_id"):
        config.transform_create_batch_request(
            model="gemini-3.8-flash",
            create_batch_data={"endpoint": "/v1/chat/completions", "completion_window": "24h"},
            optional_params={},
            litellm_params={},
        )


@pytest.mark.parametrize(
    "api_base,model,expected",
    [
        (
            None,
            "gemini-3.8-flash",
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:batchGenerateContent",
        ),
        (
            "https://proxy.example/",
            "models/gemini-3.8-flash",
            "https://proxy.example/v1beta/models/gemini-3.8-flash:batchGenerateContent",
        ),
    ],
)
def test_create_url_targets_the_model_batch_route(config, api_base, model, expected):
    data = {"input_file_id": "files/abc", "endpoint": "/v1/chat/completions", "completion_window": "24h"}
    assert config.get_complete_batch_url(api_base, None, model, {}, {}, data) == expected


def test_validate_environment_uses_goog_api_key_header(config, api_key):
    headers = config.validate_environment({"x-trace": "1"}, "gemini-3.8-flash", [], {}, {})
    assert headers == {"x-trace": "1", "x-goog-api-key": api_key}


def test_validate_environment_explicit_key_wins(config, api_key):
    headers = config.validate_environment({}, "gemini-3.8-flash", [], {}, {}, api_key="AIza-deployment")
    assert headers["x-goog-api-key"] == "AIza-deployment"


def test_validate_environment_without_key_raises(config, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        config.validate_environment({}, "gemini-3.8-flash", [], {}, {})


def test_create_response_maps_operation_onto_openai_batch(config):
    batch = config.transform_create_batch_response(
        model="gemini-3.8-flash", raw_response=_response(_operation()), logging_obj=None, litellm_params={}
    )
    assert isinstance(batch, LiteLLMBatch)
    assert batch.model_dump(exclude_none=True) == {
        "id": "la6uw1zzqrect7ugge7te7fueeslt1ms9xiw",
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "input_file_id": "haf0y9b1kydd",
        "output_file_id": "batch-la6uw1zzqrect7ugge7te7fueeslt1ms9xiw",
        "completion_window": "24h",
        "status": "completed",
        "created_at": 1794345514,
        "completed_at": 1794345661,
        "request_counts": {"total": 3, "completed": 2, "failed": 1},
        "metadata": {"display_name": "nightly-eval"},
    }


def test_retrieve_request_is_presigned_get_with_bare_or_prefixed_id(config, api_key):
    for batch_id in ("la6", "batches/la6"):
        req = config.transform_retrieve_batch_request(
            batch_id=batch_id, optional_params={}, litellm_params={"api_base": "https://proxy.example"}
        )
        assert req == {
            "method": "GET",
            "url": "https://proxy.example/v1beta/batches/la6",
            "headers": {"x-goog-api-key": api_key},
        }, batch_id


def test_retrieve_request_prefers_litellm_params_api_key(config, api_key):
    req = config.transform_retrieve_batch_request(
        batch_id="la6", optional_params={}, litellm_params={"api_key": "AIza-deployment"}
    )
    assert req["headers"]["x-goog-api-key"] == "AIza-deployment"


@pytest.mark.parametrize("gemini_state,openai_status", sorted(STATUS_MAP.items()))
def test_retrieve_response_status_mapping(config, gemini_state, openai_status):
    batch = config.transform_retrieve_batch_response(
        model=None, raw_response=_response(_operation(state=gemini_state)), logging_obj=None, litellm_params={}
    )
    assert batch.status == openai_status


@pytest.mark.parametrize(
    "gemini_state,populated_field",
    [
        ("BATCH_STATE_SUCCEEDED", "completed_at"),
        ("BATCH_STATE_FAILED", "failed_at"),
        ("BATCH_STATE_EXPIRED", "expired_at"),
        ("BATCH_STATE_CANCELLED", "cancelled_at"),
    ],
)
def test_end_time_lands_on_the_terminal_timestamp_for_its_state(config, gemini_state, populated_field):
    batch = config.transform_retrieve_batch_response(
        model=None, raw_response=_response(_operation(state=gemini_state)), logging_obj=None, litellm_params={}
    )
    timestamps = {f: getattr(batch, f) for f in ("completed_at", "failed_at", "expired_at", "cancelled_at")}
    assert timestamps == {f: (1794345661 if f == populated_field else None) for f in timestamps}


def test_running_batch_has_no_output_file_or_terminal_timestamp(config):
    running = _operation(state="BATCH_STATE_RUNNING", batchStats={"requestCount": "3", "pendingRequestCount": "3"})
    del running["metadata"]["output"]
    del running["metadata"]["endTime"]
    batch = config.transform_retrieve_batch_response(
        model=None, raw_response=_response(running), logging_obj=None, litellm_params={}
    )
    assert batch.status == "in_progress"
    assert batch.output_file_id is None
    assert batch.completed_at is None
    assert batch.request_counts.model_dump() == {"total": 3, "completed": 0, "failed": 0}


def test_failed_operation_error_is_reported_in_errors(config):
    failed = {**_operation(state="BATCH_STATE_FAILED"), "error": {"code": 3, "message": "Input file is empty."}}
    batch = config.transform_retrieve_batch_response(
        model=None, raw_response=_response(failed), logging_obj=None, litellm_params={}
    )
    assert batch.status == "failed"
    assert [(e.code, e.message) for e in batch.errors.data] == [("3", "Input file is empty.")]


def test_error_class_is_gemini_error(config):
    err = config.get_error_class("quota", 429, {"retry-after": "7"})
    assert isinstance(err, GeminiError)
    assert (err.status_code, err.message) == (429, "quota")


# Gemini writes the usage of an OpenAI-shaped output row as camelCase, with thinking tokens
# counted in totalTokens but not in completionTokens (observed 2026-11-10 on gemini-3.8-flash).
@pytest.mark.parametrize(
    "reported,expected",
    [
        (
            {"promptTokens": 9, "completionTokens": 1, "totalTokens": 75},
            {"prompt_tokens": 9, "completion_tokens": 66, "total_tokens": 75, "reasoning_tokens": 65},
        ),
        (
            {"promptTokens": 9, "completionTokens": 4, "totalTokens": 13},
            {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13, "reasoning_tokens": None},
        ),
    ],
)
def test_batch_output_usage_folds_thinking_tokens_into_completion(reported, expected):
    usage = usage_from_batch_output(reported)
    details = usage.completion_tokens_details
    assert {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "reasoning_tokens": details.reasoning_tokens if details else None,
    } == expected


@pytest.mark.parametrize(
    "usage_object,expected",
    [({"promptTokens": 9}, True), ({"prompt_tokens": 9, "completion_tokens": 1}, False), ({}, False)],
)
def test_batch_usage_shape_detection(usage_object, expected):
    assert is_gemini_batch_usage_shape(usage_object) is expected


def test_create_request_rejects_metadata_gemini_cannot_carry(config):
    with pytest.raises(ValueError, match=r"\['run_id'\]"):
        config.transform_create_batch_request(
            model="gemini-3.8-flash",
            create_batch_data={
                "input_file_id": "files/abc",
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
                "metadata": {"display_name": "nightly", "run_id": "42"},
            },
            optional_params={},
            litellm_params={},
        )


def test_native_output_row_detection():
    native = {"key": "r0", "response": {"candidates": [], "usageMetadata": {"promptTokenCount": 3}}}
    openai_shaped = {"custom_id": "r0", "response": {"status_code": 200, "body": {"usage": {"promptTokens": 3}}}}
    assert (is_native_gemini_batch_output_row(native), is_native_gemini_batch_output_row(openai_shaped)) == (
        True,
        False,
    )
