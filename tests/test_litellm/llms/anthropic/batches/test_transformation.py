"""
Tests for AnthropicBatchesConfig create/list/cancel transforms and the
OpenAI <-> Anthropic batch line translators in
litellm/llms/anthropic/batches/transformation.py
"""

from unittest.mock import MagicMock

import httpx
import pytest

from litellm.llms.anthropic.batches.transformation import (
    AnthropicBatchesConfig,
    transform_anthropic_batch_result_line,
    transform_openai_batch_lines_to_anthropic_requests,
)
from litellm.llms.anthropic.common_utils import AnthropicError
from litellm.types.utils import LiteLLMBatch


@pytest.fixture
def config():
    return AnthropicBatchesConfig()


def _response(payload, status_code=200, method="POST", url="https://api.anthropic.com/v1/messages/batches"):
    return httpx.Response(
        status_code=status_code,
        json=payload,
        request=httpx.Request(method, url),
    )


# =========================================================================== #
# transform_create_batch_request
# =========================================================================== #


def test_create_batch_request_posts_requests_inline(config):
    requests = [{"custom_id": "r1", "params": {"model": "claude-sonnet-4-5", "max_tokens": 10, "messages": []}}]
    body = config.transform_create_batch_request(
        model="claude-sonnet-4-5",
        create_batch_data={
            "completion_window": "24h",
            "endpoint": "/v1/chat/completions",
            "input_file_id": "none",
            "extra_body": {"requests": requests, "extra_flag": "keep-me"},
        },
        optional_params={},
        litellm_params={},
    )
    assert body["requests"] == requests
    assert body["extra_flag"] == "keep-me"


def test_create_batch_request_missing_requests_raises_400(config):
    with pytest.raises(AnthropicError) as exc_info:
        config.transform_create_batch_request(
            model="claude-sonnet-4-5",
            create_batch_data={
                "completion_window": "24h",
                "endpoint": "/v1/chat/completions",
                "input_file_id": "file-abc",
            },
            optional_params={},
            litellm_params={},
        )
    assert exc_info.value.status_code == 400
    assert "requests" in exc_info.value.message


# =========================================================================== #
# transform_openai_batch_lines_to_anthropic_requests
# =========================================================================== #


def test_openai_lines_translate_to_anthropic_params():
    lines = [
        {
            "custom_id": "req-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "ignored-line-model",
                "messages": [
                    {"role": "system", "content": "You are terse."},
                    {"role": "user", "content": "Hi"},
                ],
                "temperature": 0.5,
                "max_tokens": 64,
            },
        },
        {
            "custom_id": "req-2",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "ignored-line-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        },
    ]
    requests = transform_openai_batch_lines_to_anthropic_requests(lines, model="claude-sonnet-4-5")

    assert [r["custom_id"] for r in requests] == ["req-1", "req-2"]
    first, second = requests
    assert first["params"]["model"] == "claude-sonnet-4-5"
    assert first["params"]["temperature"] == 0.5
    assert first["params"]["max_tokens"] == 64
    # the system message is pulled out of messages into params["system"]
    assert first["params"]["system"]
    assert all(m["role"] != "system" for m in first["params"]["messages"])
    # max_tokens is required by Anthropic and defaulted even when absent
    assert second["params"]["max_tokens"] == 64000
    assert second["params"]["model"] == "claude-sonnet-4-5"
    # params are isolated per line: req-1's temperature must not leak into req-2
    assert "temperature" not in second["params"]


def test_openai_lines_params_do_not_leak_across_calls():
    transform_openai_batch_lines_to_anthropic_requests(
        [
            {
                "custom_id": "req-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "ignored",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "temperature": 0.2,
                    "max_tokens": 7,
                },
            }
        ],
        model="claude-sonnet-4-5",
    )
    (second_run,) = transform_openai_batch_lines_to_anthropic_requests(
        [
            {
                "custom_id": "req-2",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": "ignored", "messages": [{"role": "user", "content": "Hello"}]},
            }
        ],
        model="claude-sonnet-4-5",
    )
    assert "temperature" not in second_run["params"]
    assert second_run["params"]["max_tokens"] == 64000


def test_openai_line_missing_custom_id_raises():
    with pytest.raises(ValueError, match="custom_id"):
        transform_openai_batch_lines_to_anthropic_requests(
            [{"body": {"messages": [{"role": "user", "content": "hi"}]}}],
            model="claude-sonnet-4-5",
        )


def test_openai_line_missing_messages_raises_with_custom_id():
    with pytest.raises(TypeError, match="req-9"):
        transform_openai_batch_lines_to_anthropic_requests(
            [{"custom_id": "req-9", "body": {"model": "x"}}],
            model="claude-sonnet-4-5",
        )


def test_transform_stored_batch_input_builds_requests_and_keeps_extra_body(config):
    result = config.transform_stored_batch_input(
        model="claude-sonnet-4-5",
        endpoint="/v1/chat/completions",
        lines=(
            {
                "custom_id": "req-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "ignored",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 8,
                },
            },
        ),
        extra_body={"tag": "nightly"},
    )

    assert result["tag"] == "nightly"
    requests = result["requests"]
    assert isinstance(requests, list)
    assert [request["custom_id"] for request in requests] == ["req-1"]
    assert requests[0]["params"]["model"] == "claude-sonnet-4-5"
    assert requests[0]["params"]["max_tokens"] == 8


def test_transform_stored_batch_input_rejects_non_chat_endpoint(config):
    with pytest.raises(ValueError, match="/v1/chat/completions"):
        config.transform_stored_batch_input(
            model="claude-sonnet-4-5",
            endpoint="/v1/embeddings",
            lines=(),
            extra_body=None,
        )


# =========================================================================== #
# transform_create_batch_response
# =========================================================================== #


def test_create_batch_response_maps_to_litellm_batch(config):
    raw = _response(
        {
            "id": "msgbatch_new",
            "processing_status": "in_progress",
            "created_at": "2024-09-24T10:00:00Z",
            "request_counts": {"processing": 2},
        }
    )
    batch = config.transform_create_batch_response(
        model="claude-sonnet-4-5", raw_response=raw, logging_obj=MagicMock(), litellm_params={}
    )
    assert isinstance(batch, LiteLLMBatch)
    assert batch.id == "msgbatch_new"
    assert batch.status == "in_progress"
    assert batch.output_file_id == "msgbatch_new"


def test_create_batch_response_error_raises_anthropic_error(config):
    raw = _response({"error": {"type": "invalid_request_error", "message": "bad"}}, status_code=400)
    with pytest.raises(AnthropicError) as exc_info:
        config.transform_create_batch_response(model=None, raw_response=raw, logging_obj=MagicMock(), litellm_params={})
    assert exc_info.value.status_code == 400


# =========================================================================== #
# list batches
# =========================================================================== #


def test_get_list_batches_url_encodes_limit_and_after_id(config):
    url = config.get_list_batches_url(
        api_base="https://api.anthropic.com",
        api_key="sk-ant-test",
        model="",
        optional_params={},
        litellm_params={},
        after="msgbatch_cursor",
        limit=10,
    )
    parsed = httpx.URL(url)
    assert parsed.path == "/v1/messages/batches"
    assert parsed.params["limit"] == "10"
    assert parsed.params["after_id"] == "msgbatch_cursor"


def test_transform_list_batches_response_maps_batches_and_has_more(config):
    raw = _response(
        {
            "data": [
                {"id": "msgbatch_1", "processing_status": "in_progress", "request_counts": {}},
                {"id": "msgbatch_2", "processing_status": "ended", "request_counts": {"succeeded": 1}},
            ],
            "has_more": True,
            "first_id": "msgbatch_1",
            "last_id": "msgbatch_2",
        },
        method="GET",
    )
    result = config.transform_list_batches_response(
        model=None, raw_response=raw, logging_obj=MagicMock(), litellm_params={}
    )
    assert result["object"] == "list"
    assert result["has_more"] is True
    assert result["first_id"] == "msgbatch_1"
    assert result["last_id"] == "msgbatch_2"
    assert [b.id for b in result["data"]] == ["msgbatch_1", "msgbatch_2"]
    assert result["data"][1].status == "completed"


# =========================================================================== #
# cancel batch
# =========================================================================== #


def test_get_cancel_batch_url_ends_with_cancel(config):
    url = config.get_cancel_batch_url(
        api_base="https://api.anthropic.com",
        api_key="sk-ant-test",
        model="",
        batch_id="msgbatch_abc",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://api.anthropic.com/v1/messages/batches/msgbatch_abc/cancel"


def test_cancel_batch_response_canceling_maps_to_cancelling(config):
    raw = _response(
        {
            "id": "msgbatch_abc",
            "processing_status": "canceling",
            "created_at": "2024-09-24T10:00:00Z",
            "cancel_initiated_at": "2024-09-24T10:30:00Z",
            "request_counts": {},
        }
    )
    batch = config.transform_cancel_batch_response(
        model=None, raw_response=raw, logging_obj=MagicMock(), litellm_params={}
    )
    assert batch.status == "cancelling"
    assert batch.cancelling_at == 1727173800


def test_ended_after_cancel_initiated_maps_to_cancelled(config):
    raw = _response(
        {
            "id": "msgbatch_abc",
            "processing_status": "ended",
            "created_at": "2024-09-24T10:00:00Z",
            "cancel_initiated_at": "2024-09-24T10:30:00Z",
            "ended_at": "2024-09-24T10:35:00Z",
            "request_counts": {"canceled": 1, "succeeded": 0},
        }
    )
    batch = config.transform_retrieve_batch_response(
        model=None, raw_response=raw, logging_obj=MagicMock(), litellm_params={}
    )
    assert batch.status == "cancelled"
    assert batch.cancelled_at == 1727174100
    assert batch.cancelling_at == 1727173800
    assert batch.completed_at is None


def test_ended_without_cancel_maps_to_completed(config):
    raw = _response(
        {
            "id": "msgbatch_abc",
            "processing_status": "ended",
            "created_at": "2024-09-24T10:00:00Z",
            "ended_at": "2024-09-24T10:35:00Z",
            "request_counts": {"succeeded": 2},
        }
    )
    batch = config.transform_retrieve_batch_response(
        model=None, raw_response=raw, logging_obj=MagicMock(), litellm_params={}
    )
    assert batch.status == "completed"
    assert batch.completed_at == 1727174100
    assert batch.cancelled_at is None


# =========================================================================== #
# transform_anthropic_batch_result_line
# =========================================================================== #


def _raw_results_response():
    return httpx.Response(
        status_code=200,
        request=httpx.Request("GET", "https://api.anthropic.com/v1/messages/batches/msgbatch_1/results"),
    )


def test_result_line_succeeded_maps_to_openai_body():
    line = {
        "custom_id": "req-1",
        "result": {
            "type": "succeeded",
            "message": {
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5",
                "content": [{"type": "text", "text": "Hello back"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 12, "output_tokens": 7},
            },
        },
    }
    out = transform_anthropic_batch_result_line(line, _raw_results_response())
    assert out["custom_id"] == "req-1"
    assert out["error"] is None
    assert out["response"]["status_code"] == 200
    assert out["response"]["request_id"] == "msg_123"
    body = out["response"]["body"]
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "Hello back"
    assert body["usage"]["prompt_tokens"] == 12
    assert body["usage"]["completion_tokens"] == 7


def test_result_line_errored_maps_status_code():
    line = {
        "custom_id": "req-2",
        "result": {
            "type": "errored",
            "error": {"type": "rate_limit_error", "message": "slow down"},
        },
    }
    out = transform_anthropic_batch_result_line(line, _raw_results_response())
    assert out["response"]["status_code"] == 429
    assert out["error"]["code"] == "rate_limit_error"
    assert out["error"]["message"] == "slow down"


def test_result_line_canceled_and_expired_have_no_response():
    for result_type, expected_message in (
        ("canceled", "The request was canceled."),
        ("expired", "The request expired before it was processed."),
    ):
        out = transform_anthropic_batch_result_line(
            {"custom_id": "req-3", "result": {"type": result_type}}, _raw_results_response()
        )
        assert out["response"] is None
        assert out["error"] == {"code": result_type, "message": expected_message}
