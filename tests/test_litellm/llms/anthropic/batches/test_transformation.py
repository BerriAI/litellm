"""
Unit tests for litellm/llms/anthropic/batches/transformation.py

AnthropicBatchesConfig is the pure request/response mapping layer for Anthropic
Message Batches. It builds auth headers, constructs the batch create/retrieve
URLs, and (most importantly) maps an Anthropic MessageBatch JSON response into a
LiteLLM/OpenAI ``LiteLLMBatch`` (status mapping, timestamp parsing, request
counts). A silent bug here mis-reports batch status or counts to the caller, so
these tests assert EXACT output values rather than "ran without error".

Pure transform code runs for real. The only mocked boundaries are the credential
resolvers on AnthropicModelInfo (get_api_base / get_auth_header), which would
otherwise read process env / secret managers - mocking them keeps the URL/header
assertions deterministic without touching production transform logic.
"""

import os
import sys
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.anthropic.batches.transformation import AnthropicBatchesConfig
from litellm.types.utils import LiteLLMBatch, LlmProviders


@pytest.fixture
def config():
    return AnthropicBatchesConfig()


def _response(payload):
    """A real httpx.Response whose .json() yields ``payload``."""
    return httpx.Response(
        status_code=200,
        json=payload,
        request=httpx.Request("GET", "https://api.anthropic.com"),
    )


# =========================================================================== #
# custom_llm_provider
# =========================================================================== #


def test_custom_llm_provider_is_anthropic(config):
    assert config.custom_llm_provider == LlmProviders.ANTHROPIC


# =========================================================================== #
# validate_environment  (auth + fixed headers + beta header)
# =========================================================================== #


def test_validate_environment_builds_headers_with_api_key(config):
    headers = config.validate_environment(
        headers={},
        model="",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="sk-ant-test",
    )
    assert headers["accept"] == "application/json"
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["content-type"] == "application/json"
    # Plain api key -> x-api-key auth header.
    assert headers["x-api-key"] == "sk-ant-test"
    # Beta header is injected when not already present.
    assert headers["anthropic-beta"] == "message-batches-2024-09-24"


def test_validate_environment_preserves_existing_beta_header(config):
    headers = config.validate_environment(
        headers={"anthropic-beta": "custom-beta-value"},
        model="",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="sk-ant-test",
    )
    # Existing beta header must NOT be overwritten.
    assert headers["anthropic-beta"] == "custom-beta-value"


def test_validate_environment_oauth_key_uses_bearer(config):
    headers = config.validate_environment(
        headers={},
        model="",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="sk-ant-oat-abc123",
    )
    # OAuth tokens map to Authorization: Bearer, not x-api-key.
    assert headers["authorization"] == "Bearer sk-ant-oat-abc123"
    assert "x-api-key" not in headers


def test_validate_environment_missing_key_raises(config):
    # No api_key passed and no env credentials -> get_auth_header returns None.
    with patch.object(
        config.anthropic_model_info, "get_auth_header", return_value=None
    ):
        with pytest.raises(ValueError, match="Missing Anthropic API Key"):
            config.validate_environment(
                headers={},
                model="",
                messages=[],
                optional_params={},
                litellm_params={},
                api_key=None,
            )


# =========================================================================== #
# get_complete_batch_url  (batch creation URL)
# =========================================================================== #


def test_get_complete_batch_url_appends_path(config):
    url = config.get_complete_batch_url(
        api_base="https://api.anthropic.com",
        api_key="sk",
        model="claude-3",
        optional_params={},
        litellm_params={},
        data={},  # type: ignore[arg-type]
    )
    assert url == "https://api.anthropic.com/v1/messages/batches"


def test_get_complete_batch_url_strips_trailing_slash(config):
    url = config.get_complete_batch_url(
        api_base="https://api.anthropic.com/",
        api_key="sk",
        model="claude-3",
        optional_params={},
        litellm_params={},
        data={},  # type: ignore[arg-type]
    )
    assert url == "https://api.anthropic.com/v1/messages/batches"


def test_get_complete_batch_url_already_complete_is_unchanged(config):
    complete = "https://proxy.internal/v1/messages/batches"
    url = config.get_complete_batch_url(
        api_base=complete,
        api_key="sk",
        model="claude-3",
        optional_params={},
        litellm_params={},
        data={},  # type: ignore[arg-type]
    )
    assert url == complete


def test_get_complete_batch_url_uses_default_api_base(config):
    # api_base=None -> falls back to get_api_base() default.
    with patch.object(
        config.anthropic_model_info,
        "get_api_base",
        return_value="https://api.anthropic.com",
    ):
        url = config.get_complete_batch_url(
            api_base=None,
            api_key="sk",
            model="claude-3",
            optional_params={},
            litellm_params={},
            data={},  # type: ignore[arg-type]
        )
    assert url == "https://api.anthropic.com/v1/messages/batches"


# =========================================================================== #
# get_retrieve_batch_url  (batch retrieval URL + path encoding)
# =========================================================================== #


def test_get_retrieve_batch_url_happy_path(config):
    url = config.get_retrieve_batch_url(
        api_base="https://api.anthropic.com",
        batch_id="msgbatch_123",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://api.anthropic.com/v1/messages/batches/msgbatch_123"


def test_get_retrieve_batch_url_strips_trailing_slash(config):
    url = config.get_retrieve_batch_url(
        api_base="https://api.anthropic.com/",
        batch_id="msgbatch_123",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://api.anthropic.com/v1/messages/batches/msgbatch_123"


def test_get_retrieve_batch_url_encodes_batch_id(config):
    # batch_id is user-controlled; a path-traversal attempt must be percent-encoded.
    url = config.get_retrieve_batch_url(
        api_base="https://api.anthropic.com",
        batch_id="a/b id",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://api.anthropic.com/v1/messages/batches/a%2Fb%20id"


def test_get_retrieve_batch_url_rejects_dot_segment(config):
    with pytest.raises(ValueError, match="dot path segment"):
        config.get_retrieve_batch_url(
            api_base="https://api.anthropic.com",
            batch_id="..",
            optional_params={},
            litellm_params={},
        )


def test_get_retrieve_batch_url_uses_default_api_base(config):
    with patch.object(
        config.anthropic_model_info,
        "get_api_base",
        return_value="https://api.anthropic.com",
    ):
        url = config.get_retrieve_batch_url(
            api_base=None,
            batch_id="msgbatch_123",
            optional_params={},
            litellm_params={},
        )
    assert url == "https://api.anthropic.com/v1/messages/batches/msgbatch_123"


# =========================================================================== #
# transform_retrieve_batch_request  (no-op for Anthropic)
# =========================================================================== #


def test_transform_retrieve_batch_request_returns_empty_dict(config):
    assert (
        config.transform_retrieve_batch_request(
            batch_id="msgbatch_123", optional_params={}, litellm_params={}
        )
        == {}
    )


# =========================================================================== #
# Unimplemented create-batch methods raise NotImplementedError
# =========================================================================== #


def test_transform_create_batch_request_not_implemented(config):
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        config.transform_create_batch_request(
            model="claude-3",
            create_batch_data={},  # type: ignore[arg-type]
            optional_params={},
            litellm_params={},
        )


def test_transform_create_batch_response_not_implemented(config):
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        config.transform_create_batch_response(
            model="claude-3",
            raw_response=_response({}),
            logging_obj=MagicMock(),
            litellm_params={},
        )


# =========================================================================== #
# transform_retrieve_batch_response  (the core mapping - exact values)
# =========================================================================== #


def test_transform_retrieve_response_in_progress(config):
    raw = _response(
        {
            "id": "msgbatch_abc",
            "processing_status": "in_progress",
            "created_at": "2024-09-24T10:00:00Z",
            "expires_at": "2024-09-25T10:00:00Z",
            "request_counts": {
                "processing": 3,
                "succeeded": 2,
                "errored": 1,
                "canceled": 0,
                "expired": 0,
            },
        }
    )
    batch = config.transform_retrieve_batch_response(
        model=None, raw_response=raw, logging_obj=MagicMock(), litellm_params={}
    )

    assert isinstance(batch, LiteLLMBatch)
    assert batch.id == "msgbatch_abc"
    assert batch.object == "batch"
    assert batch.endpoint == "/v1/messages"
    assert batch.status == "in_progress"
    # output_file_id mirrors the batch id for Anthropic.
    assert batch.output_file_id == "msgbatch_abc"
    assert batch.input_file_id == "None"
    assert batch.completion_window == "24h"
    # created_at parsed from ISO8601 (UTC).
    assert batch.created_at == 1727172000
    assert batch.expires_at == 1727258400
    # in_progress -> in_progress_at is set to created_at.
    assert batch.in_progress_at == 1727172000
    assert batch.completed_at is None
    assert batch.cancelling_at is None
    assert batch.cancelled_at is None
    # request_counts: total = processing+succeeded+errored+canceled+expired.
    assert batch.request_counts.total == 6
    assert batch.request_counts.completed == 2
    assert batch.request_counts.failed == 1
    assert batch.metadata == {}


def test_transform_retrieve_response_ended_maps_to_completed(config):
    raw = _response(
        {
            "id": "msgbatch_done",
            "processing_status": "ended",
            "created_at": "2024-09-24T10:00:00Z",
            "ended_at": "2024-09-24T11:00:00Z",
            "request_counts": {"succeeded": 5, "errored": 0},
        }
    )
    batch = config.transform_retrieve_batch_response(
        model=None, raw_response=raw, logging_obj=MagicMock(), litellm_params={}
    )
    # "ended" -> OpenAI "completed".
    assert batch.status == "completed"
    # completed_at populated only because processing_status == "ended".
    assert batch.completed_at == 1727175600
    # not in_progress -> in_progress_at stays None.
    assert batch.in_progress_at is None
    assert batch.request_counts.total == 5
    assert batch.request_counts.completed == 5


def test_transform_retrieve_response_canceling_maps_to_cancelling(config):
    raw = _response(
        {
            "id": "msgbatch_cancel",
            "processing_status": "canceling",
            "created_at": "2024-09-24T10:00:00Z",
            "cancel_initiated_at": "2024-09-24T10:30:00Z",
            "ended_at": "2024-09-24T10:45:00Z",
            "request_counts": {},
        }
    )
    batch = config.transform_retrieve_batch_response(
        model=None, raw_response=raw, logging_obj=MagicMock(), litellm_params={}
    )
    # "canceling" -> OpenAI "cancelling".
    assert batch.status == "cancelling"
    assert batch.cancelling_at == 1727173800
    # cancelled_at = ended_at when canceling and ended_at present.
    assert batch.cancelled_at == 1727174700
    assert batch.completed_at is None


def test_transform_retrieve_response_unknown_status_defaults_in_progress(config):
    raw = _response(
        {
            "id": "msgbatch_x",
            "processing_status": "some_future_status",
            "request_counts": {},
        }
    )
    batch = config.transform_retrieve_batch_response(
        model=None, raw_response=raw, logging_obj=MagicMock(), litellm_params={}
    )
    # Unmapped status falls back to in_progress (don't 500 on new enum values).
    assert batch.status == "in_progress"


def test_transform_retrieve_response_missing_id_and_status_defaults(config):
    # Empty body: id defaults to "", status defaults to "in_progress".
    raw = _response({})
    before = int(time.time())
    batch = config.transform_retrieve_batch_response(
        model=None, raw_response=raw, logging_obj=MagicMock(), litellm_params={}
    )
    after = int(time.time())
    assert batch.id == ""
    assert batch.status == "in_progress"
    # No created_at -> created_at falls back to int(time.time()).
    assert before <= batch.created_at <= after
    # No created_at -> in_progress_at (which mirrors created_at) is None.
    assert batch.in_progress_at is None
    assert batch.request_counts.total == 0


def test_transform_retrieve_response_archived_sets_expired_at(config):
    raw = _response(
        {
            "id": "msgbatch_arch",
            "processing_status": "ended",
            "created_at": "2024-09-24T10:00:00Z",
            "ended_at": "2024-09-24T11:00:00Z",
            "archived_at": "2024-09-26T10:00:00Z",
            "request_counts": {},
        }
    )
    batch = config.transform_retrieve_batch_response(
        model=None, raw_response=raw, logging_obj=MagicMock(), litellm_params={}
    )
    # archived_at present -> expired_at populated.
    assert batch.expired_at == 1727344800


def test_transform_retrieve_response_bad_timestamp_is_none(config):
    raw = _response(
        {
            "id": "msgbatch_bad",
            "processing_status": "in_progress",
            "created_at": "not-a-real-timestamp",
            "request_counts": {},
        }
    )
    before = int(time.time())
    batch = config.transform_retrieve_batch_response(
        model=None, raw_response=raw, logging_obj=MagicMock(), litellm_params={}
    )
    after = int(time.time())
    # Unparseable created_at -> parse_timestamp returns None, created_at falls
    # back to time.time().
    assert before <= batch.created_at <= after


def test_transform_retrieve_response_unparseable_json_raises(config):
    bad = httpx.Response(
        status_code=200,
        content=b"not json",
        request=httpx.Request("GET", "https://api.anthropic.com"),
    )
    with pytest.raises(ValueError, match="Failed to parse Anthropic batch response"):
        config.transform_retrieve_batch_response(
            model=None, raw_response=bad, logging_obj=MagicMock(), litellm_params={}
        )


# =========================================================================== #
# get_error_class
# =========================================================================== #


def test_get_error_class_with_dict_headers(config):
    err = config.get_error_class(
        error_message="rate limited", status_code=429, headers={"x-ratelimit": "0"}
    )
    from litellm.llms.anthropic.common_utils import AnthropicError

    assert isinstance(err, AnthropicError)
    assert err.status_code == 429
    assert err.message == "rate limited"


def test_get_error_class_with_httpx_headers(config):
    hdrs = httpx.Headers({"retry-after": "5"})
    err = config.get_error_class(
        error_message="server error", status_code=500, headers=hdrs
    )
    assert err.status_code == 500
    assert err.message == "server error"


# =========================================================================== #
# transform_response  (batch results JSONL -> summed usage on ModelResponse)
# =========================================================================== #


def test_transform_response_sums_usage_across_lines(config):
    from litellm.types.utils import ModelResponse, Usage

    # Two result lines; transform_parsed_response is stubbed to attach a fixed
    # Usage per line so we can assert the SUM is what lands on model_response.
    line1 = '{"result": {"message": {"content": [{"type": "text", "text": "a"}]}}}'
    line2 = '{"result": {"message": {"content": [{"type": "text", "text": "b"}]}}}'
    raw = httpx.Response(
        status_code=200,
        text=f"{line1}\n{line2}\n",
        request=httpx.Request("GET", "https://api.anthropic.com"),
    )

    model_response = ModelResponse()

    def fake_transform_parsed(*, completion_response, raw_response, model_response):
        mr = ModelResponse()
        setattr(
            mr,
            "usage",
            Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        return mr

    with patch.object(
        config.anthropic_chat_config,
        "transform_parsed_response",
        side_effect=fake_transform_parsed,
    ):
        out = config.transform_response(
            model="claude-3",
            raw_response=raw,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
        )

    assert out is model_response
    usage = getattr(out, "usage")
    # Two lines * (10 prompt, 5 completion) summed.
    assert usage.prompt_tokens == 20
    assert usage.completion_tokens == 10
    assert usage.total_tokens == 30


def test_transform_response_skips_malformed_lines(config):
    from litellm.types.utils import ModelResponse, Usage

    valid = '{"result": {"message": {"content": [{"type": "text", "text": "a"}]}}}'
    # Interior blank line (survives the outer strip) exercises the empty-line
    # `continue`; leading not-json exercises the JSONDecodeError `continue`.
    raw = httpx.Response(
        status_code=200,
        text=f"not-json\n\n{valid}\n",
        request=httpx.Request("GET", "https://api.anthropic.com"),
    )
    model_response = ModelResponse()

    def fake_transform_parsed(*, completion_response, raw_response, model_response):
        mr = ModelResponse()
        setattr(
            mr, "usage", Usage(prompt_tokens=7, completion_tokens=3, total_tokens=10)
        )
        return mr

    with patch.object(
        config.anthropic_chat_config,
        "transform_parsed_response",
        side_effect=fake_transform_parsed,
    ):
        out = config.transform_response(
            model="claude-3",
            raw_response=raw,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
        )

    # Only the single valid line contributed usage; malformed/empty skipped.
    usage = getattr(out, "usage")
    assert usage.prompt_tokens == 7
    assert usage.completion_tokens == 3


def test_transform_response_reraises_unexpected_error(config):
    from litellm.types.utils import ModelResponse, Usage

    valid = '{"result": {"message": {"content": [{"type": "text", "text": "a"}]}}}'
    raw = httpx.Response(
        status_code=200,
        text=f"{valid}\n",
        request=httpx.Request("GET", "https://api.anthropic.com"),
    )

    def fake_transform_parsed(*, completion_response, raw_response, model_response):
        mr = ModelResponse()
        setattr(mr, "usage", Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2))
        return mr

    # A non-JSONDecodeError raised during usage aggregation must propagate
    # (the outer `except Exception: raise e`), not be swallowed.
    with patch.object(
        config.anthropic_chat_config,
        "transform_parsed_response",
        side_effect=fake_transform_parsed,
    ), patch(
        "litellm.cost_calculator.BaseTokenUsageProcessor.combine_usage_objects",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            config.transform_response(
                model="claude-3",
                raw_response=raw,
                model_response=ModelResponse(),
                logging_obj=MagicMock(),
                request_data={},
                messages=[],
                optional_params={},
                litellm_params={},
                encoding=None,
            )


# --------------------------------------------------------------------------- #
# Shared BaseBatchesConfig contract suite (consistency net across providers).
# This subclass supplies anthropic fixtures; the inherited contract tests run
# automatically. See base_batches_config_test.py.
# --------------------------------------------------------------------------- #

from litellm.types.utils import LlmProviders  # noqa: E402
from tests.test_litellm.llms.base_llm.batches.base_batches_config_test import (  # noqa: E402
    BatchesConfigContractTests,
)


class TestAnthropicBatchesContract(BatchesConfigContractTests):
    def make_config(self):
        from litellm.llms.anthropic.batches.transformation import (
            AnthropicBatchesConfig,
        )

        return AnthropicBatchesConfig()

    expected_provider = LlmProviders.ANTHROPIC
    supports_create = False  # anthropic raises NotImplementedError on create
    supports_retrieve_response = True

    def sample_retrieve_response_body(self) -> dict:
        return {
            "id": "msgbatch_123",
            "processing_status": "ended",
            "created_at": "2024-01-01T00:00:00Z",
            "ended_at": "2024-01-02T00:00:00Z",
            "request_counts": {"succeeded": 2, "errored": 1},
        }

    expected_retrieve_batch_id = "msgbatch_123"
    expected_retrieve_status = "completed"  # "ended" -> "completed"
