"""
Regression tests for https://github.com/BerriAI/litellm/issues/32019.

Streaming /v1/messages responses were silently dropped by loggers gated on
``standard_logging_object`` (e.g. ``s3_v2``) whenever StandardLoggingPayload
construction failed: ``get_standard_logging_object_payload`` swallows the
exception and returns ``None``, spend tracking still succeeds via the
``response_cost`` fallback, and no error surfaces to the caller.
"""

import logging
from datetime import datetime

import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.types.utils import ModelResponse

DEGRADED_PAYLOAD = {"trace_id": "degraded-fallback"}


@pytest.mark.asyncio
async def test_streaming_slo_build_failure_falls_back_to_degraded_payload(monkeypatch):
    """When the SLP build returns None for an assembled streaming response,
    async_success_handler must rebuild it from an empty response object so
    SLP-gated loggers still record the request."""
    logging_obj = Logging(
        model="claude-3-5-sonnet-20240620",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="anthropic_messages",
        start_time=datetime.now(),
        litellm_call_id="test-32019",
        function_id="test-32019",
    )
    build_calls = []

    def fake_build(self, init_response_obj, start_time, end_time):
        build_calls.append(init_response_obj)
        if init_response_obj == {}:
            return DEGRADED_PAYLOAD
        return None  # simulates get_standard_logging_object_payload swallowing an exception

    monkeypatch.setattr(Logging, "_build_standard_logging_payload", fake_build)
    monkeypatch.setattr(litellm, "_async_success_callback", [])
    monkeypatch.setattr(litellm, "success_callback", [])

    result = ModelResponse(model="claude-3-5-sonnet-20240620")

    await logging_obj.async_success_handler(
        result,
        start_time=datetime.now(),
        end_time=datetime.now(),
        cache_hit=False,
    )

    assert logging_obj.model_call_details["standard_logging_object"] == DEGRADED_PAYLOAD
    assert build_calls == [result, {}]


@pytest.mark.asyncio
async def test_streaming_slo_build_success_does_not_rebuild(monkeypatch):
    """The degraded rebuild must not run when the SLP builds normally."""
    logging_obj = Logging(
        model="claude-3-5-sonnet-20240620",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="anthropic_messages",
        start_time=datetime.now(),
        litellm_call_id="test-32019-ok",
        function_id="test-32019-ok",
    )
    build_calls = []
    healthy_payload = {"trace_id": "healthy"}

    def fake_build(self, init_response_obj, start_time, end_time):
        build_calls.append(init_response_obj)
        return healthy_payload

    monkeypatch.setattr(Logging, "_build_standard_logging_payload", fake_build)
    monkeypatch.setattr(litellm, "_async_success_callback", [])
    monkeypatch.setattr(litellm, "success_callback", [])

    result = ModelResponse(model="claude-3-5-sonnet-20240620")

    await logging_obj.async_success_handler(
        result,
        start_time=datetime.now(),
        end_time=datetime.now(),
        cache_hit=False,
    )

    assert logging_obj.model_call_details["standard_logging_object"] == healthy_payload
    assert len(build_calls) == 1


@pytest.mark.asyncio
async def test_streaming_slo_double_build_failure_is_logged(monkeypatch, caplog):
    """If the empty-response rebuild also fails, the persistent failure must be
    surfaced with a second error log instead of silently leaving the payload None."""
    logging_obj = Logging(
        model="claude-3-5-sonnet-20240620",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="anthropic_messages",
        start_time=datetime.now(),
        litellm_call_id="test-32019-double",
        function_id="test-32019-double",
    )
    build_calls = []

    def fake_build(self, init_response_obj, start_time, end_time):
        build_calls.append(init_response_obj)
        return None  # both the primary build and the empty-response retry fail

    monkeypatch.setattr(Logging, "_build_standard_logging_payload", fake_build)
    monkeypatch.setattr(litellm, "_async_success_callback", [])
    monkeypatch.setattr(litellm, "success_callback", [])

    result = ModelResponse(model="claude-3-5-sonnet-20240620")

    with caplog.at_level(logging.ERROR, logger="LiteLLM"):
        await logging_obj.async_success_handler(
            result,
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
        )

    assert logging_obj.model_call_details["standard_logging_object"] is None
    assert build_calls == [result, {}]
    assert "rebuild failed" in caplog.text
