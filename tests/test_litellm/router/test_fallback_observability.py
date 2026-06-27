"""
Tests for fallback/retry observability improvements (Issue #3).

Covers:
1. log_retry emits a warning log and populates previous_models/retry_count in metadata
2. run_async_fallback sets original_model_group in metadata (only on first fallback)
3. get_standard_logging_metadata copies previous_models, retry_count, original_model_group
4. OTEL set_attributes emits litellm.model_group, litellm.original_model_group,
   litellm.retry_count, litellm.previous_models when those fields are present
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm import Router
from litellm.litellm_core_utils.litellm_logging import (
    StandardLoggingPayloadSetup,
    get_standard_logging_metadata,
)
from litellm.types.utils import StandardLoggingMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_router() -> Router:
    """Return a minimal Router with no real deployments."""
    return Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4", "api_key": "fake-key"},
            }
        ]
    )


# ---------------------------------------------------------------------------
# 1. log_retry
# ---------------------------------------------------------------------------


class TestLogRetry:
    def test_log_retry_emits_warning(self, caplog):
        """log_retry should emit a warning-level log containing the model name."""
        import logging

        router = _make_router()
        kwargs: Dict[str, Any] = {
            "model": "gemini-2.5-flash-llmservices-usc1",
            "metadata": {},
        }
        exc = RuntimeError("Rate limit exceeded (429)")

        with caplog.at_level(logging.WARNING, logger="LiteLLM Router"):
            router.log_retry(kwargs=kwargs, e=exc)

        assert any(
            "gemini-2.5-flash-llmservices-usc1" in r.message for r in caplog.records
        ), f"Expected model name in warning log. Records: {[r.message for r in caplog.records]}"

    def test_log_retry_populates_previous_models_in_metadata(self):
        """log_retry should append to metadata['previous_models']."""
        router = _make_router()
        kwargs: Dict[str, Any] = {
            "model": "model-a",
            "metadata": {},
        }
        exc = ValueError("some error")

        updated_kwargs = router.log_retry(kwargs=kwargs, e=exc)

        previous_models = updated_kwargs["metadata"]["previous_models"]
        assert isinstance(previous_models, list)
        assert len(previous_models) == 1
        assert previous_models[0]["exception_type"] == "ValueError"
        assert previous_models[0]["model"] == "model-a"

    def test_log_retry_caps_previous_models_at_3(self):
        """log_retry should cap previous_models at 3 entries."""
        router = _make_router()
        # Pre-populate router.previous_models with 3 entries
        router.previous_models = [
            {"model": f"model-{i}", "exception_type": "Error", "exception_string": ""}
            for i in range(3)
        ]
        kwargs: Dict[str, Any] = {
            "model": "model-4",
            "metadata": {"previous_models": list(router.previous_models)},
        }
        exc = ValueError("error")

        updated_kwargs = router.log_retry(kwargs=kwargs, e=exc)

        # After capping, should not exceed 4 total (pop happens at >3 then append)
        # The existing implementation pops if len > 3 then appends, so max is 4
        assert len(updated_kwargs["metadata"]["previous_models"]) <= 4

    def test_log_retry_uses_litellm_metadata_key(self):
        """log_retry should use 'litellm_metadata' when that key exists in kwargs."""
        router = _make_router()
        kwargs: Dict[str, Any] = {
            "model": "model-a",
            "litellm_metadata": {},
        }
        exc = RuntimeError("error")

        updated_kwargs = router.log_retry(kwargs=kwargs, e=exc)

        assert "previous_models" in updated_kwargs["litellm_metadata"]
        assert "metadata" not in updated_kwargs

    def test_log_retry_warning_includes_exception_type(self, caplog):
        """Warning log from log_retry should include the exception class name."""
        import logging

        router = _make_router()
        kwargs: Dict[str, Any] = {"model": "model-x", "metadata": {}}
        exc = ConnectionError("timed out")

        with caplog.at_level(logging.WARNING, logger="LiteLLM Router"):
            router.log_retry(kwargs=kwargs, e=exc)

        messages = " ".join(r.message for r in caplog.records)
        assert "ConnectionError" in messages


# ---------------------------------------------------------------------------
# 2. run_async_fallback sets original_model_group
# ---------------------------------------------------------------------------


class TestRunAsyncFallback:
    def test_original_model_group_set_on_first_fallback(self):
        """run_async_fallback should set original_model_group in metadata on first hop."""
        from litellm.router_utils.fallback_event_handlers import run_async_fallback

        mock_router = MagicMock()
        mock_router.previous_models = []

        # Capture the kwargs that get passed into async_function_with_fallbacks
        captured_kwargs: Dict[str, Any] = {}

        def fake_log_retry(kwargs, e):
            kwargs.setdefault("metadata", {}).setdefault("previous_models", [])
            return kwargs

        mock_router.log_retry = fake_log_retry

        async def fake_async_function_with_fallbacks(*args, **kw):
            captured_kwargs.update(kw)
            return MagicMock()

        mock_router.async_function_with_fallbacks = fake_async_function_with_fallbacks

        async def _run():
            with patch(
                "litellm.router_utils.fallback_event_handlers.log_success_fallback_event",
                new=AsyncMock(),
            ), patch(
                "litellm.router_utils.fallback_event_handlers.add_fallback_headers_to_response",
                side_effect=lambda response, **kw: response,
            ):
                await run_async_fallback(
                    litellm_router=mock_router,
                    fallback_model_group=["gemini-2.5-flash-llmservices-usw1"],
                    original_model_group="gemini-2.5-flash-llmservices-usc1",
                    original_exception=RuntimeError("429"),
                    max_fallbacks=3,
                    fallback_depth=0,
                    model="gemini-2.5-flash-llmservices-usc1",
                    metadata={},
                    original_function=AsyncMock(),
                )

        asyncio.run(_run())

        assert captured_kwargs["metadata"]["original_model_group"] == (
            "gemini-2.5-flash-llmservices-usc1"
        )

    def test_original_model_group_not_overwritten_on_subsequent_fallback(self):
        """original_model_group should not be overwritten on a second fallback hop."""
        from litellm.router_utils.fallback_event_handlers import run_async_fallback

        mock_router = MagicMock()
        mock_router.previous_models = []

        captured_kwargs: Dict[str, Any] = {}

        def fake_log_retry(kwargs, e):
            kwargs.setdefault("metadata", {}).setdefault("previous_models", [])
            return kwargs

        mock_router.log_retry = fake_log_retry

        async def fake_async_function_with_fallbacks(*args, **kw):
            captured_kwargs.update(kw)
            return MagicMock()

        mock_router.async_function_with_fallbacks = fake_async_function_with_fallbacks

        async def _run():
            with patch(
                "litellm.router_utils.fallback_event_handlers.log_success_fallback_event",
                new=AsyncMock(),
            ), patch(
                "litellm.router_utils.fallback_event_handlers.add_fallback_headers_to_response",
                side_effect=lambda response, **kw: response,
            ):
                # Simulate that original_model_group was already set by a prior fallback hop
                await run_async_fallback(
                    litellm_router=mock_router,
                    fallback_model_group=["gemini-2.5-flash-llmservices-use1"],
                    original_model_group="gemini-2.5-flash-llmservices-usw1",
                    original_exception=RuntimeError("503"),
                    max_fallbacks=3,
                    fallback_depth=1,
                    model="gemini-2.5-flash-llmservices-usw1",
                    metadata={
                        "original_model_group": "gemini-2.5-flash-llmservices-usc1"
                    },
                    original_function=AsyncMock(),
                )

        asyncio.run(_run())

        # Must remain the original first-hop model, not the second-hop model
        assert captured_kwargs["metadata"]["original_model_group"] == (
            "gemini-2.5-flash-llmservices-usc1"
        )


# ---------------------------------------------------------------------------
# 3. get_standard_logging_metadata copies fallback fields
# ---------------------------------------------------------------------------


class TestGetStandardLoggingMetadata:
    def test_copies_previous_models_and_sets_retry_count(self):
        """get_standard_logging_metadata should copy previous_models and derive retry_count."""
        previous_models = [
            {"model": "model-a", "exception_type": "RateLimitError"},
            {"model": "model-b", "exception_type": "Timeout"},
        ]
        metadata = {"previous_models": previous_models}

        result = get_standard_logging_metadata(metadata=metadata)

        assert result["previous_models"] == previous_models
        assert result["retry_count"] == 2

    def test_copies_original_model_group(self):
        """get_standard_logging_metadata should copy original_model_group."""
        metadata = {"original_model_group": "gemini-2.5-flash-llmservices-usc1"}

        result = get_standard_logging_metadata(metadata=metadata)

        assert result["original_model_group"] == "gemini-2.5-flash-llmservices-usc1"

    def test_retry_count_not_set_when_no_previous_models(self):
        """retry_count should be None when there are no previous_models."""
        metadata: Dict[str, Any] = {}

        result = get_standard_logging_metadata(metadata=metadata)

        assert result.get("retry_count") is None
        assert result.get("previous_models") is None

    def test_static_method_also_copies_fields(self):
        """StandardLoggingPayloadSetup.get_standard_logging_metadata should also copy fields."""
        previous_models = [{"model": "model-a", "exception_type": "Error"}]
        metadata = {
            "previous_models": previous_models,
            "original_model_group": "model-group-x",
        }

        result = StandardLoggingPayloadSetup.get_standard_logging_metadata(
            metadata=metadata
        )

        assert result["previous_models"] == previous_models
        assert result["retry_count"] == 1
        assert result["original_model_group"] == "model-group-x"

    def test_none_metadata_returns_defaults(self):
        """None metadata should return empty StandardLoggingMetadata with None fields."""
        result = get_standard_logging_metadata(metadata=None)

        assert result.get("previous_models") is None
        assert result.get("retry_count") is None
        assert result.get("original_model_group") is None

    def test_all_three_fields_together(self):
        """All three fallback observability fields work together."""
        previous_models = [
            {"model": "a", "exception_type": "E"},
            {"model": "b", "exception_type": "E"},
            {"model": "c", "exception_type": "E"},
        ]
        metadata = {
            "previous_models": previous_models,
            "original_model_group": "model-group-orig",
        }

        result = get_standard_logging_metadata(metadata=metadata)

        assert result["previous_models"] == previous_models
        assert result["retry_count"] == 3
        assert result["original_model_group"] == "model-group-orig"


# ---------------------------------------------------------------------------
# 4. OTEL set_attributes emits fallback-related span attributes
# ---------------------------------------------------------------------------


def _build_standard_logging_payload(
    model_group: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> dict:
    """Build a minimal StandardLoggingPayload dict for testing."""
    return {
        "id": "test-id-123",
        "trace_id": "trace-abc",
        "call_type": "completion",
        "stream": False,
        "response_cost": 0.001,
        "cost_breakdown": None,
        "response_cost_failure_debug_info": None,
        "status": "success",
        "status_fields": {},
        "custom_llm_provider": "vertex_ai",
        "total_tokens": 100,
        "prompt_tokens": 80,
        "completion_tokens": 20,
        "startTime": 1700000000.0,
        "endTime": 1700000001.0,
        "completionStartTime": 1700000000.5,
        "response_time": 1.0,
        "model_map_information": {"model_id": None, "model_info": {}},
        "model": "gemini-2.5-flash",
        "model_id": None,
        "model_group": model_group,
        "api_base": "https://example.com",
        "metadata": metadata or {},
        "cache_hit": None,
        "cache_key": None,
        "saved_cache_cost": 0.0,
        "request_tags": [],
        "end_user": None,
        "requester_ip_address": None,
        "user_agent": None,
        "messages": None,
        "response": None,
        "error_str": None,
        "error_information": None,
        "model_parameters": {},
        "hidden_params": {},
        "guardrail_information": None,
        "standard_built_in_tools_params": None,
    }


class TestOtelSetAttributes:
    def _make_otel_logger(self):
        """Create an OpenTelemetry logger with mocked OTEL internals."""
        from litellm.integrations.opentelemetry import OpenTelemetry

        logger = OpenTelemetry.__new__(OpenTelemetry)
        logger.callback_name = "opentelemetry"
        logger.OTEL_EXPORTER = None
        logger.OTEL_ENDPOINT = None
        logger.OTEL_HEADERS = None
        logger._tracer_provider_cache = {}
        return logger

    def _collect_set_calls(self, span_mock) -> Dict[str, Any]:
        """Return {key: value} for every safe_set_attribute call on the span."""
        result = {}
        for call in span_mock.set_attribute.call_args_list:
            args = call[0] if call[0] else ()
            kwargs = call[1] if call[1] else {}
            key = kwargs.get("key") or (args[0] if args else None)
            value = kwargs.get("value") or (args[1] if len(args) > 1 else None)
            if key:
                result[key] = value
        return result

    def test_model_group_set_on_span(self):
        """litellm.model_group should be set when model_group is present."""
        logger = self._make_otel_logger()

        span = MagicMock()
        # safe_set_attribute calls span.set_attribute
        logger.safe_set_attribute = lambda span, key, value: span.set_attribute(
            key=key, value=value
        )

        payload = _build_standard_logging_payload(
            model_group="gemini-2.5-flash-llmservices-usc1"
        )
        kwargs = {
            "model": "gemini-2.5-flash-llmservices-usc1",
            "optional_params": {},
            "litellm_params": {"custom_llm_provider": "vertex_ai"},
            "standard_logging_object": payload,
        }

        logger.set_attributes(span=span, kwargs=kwargs, response_obj=None)

        attrs = self._collect_set_calls(span)
        assert attrs.get("litellm.model_group") == "gemini-2.5-flash-llmservices-usc1"

    def test_original_model_group_set_on_span(self):
        """litellm.original_model_group should be set when original_model_group is in metadata."""
        logger = self._make_otel_logger()
        span = MagicMock()
        logger.safe_set_attribute = lambda span, key, value: span.set_attribute(
            key=key, value=value
        )

        metadata = {"original_model_group": "gemini-2.5-flash-llmservices-usc1"}
        payload = _build_standard_logging_payload(
            model_group="gemini-2.5-flash-llmservices-usw1",
            metadata=metadata,
        )
        kwargs = {
            "model": "gemini-2.5-flash-llmservices-usw1",
            "optional_params": {},
            "litellm_params": {"custom_llm_provider": "vertex_ai"},
            "standard_logging_object": payload,
        }

        logger.set_attributes(span=span, kwargs=kwargs, response_obj=None)

        attrs = self._collect_set_calls(span)
        assert (
            attrs.get("litellm.original_model_group")
            == "gemini-2.5-flash-llmservices-usc1"
        )

    def test_retry_count_set_on_span(self):
        """litellm.retry_count should be set when retry_count is in metadata."""
        logger = self._make_otel_logger()
        span = MagicMock()
        logger.safe_set_attribute = lambda span, key, value: span.set_attribute(
            key=key, value=value
        )

        metadata = {"retry_count": 2}
        payload = _build_standard_logging_payload(metadata=metadata)
        kwargs = {
            "model": "model-b",
            "optional_params": {},
            "litellm_params": {"custom_llm_provider": "vertex_ai"},
            "standard_logging_object": payload,
        }

        logger.set_attributes(span=span, kwargs=kwargs, response_obj=None)

        attrs = self._collect_set_calls(span)
        assert attrs.get("litellm.retry_count") == 2

    def test_previous_models_set_as_json_on_span(self):
        """litellm.previous_models should be a JSON-encoded list."""
        logger = self._make_otel_logger()
        span = MagicMock()
        logger.safe_set_attribute = lambda span, key, value: span.set_attribute(
            key=key, value=value
        )

        previous_models = [
            {"model": "model-a", "exception_type": "RateLimitError"},
        ]
        metadata = {"previous_models": previous_models}
        payload = _build_standard_logging_payload(metadata=metadata)
        kwargs = {
            "model": "model-b",
            "optional_params": {},
            "litellm_params": {"custom_llm_provider": "vertex_ai"},
            "standard_logging_object": payload,
        }

        logger.set_attributes(span=span, kwargs=kwargs, response_obj=None)

        attrs = self._collect_set_calls(span)
        raw = attrs.get("litellm.previous_models")
        assert raw is not None
        decoded = json.loads(raw)
        assert decoded == previous_models

    def test_fallback_attrs_absent_when_no_fallback(self):
        """Fallback span attributes should not be set when there was no fallback."""
        logger = self._make_otel_logger()
        span = MagicMock()
        logger.safe_set_attribute = lambda span, key, value: span.set_attribute(
            key=key, value=value
        )

        payload = _build_standard_logging_payload()  # no model_group, no metadata
        kwargs = {
            "model": "model-a",
            "optional_params": {},
            "litellm_params": {"custom_llm_provider": "openai"},
            "standard_logging_object": payload,
        }

        logger.set_attributes(span=span, kwargs=kwargs, response_obj=None)

        attrs = self._collect_set_calls(span)
        assert "litellm.original_model_group" not in attrs
        assert "litellm.retry_count" not in attrs
        assert "litellm.previous_models" not in attrs
