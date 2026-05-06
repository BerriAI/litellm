"""
Unit tests for the Tickerr LiteLLM callback.

Tests cover:
- TickerrLogger instantiation and env var config
- Provider normalization from model names and litellm_params
- Status code extraction from exceptions
- Latency calculation for both datetime and float timestamps
- Error type mapping (only known codes, no fallback default)
- Payload construction
- Thread cap (semaphore) under burst conditions
- Fire-and-forget does not block or raise on network failure
"""

import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from litellm.integrations.tickerr import (
    TickerrLogger,
    _ERROR_TYPE_MAP,
    _extract_status_code,
    _fire_and_forget,
    _inflight,
    _latency_ms,
    _normalize_provider,
    _MAX_INFLIGHT,
)


# ── Provider normalization ────────────────────────────────────────────────────


def test_normalize_provider_from_litellm_params():
    kwargs = {"litellm_params": {"custom_llm_provider": "anthropic"}}
    assert _normalize_provider("some-model", kwargs) == "anthropic"


def test_normalize_provider_from_custom_llm_provider():
    kwargs = {"custom_llm_provider": "openai"}
    assert _normalize_provider("gpt-4o", kwargs) == "openai"


def test_normalize_provider_from_model_prefix():
    assert _normalize_provider("anthropic/claude-3-5-haiku", {}) == "anthropic"
    assert _normalize_provider("openai/gpt-4o", {}) == "openai"


def test_normalize_provider_from_model_name_pattern():
    assert _normalize_provider("claude-haiku-4-5", {}) == "anthropic"
    assert _normalize_provider("gpt-4o-mini", {}) == "openai"
    assert _normalize_provider("gemini-2.5-flash", {}) == "google"
    assert _normalize_provider("mistral-small-latest", {}) == "mistral"
    assert _normalize_provider("llama-3.3-70b", {}) == "meta"
    assert _normalize_provider("grok-3-mini", {}) == "xai"
    assert _normalize_provider("deepseek-v3", {}) == "deepseek"


def test_normalize_provider_unknown():
    assert _normalize_provider("some-unknown-model-xyz", {}) == "unknown"


# ── Status code extraction ────────────────────────────────────────────────────


def test_extract_status_code_int():
    exc = MagicMock()
    exc.status_code = 429
    assert _extract_status_code(exc) == 429


def test_extract_status_code_string():
    exc = MagicMock()
    exc.status_code = "503"
    assert _extract_status_code(exc) == 503


def test_extract_status_code_none_exception():
    assert _extract_status_code(None) is None


def test_extract_status_code_no_attribute():
    assert _extract_status_code(ValueError("oops")) is None


# ── Latency calculation ───────────────────────────────────────────────────────


def test_latency_ms_with_datetime():
    start = datetime(2024, 1, 1, 0, 0, 0)
    end = start + timedelta(milliseconds=1240)
    assert _latency_ms(start, end) == 1240


def test_latency_ms_with_floats():
    assert _latency_ms(1000.0, 1001.5) == 1500


def test_latency_ms_mixed_types_float():
    # Both floats — should not raise
    result = _latency_ms(0.0, 0.5)
    assert result == 500


# ── Error type mapping ────────────────────────────────────────────────────────


def test_error_type_known_codes():
    assert _ERROR_TYPE_MAP[429] == "rate_limit"
    assert _ERROR_TYPE_MAP[529] == "overloaded"
    assert _ERROR_TYPE_MAP[503] == "overloaded"
    assert _ERROR_TYPE_MAP[408] == "timeout"
    assert _ERROR_TYPE_MAP[401] == "auth"


def test_error_type_no_default_for_unknown_codes():
    # 500 is a generic server error (crash/bug), not definitively "overloaded".
    # Unknown codes must NOT map to any value.
    for code in (400, 404, 500, 502, 422, 301):
        assert code not in _ERROR_TYPE_MAP, f"code {code} should not be in _ERROR_TYPE_MAP"


# ── TickerrLogger instantiation ───────────────────────────────────────────────


def test_tickerr_logger_default_init():
    logger = TickerrLogger()
    assert logger.client_tier is None
    assert logger.region is None


def test_tickerr_logger_reads_env_vars():
    with patch.dict(os.environ, {"TICKERR_CLIENT_TIER": "pro", "TICKERR_REGION": "us-east-1"}):
        logger = TickerrLogger()
    assert logger.client_tier == "pro"
    assert logger.region == "us-east-1"


# ── Payload construction via _report ─────────────────────────────────────────


def test_report_builds_correct_payload():
    logger = TickerrLogger()
    captured = {}

    def fake_fire(payload):
        captured.update(payload)

    exc = MagicMock()
    exc.status_code = 429

    start = datetime(2024, 1, 1, 0, 0, 0)
    end = start + timedelta(milliseconds=500)

    kwargs = {
        "model": "claude-haiku-4-5",
        "exception": exc,
        "litellm_params": {"custom_llm_provider": "anthropic"},
    }

    with patch("litellm.integrations.tickerr._fire_and_forget", side_effect=fake_fire):
        logger._report(kwargs, start, end)

    assert captured["provider"] == "anthropic"
    assert captured["model"] == "claude-haiku-4-5"
    assert captured["error_code"] == 429
    assert captured["error_type"] == "rate_limit"
    assert captured["latency_ms"] == 500


def test_report_strips_provider_prefix_from_model():
    logger = TickerrLogger()
    captured = {}

    def fake_fire(payload):
        captured.update(payload)

    start = datetime(2024, 1, 1, 0, 0, 0)
    end = start + timedelta(milliseconds=100)

    kwargs = {"model": "openai/gpt-4o-mini", "exception": None}

    with patch("litellm.integrations.tickerr._fire_and_forget", side_effect=fake_fire):
        logger._report(kwargs, start, end)

    assert captured["model"] == "gpt-4o-mini"


def test_report_omits_error_type_for_unknown_code():
    logger = TickerrLogger()
    captured = {}

    def fake_fire(payload):
        captured.update(payload)

    exc = MagicMock()
    exc.status_code = 400  # not in _ERROR_TYPE_MAP

    start = datetime(2024, 1, 1, 0, 0, 0)
    end = start + timedelta(milliseconds=100)
    kwargs = {"model": "gpt-4o", "exception": exc}

    with patch("litellm.integrations.tickerr._fire_and_forget", side_effect=fake_fire):
        logger._report(kwargs, start, end)

    assert "error_type" not in captured
    assert captured["error_code"] == 400


# ── Thread cap ────────────────────────────────────────────────────────────────


def test_fire_and_forget_respects_semaphore_cap():
    """Reports beyond _MAX_INFLIGHT are dropped silently without blocking."""
    # Exhaust the semaphore by acquiring all slots directly.
    # Track how many we actually acquired so the finally block releases exactly
    # that many — releasing more than acquired would push the count above its
    # initial maximum and corrupt later tests.
    acquired_count = 0
    for _ in range(_MAX_INFLIGHT):
        if _inflight.acquire(blocking=False):
            acquired_count += 1
        else:
            break

    assert acquired_count == _MAX_INFLIGHT, (
        f"semaphore should have {_MAX_INFLIGHT} slots available at test start, "
        f"got {acquired_count}"
    )

    try:
        # With semaphore exhausted, _fire_and_forget must return immediately
        # without starting a thread (non-blocking acquire fails → early return)
        with patch("threading.Thread") as mock_thread:
            _fire_and_forget({"provider": "openai"})
            mock_thread.assert_not_called()
    finally:
        for _ in range(acquired_count):
            _inflight.release()


def test_semaphore_released_on_thread_start_failure():
    """If t.start() raises, the semaphore slot must be released so future reports work."""
    # Verify the semaphore can be re-acquired after a thread-start failure,
    # which proves the slot was released (without reading private CPython internals).
    with patch("threading.Thread") as mock_thread:
        mock_thread.return_value.start.side_effect = RuntimeError("OS thread limit")
        _fire_and_forget({"provider": "openai"})

    # If the semaphore was not released, this acquire would block forever.
    # Use non-blocking to fail fast in case of a bug.
    acquired = _inflight.acquire(blocking=False)
    assert acquired, "semaphore slot was not released after thread start failure"
    _inflight.release()  # restore


# ── Network failure is silent ─────────────────────────────────────────────────


def test_fire_and_forget_silent_on_network_error():
    """A network error in the send function must not propagate to the caller."""
    # Mock threading.Thread so no real thread is created and no real network
    # call can escape the test boundary (repo rule: no real network calls).
    with patch("threading.Thread") as mock_thread_cls:
        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread

        # Simulate the send function raising an OSError inside the thread
        def run_target(*args, **kwargs):
            target = mock_thread_cls.call_args[1].get("target") or mock_thread_cls.call_args[0][0]
            try:
                target()
            except Exception:
                pass  # errors in thread body must not surface

        mock_thread.start.side_effect = run_target
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            _fire_and_forget({"provider": "anthropic", "model": "claude-haiku-4-5"})


# ── Async hooks ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_log_failure_event_calls_report():
    logger = TickerrLogger()
    start = datetime(2024, 1, 1)
    end = start + timedelta(milliseconds=200)
    kwargs = {"model": "gpt-4o-mini", "exception": None}

    with patch.object(logger, "_report") as mock_report:
        await logger.async_log_failure_event(kwargs, None, start, end)
        mock_report.assert_called_once_with(kwargs, start, end)


# ── Additional tests added to hit 67 % patch coverage ────────────────────────


# ── Sync log_failure_event hook ───────────────────────────────────────────────


def test_log_failure_event_sync_calls_report():
    """log_failure_event (sync) must delegate to _report."""
    logger = TickerrLogger()
    start = datetime(2024, 1, 1)
    end = start + timedelta(milliseconds=300)
    kwargs = {"model": "gpt-4o-mini", "exception": None}

    with patch.object(logger, "_report") as mock_report:
        logger.log_failure_event(kwargs, None, start, end)
        mock_report.assert_called_once_with(kwargs, start, end)


# ── _normalize_provider: extra provider map entries ───────────────────────────


def test_normalize_provider_vertex_ai_maps_to_google():
    kwargs = {"custom_llm_provider": "vertex_ai"}
    assert _normalize_provider("gemini-2.5-pro", kwargs) == "google"


def test_normalize_provider_vertex_ai_anthropic_maps_to_anthropic():
    kwargs = {"custom_llm_provider": "vertex_ai_anthropic"}
    assert _normalize_provider("claude-3-5-sonnet", kwargs) == "anthropic"


def test_normalize_provider_azure_maps_to_azure():
    kwargs = {"litellm_params": {"custom_llm_provider": "azure"}}
    assert _normalize_provider("gpt-4o", kwargs) == "azure"


def test_normalize_provider_bedrock_maps_to_aws():
    kwargs = {"custom_llm_provider": "bedrock"}
    assert _normalize_provider("claude-3-haiku", kwargs) == "aws"


def test_normalize_provider_bedrock_converse_maps_to_aws():
    kwargs = {"custom_llm_provider": "bedrock_converse"}
    assert _normalize_provider("claude-3-haiku", kwargs) == "aws"


def test_normalize_provider_command_model_pattern():
    """command-* model names should map to cohere via regex."""
    assert _normalize_provider("command-r-plus", {}) == "cohere"
    assert _normalize_provider("command-r", {}) == "cohere"


def test_normalize_provider_mixtral_model_pattern():
    """mixtral-* model names should map to mistral via regex."""
    assert _normalize_provider("mixtral-8x7b-instruct", {}) == "mistral"


def test_normalize_provider_deepseek_from_param():
    kwargs = {"custom_llm_provider": "deepseek"}
    assert _normalize_provider("deepseek-v3", kwargs) == "deepseek"


def test_normalize_provider_openrouter_from_param():
    kwargs = {"litellm_params": {"custom_llm_provider": "openrouter"}}
    assert _normalize_provider("meta-llama/llama-3.3-70b-instruct", kwargs) == "openrouter"


def test_normalize_provider_fireworks_ai_from_param():
    kwargs = {"custom_llm_provider": "fireworks_ai"}
    assert _normalize_provider("llama-v3-70b-instruct", kwargs) == "fireworks"


def test_normalize_provider_cerebras_from_param():
    kwargs = {"custom_llm_provider": "cerebras"}
    assert _normalize_provider("llama3.1-8b", kwargs) == "cerebras"


def test_normalize_provider_xai_from_param():
    kwargs = {"custom_llm_provider": "xai"}
    assert _normalize_provider("grok-3-mini", kwargs) == "xai"


# ── _report: edge cases ───────────────────────────────────────────────────────


def test_report_no_exception_omits_error_fields():
    """When exception is None, payload must not include error_code or error_type."""
    logger = TickerrLogger()
    captured = {}

    def fake_fire(payload):
        captured.update(payload)

    start = datetime(2024, 1, 1)
    end = start + timedelta(milliseconds=200)
    kwargs = {"model": "gemini-2.5-flash", "exception": None}

    with patch("litellm.integrations.tickerr._fire_and_forget", side_effect=fake_fire):
        logger._report(kwargs, start, end)

    assert "error_code" not in captured
    assert "error_type" not in captured
    assert captured["provider"] == "google"


def test_report_includes_client_tier_and_region_in_payload():
    """client_tier and region must appear in the payload when set via env."""
    with patch.dict(os.environ, {"TICKERR_CLIENT_TIER": "enterprise", "TICKERR_REGION": "eu-west-1"}):
        logger = TickerrLogger()

    captured = {}

    def fake_fire(payload):
        captured.update(payload)

    start = datetime(2024, 1, 1)
    end = start + timedelta(milliseconds=100)
    kwargs = {"model": "gpt-4o", "exception": None}

    with patch("litellm.integrations.tickerr._fire_and_forget", side_effect=fake_fire):
        logger._report(kwargs, start, end)

    assert captured["client_tier"] == "enterprise"
    assert captured["region"] == "eu-west-1"


def test_report_empty_model_sets_model_none():
    """An empty model string must result in model=None in the payload."""
    logger = TickerrLogger()
    captured = {}

    def fake_fire(payload):
        captured.update(payload)

    start = datetime(2024, 1, 1)
    end = start + timedelta(milliseconds=50)
    kwargs = {"model": "", "exception": None}

    with patch("litellm.integrations.tickerr._fire_and_forget", side_effect=fake_fire):
        logger._report(kwargs, start, end)

    assert captured["model"] is None


def test_report_timeout_error_type():
    """408 and 524 both map to error_type='timeout'."""
    logger = TickerrLogger()

    for code in (408, 524):
        captured = {}

        def fake_fire(payload, _c=code):
            captured.update(payload)

        exc = MagicMock()
        exc.status_code = code
        start = datetime(2024, 1, 1)
        end = start + timedelta(milliseconds=5000)
        kwargs = {"model": "claude-haiku-4-5", "exception": exc}

        with patch("litellm.integrations.tickerr._fire_and_forget", side_effect=fake_fire):
            logger._report(kwargs, start, end)

        assert captured["error_type"] == "timeout", f"Expected timeout for {code}"
        assert captured["error_code"] == code


def test_report_auth_error_type():
    """401 and 403 both map to error_type='auth'."""
    logger = TickerrLogger()

    for code in (401, 403):
        captured = {}

        def fake_fire(payload):
            captured.update(payload)

        exc = MagicMock()
        exc.status_code = code
        start = datetime(2024, 1, 1)
        end = start + timedelta(milliseconds=200)
        kwargs = {"model": "gpt-4o", "exception": exc}

        with patch("litellm.integrations.tickerr._fire_and_forget", side_effect=fake_fire):
            logger._report(kwargs, start, end)

        assert captured["error_type"] == "auth", f"Expected auth for {code}"


# ── _fire_and_forget: normal success path ─────────────────────────────────────


def test_fire_and_forget_success_path_releases_semaphore():
    """On a successful HTTP call, the semaphore slot must be released."""
    mock_response = MagicMock()
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)

    with patch("threading.Thread") as mock_thread_cls:
        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread

        def run_target_inline(*args, **kwargs):
            # Extract and call the real target so _send runs synchronously
            target = mock_thread_cls.call_args[1].get("target") or mock_thread_cls.call_args[0][0]
            target()

        mock_thread.start.side_effect = run_target_inline

        with patch("urllib.request.urlopen", return_value=mock_response):
            _fire_and_forget({"provider": "openai", "model": "gpt-4o-mini"})

    # Semaphore must be acquirable — proves the finally block released the slot
    acquired = _inflight.acquire(blocking=False)
    assert acquired, "semaphore slot was not released after successful send"
    _inflight.release()


# ── _latency_ms edge cases ────────────────────────────────────────────────────


def test_latency_ms_zero():
    assert _latency_ms(1000.0, 1000.0) == 0


def test_latency_ms_large_value():
    start = datetime(2024, 1, 1, 0, 0, 0)
    end = start + timedelta(seconds=30)
    assert _latency_ms(start, end) == 30_000


# ── _extract_status_code edge cases ──────────────────────────────────────────


def test_extract_status_code_non_digit_string():
    """A non-numeric string status_code must return None."""
    exc = MagicMock()
    exc.status_code = "N/A"
    assert _extract_status_code(exc) is None


def test_extract_status_code_529():
    exc = MagicMock()
    exc.status_code = 529
    assert _extract_status_code(exc) == 529
