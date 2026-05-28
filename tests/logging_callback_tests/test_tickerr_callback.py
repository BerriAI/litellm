"""
Unit tests for the Tickerr LiteLLM callback.
"""

import json
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from litellm.integrations.tickerr import TickerrLogger


# -- Config --------------------------------------------------------------------


def test_default_init():
    logger = TickerrLogger()
    assert logger.disabled is False
    assert logger.sample_rate == 0.0
    assert logger.region is None


def test_reads_env_vars():
    with patch.dict(os.environ, {"TICKERR_REGION": "us-east-1", "TICKERR_SAMPLE_RATE": "0.1"}):
        logger = TickerrLogger()
    assert logger.region == "us-east-1"
    assert logger.sample_rate == 0.1


def test_disabled_flag():
    with patch.dict(os.environ, {"TICKERR_DISABLED": "true"}):
        logger = TickerrLogger()
    assert logger.disabled is True


def test_invalid_sample_rate_does_not_crash():
    with patch.dict(os.environ, {"TICKERR_SAMPLE_RATE": "notanumber"}):
        logger = TickerrLogger()
    assert logger.sample_rate == 0.0


def test_sample_rate_clamped_to_one():
    with patch.dict(os.environ, {"TICKERR_SAMPLE_RATE": "5.0"}):
        logger = TickerrLogger()
    assert logger.sample_rate == 1.0


# -- Disabled ------------------------------------------------------------------


def test_disabled_skips_report():
    with patch.dict(os.environ, {"TICKERR_DISABLED": "1"}):
        logger = TickerrLogger()

    start = datetime(2024, 1, 1)
    end = start + timedelta(milliseconds=100)

    with patch("litellm.integrations.tickerr.threading.Thread") as mock_thread:
        logger._report({"model": "gpt-4o-mini", "exception": None}, start, end)
        mock_thread.assert_not_called()


# -- Payload -------------------------------------------------------------------


def test_failure_payload():
    logger = TickerrLogger()

    exc = MagicMock()
    exc.status_code = 429
    start = datetime(2024, 1, 1)
    end = start + timedelta(milliseconds=500)
    kwargs = {
        "model": "claude-haiku-4-5",
        "exception": exc,
        "litellm_params": {"custom_llm_provider": "anthropic"},
    }

    with patch("litellm.integrations.tickerr.urllib.request.urlopen") as mock_urlopen:
        with patch("litellm.integrations.tickerr.threading.Thread") as mock_thread:
            mock_thread.return_value.start = lambda: mock_thread.call_args[1]["target"]()
            logger._report(kwargs, start, end)

    sent = mock_urlopen.call_args[0][0]
    payload = json.loads(sent.data)
    assert payload["provider"] == "anthropic"
    assert payload["model"] == "claude-haiku-4-5"
    assert payload["status_code"] == 429
    assert payload["event_type"] == "failure"
    assert payload["latency_ms"] == 500


def test_model_passed_as_is():
    logger = TickerrLogger()
    start = datetime(2024, 1, 1)
    end = start + timedelta(milliseconds=100)

    with patch("litellm.integrations.tickerr.urllib.request.urlopen") as mock_urlopen:
        with patch("litellm.integrations.tickerr.threading.Thread") as mock_thread:
            mock_thread.return_value.start = lambda: mock_thread.call_args[1]["target"]()
            logger._report({"model": "openai/gpt-4o-mini", "exception": None}, start, end)

    payload = json.loads(mock_urlopen.call_args[0][0].data)
    assert payload["model"] == "openai/gpt-4o-mini"


def test_no_exception_omits_status_code():
    logger = TickerrLogger()
    start = datetime(2024, 1, 1)
    end = start + timedelta(milliseconds=200)

    with patch("litellm.integrations.tickerr.urllib.request.urlopen") as mock_urlopen:
        with patch("litellm.integrations.tickerr.threading.Thread") as mock_thread:
            mock_thread.return_value.start = lambda: mock_thread.call_args[1]["target"]()
            logger._report({"model": "gpt-4o", "exception": None}, start, end)

    payload = json.loads(mock_urlopen.call_args[0][0].data)
    assert "status_code" not in payload


def test_region_included():
    with patch.dict(os.environ, {"TICKERR_REGION": "eu-west-1"}):
        logger = TickerrLogger()

    start = datetime(2024, 1, 1)
    end = start + timedelta(milliseconds=100)

    with patch("litellm.integrations.tickerr.urllib.request.urlopen") as mock_urlopen:
        with patch("litellm.integrations.tickerr.threading.Thread") as mock_thread:
            mock_thread.return_value.start = lambda: mock_thread.call_args[1]["target"]()
            logger._report({"model": "gpt-4o", "exception": None}, start, end)

    payload = json.loads(mock_urlopen.call_args[0][0].data)
    assert payload["region"] == "eu-west-1"


def test_latency_from_floats():
    logger = TickerrLogger()

    with patch("litellm.integrations.tickerr.urllib.request.urlopen") as mock_urlopen:
        with patch("litellm.integrations.tickerr.threading.Thread") as mock_thread:
            mock_thread.return_value.start = lambda: mock_thread.call_args[1]["target"]()
            logger._report({"model": "gpt-4o", "exception": None}, 1000.0, 1001.5)

    payload = json.loads(mock_urlopen.call_args[0][0].data)
    assert payload["latency_ms"] == 1500


def test_provider_from_top_level_kwarg():
    logger = TickerrLogger()
    start = datetime(2024, 1, 1)
    end = start + timedelta(milliseconds=100)

    with patch("litellm.integrations.tickerr.urllib.request.urlopen") as mock_urlopen:
        with patch("litellm.integrations.tickerr.threading.Thread") as mock_thread:
            mock_thread.return_value.start = lambda: mock_thread.call_args[1]["target"]()
            logger._report({"model": "gpt-4o", "exception": None, "custom_llm_provider": "openai"}, start, end)

    payload = json.loads(mock_urlopen.call_args[0][0].data)
    assert payload["provider"] == "openai"


def test_no_provider_omits_field():
    logger = TickerrLogger()
    start = datetime(2024, 1, 1)
    end = start + timedelta(milliseconds=100)

    with patch("litellm.integrations.tickerr.urllib.request.urlopen") as mock_urlopen:
        with patch("litellm.integrations.tickerr.threading.Thread") as mock_thread:
            mock_thread.return_value.start = lambda: mock_thread.call_args[1]["target"]()
            logger._report({"model": "gpt-4o", "exception": None}, start, end)

    payload = json.loads(mock_urlopen.call_args[0][0].data)
    assert "provider" not in payload


# -- Hooks ---------------------------------------------------------------------


def test_sync_failure_delegates():
    logger = TickerrLogger()
    start = datetime(2024, 1, 1)
    end = start + timedelta(milliseconds=300)

    with patch.object(logger, "_report") as mock:
        logger.log_failure_event({"model": "gpt-4o-mini", "exception": None}, None, start, end)
        mock.assert_called_once()


@pytest.mark.asyncio
async def test_async_failure_delegates():
    logger = TickerrLogger()
    start = datetime(2024, 1, 1)
    end = start + timedelta(milliseconds=200)

    with patch.object(logger, "_report") as mock:
        await logger.async_log_failure_event({"model": "gpt-4o-mini", "exception": None}, None, start, end)
        mock.assert_called_once()


# -- Success sampling ----------------------------------------------------------


def test_success_not_reported_at_rate_zero():
    logger = TickerrLogger()
    start = datetime(2024, 1, 1)
    end = start + timedelta(milliseconds=100)

    with patch.object(logger, "_report") as mock:
        logger.log_success_event({"model": "gpt-4o", "exception": None}, None, start, end)
        mock.assert_not_called()


def test_success_reported_when_sampled():
    with patch.dict(os.environ, {"TICKERR_SAMPLE_RATE": "1.0"}):
        logger = TickerrLogger()

    start = datetime(2024, 1, 1)
    end = start + timedelta(milliseconds=200)

    with patch("litellm.integrations.tickerr.urllib.request.urlopen") as mock_urlopen:
        with patch("litellm.integrations.tickerr.threading.Thread") as mock_thread:
            mock_thread.return_value.start = lambda: mock_thread.call_args[1]["target"]()
            logger.log_success_event({"model": "gpt-4o", "exception": None, "litellm_params": {"custom_llm_provider": "openai"}}, None, start, end)

    payload = json.loads(mock_urlopen.call_args[0][0].data)
    assert payload["event_type"] == "success"
    assert payload["provider"] == "openai"


@pytest.mark.asyncio
async def test_async_success_reported_when_sampled():
    with patch.dict(os.environ, {"TICKERR_SAMPLE_RATE": "1.0"}):
        logger = TickerrLogger()

    start = datetime(2024, 1, 1)
    end = start + timedelta(milliseconds=150)

    with patch("litellm.integrations.tickerr.urllib.request.urlopen") as mock_urlopen:
        with patch("litellm.integrations.tickerr.threading.Thread") as mock_thread:
            mock_thread.return_value.start = lambda: mock_thread.call_args[1]["target"]()
            await logger.async_log_success_event({"model": "gpt-4o-mini", "exception": None, "litellm_params": {"custom_llm_provider": "openai"}}, None, start, end)

    payload = json.loads(mock_urlopen.call_args[0][0].data)
    assert payload["event_type"] == "success"


# -- Network failure does not crash --------------------------------------------


def test_silent_on_network_error():
    logger = TickerrLogger()
    start = datetime(2024, 1, 1)
    end = start + timedelta(milliseconds=100)

    with patch("litellm.integrations.tickerr.urllib.request.urlopen", side_effect=OSError("refused")):
        with patch("litellm.integrations.tickerr.threading.Thread") as mock_thread:
            mock_thread.return_value.start = lambda: mock_thread.call_args[1]["target"]()
            logger._report({"model": "gpt-4o", "exception": None}, start, end)
    # No exception raised — test passes
