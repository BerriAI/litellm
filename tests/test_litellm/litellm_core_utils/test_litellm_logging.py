import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import time

from litellm.litellm_core_utils.litellm_logging import Logging as LitellmLogging
from litellm.litellm_core_utils.litellm_logging import set_callbacks


@pytest.fixture
def logging_obj():
    return LitellmLogging(
        model="bedrock/claude-3-5-sonnet-20240620-v1:0",
        messages=[{"role": "user", "content": "Hey"}],
        stream=True,
        call_type="completion",
        start_time=time.time(),
        litellm_call_id="12345",
        function_id="1245",
    )


def test_get_masked_api_base(logging_obj):
    api_base = "https://api.openai.com/v1"
    masked_api_base = logging_obj._get_masked_api_base(api_base)
    assert masked_api_base == "https://api.openai.com/v1"
    assert type(masked_api_base) == str


def test_sentry_sample_rate():
    existing_sample_rate = os.getenv("SENTRY_API_SAMPLE_RATE")
    try:
        # test with default value by removing the environment variable
        if existing_sample_rate:
            del os.environ["SENTRY_API_SAMPLE_RATE"]

        set_callbacks(["sentry"])
        # Check if the default sample rate is set to 1.0
        assert os.environ.get("SENTRY_API_SAMPLE_RATE") == "1.0"

        # test with custom value
        os.environ["SENTRY_API_SAMPLE_RATE"] = "0.5"

        set_callbacks(["sentry"])
        # Check if the custom sample rate is set correctly
        assert os.environ.get("SENTRY_API_SAMPLE_RATE") == "0.5"
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Restore the original environment variable
        if existing_sample_rate:
            os.environ["SENTRY_API_SAMPLE_RATE"] = existing_sample_rate
        else:
            if "SENTRY_API_SAMPLE_RATE" in os.environ:
                del os.environ["SENTRY_API_SAMPLE_RATE"]


def test_use_custom_pricing_for_model():
    from litellm.litellm_core_utils.litellm_logging import use_custom_pricing_for_model

    litellm_params = {
        "custom_llm_provider": "azure",
        "input_cost_per_pixel": 10,
    }
    assert use_custom_pricing_for_model(litellm_params) == True


def test_logging_prevent_double_logging(logging_obj):
    """
    When using a bridge, log only once from the underlying bridge call.
    This is to avoid double logging.
    """
    logging_obj.stream = False
    logging_obj.has_run_logging(event_type="sync_success")
    assert logging_obj.should_run_logging(event_type="sync_success") == False
    assert logging_obj.should_run_logging(event_type="sync_failure") == True
    assert logging_obj.should_run_logging(event_type="async_success") == True
    assert logging_obj.should_run_logging(event_type="async_failure") == True


@pytest.mark.asyncio
async def test_logging_result_for_bridge_calls(logging_obj):
    """
    When using a bridge, log only once from the underlying bridge call.
    This is to avoid double logging.
    """
    import asyncio

    import litellm

    with patch.object(
        litellm.litellm_core_utils.litellm_logging,
        "get_standard_logging_object_payload",
    ) as mock_should_run_logging:
        await litellm.anthropic_messages(
            max_tokens=100,
            messages=[{"role": "user", "content": "Hey"}],
            model="openai/codex-mini-latest",
            mock_response="Hello, world!",
        )
        await asyncio.sleep(1)
        assert mock_should_run_logging.call_count == 2  # called twice per call


@pytest.mark.asyncio
async def test_logging_non_streaming_request():
    import asyncio

    from litellm.integrations.custom_logger import CustomLogger

    class MockPrometheusLogger(CustomLogger):
        pass

    import litellm

    mock_logging_obj = MockPrometheusLogger()

    litellm.callbacks = [mock_logging_obj]

    with patch.object(
        mock_logging_obj,
        "async_log_success_event",
    ) as mock_async_log_success_event:
        await litellm.acompletion(
            max_tokens=100,
            messages=[{"role": "user", "content": "Hey"}],
            model="openai/codex-mini-latest",
            mock_response="Hello, world!",
        )
        await asyncio.sleep(1)
        mock_async_log_success_event.assert_called_once()
        assert mock_async_log_success_event.call_count == 1
        print(
            "mock_async_log_success_event.call_args.kwargs",
            mock_async_log_success_event.call_args.kwargs,
        )
        standard_logging_object = mock_async_log_success_event.call_args.kwargs[
            "kwargs"
        ]["standard_logging_object"]
        assert standard_logging_object["stream"] is not True
