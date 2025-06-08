import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import time

from litellm.litellm_core_utils.litellm_logging import Logging as LitellmLogging, set_callbacks


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
