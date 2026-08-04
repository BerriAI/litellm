"""Logging-path tests for the google_genai generate_content adapter (#30707).

The adapter path (``generate_content_provider_config is None``) routes through
``litellm.completion`` and hands the success-logging handler an already
transformed generate_content dict, with no raw ``httpx.Response`` recorded. The
handler must fall back to that dict instead of crashing the logging worker.
"""

import os
import sys
import time

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.litellm_core_utils.litellm_logging import Logging as LitellmLogging
from litellm.types.utils import ModelResponse


def _make_logging_obj() -> LitellmLogging:
    logging_obj = LitellmLogging(
        model="gemini-3.1-flash-lite",
        messages=[{"role": "user", "content": "Hello"}],
        stream=False,
        call_type="generate_content",
        start_time=time.time(),
        litellm_call_id="30707-test",
        function_id="30707-test",
    )
    # set in update_environment_variables() during a real call; the response
    # transform reads it, so provide it for the isolated handler test.
    logging_obj.optional_params = {}
    return logging_obj


def _generate_content_dict() -> dict:
    return {
        "candidates": [
            {
                "content": {"parts": [{"text": "Hello there"}], "role": "model"},
                "finishReason": "STOP",
                "index": 0,
                "safetyRatings": [],
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 2,
            "totalTokenCount": 7,
        },
        # the adapter adds a convenience top-level text field
        "text": "Hello there",
    }


def test_adapter_path_logs_without_httpx_response():
    """No httpx_response (adapter path) must not raise; the transformed dict is used."""
    logging_obj = _make_logging_obj()
    logging_obj.model_call_details["httpx_response"] = None

    result = logging_obj._handle_non_streaming_google_genai_generate_content_response_logging(
        result=_generate_content_dict()
    )

    assert isinstance(result, ModelResponse)
    assert result.choices[0].message.content == "Hello there"
    assert result.usage.total_tokens == 7


def test_httpx_response_path_still_used_when_present():
    """When a raw httpx_response exists it is still the source of truth."""
    logging_obj = _make_logging_obj()
    logging_obj.model_call_details["httpx_response"] = httpx.Response(
        status_code=200, json=_generate_content_dict()
    )

    result = logging_obj._handle_non_streaming_google_genai_generate_content_response_logging(
        result=None
    )

    assert isinstance(result, ModelResponse)
    assert result.choices[0].message.content == "Hello there"
