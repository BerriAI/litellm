"""Logging-handler coverage for the Google GenAI generate_content success path.

Ported from the original test_litellm_logging.py location; the tolerance
contract under test ships in litellm_logging.py's
_handle_non_streaming_google_genai_generate_content_response_logging.
"""

import json
import time

import httpx
import pytest

from litellm.litellm_core_utils.litellm_logging import Logging as LitellmLogging
from litellm.types.utils import ModelResponse

_GENAI_DICT = {
    "candidates": [
        {
            "content": {"parts": [{"text": "hello world"}], "role": "model"},
            "finishReason": "STOP",
            "index": 0,
            "safetyRatings": [],
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 10,
        "candidatesTokenCount": 5,
        "totalTokenCount": 15,
    },
    "text": "hello world",
}


def _generate_content_logging_obj():
    lo = LitellmLogging(
        model="gemini-1.5-flash",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="generate_content",
        start_time=time.time(),
        litellm_call_id="gc-test-1",
        function_id="gc-test-1",
    )
    lo.optional_params = {}
    return lo


def test_generate_content_logging_adapter_dict_path():
    """INV-1: adapter path returns a dict and no httpx_response -> must return ModelResponse, not raise."""
    logging_obj = _generate_content_logging_obj()

    result = logging_obj._handle_non_streaming_google_genai_generate_content_response_logging(
        result=_GENAI_DICT
    )

    assert isinstance(result, ModelResponse)
    assert result.choices[0].message.content == "hello world"  # type: ignore[union-attr]
    assert result.usage.prompt_tokens == 10  # type: ignore[attr-defined]
    assert result.usage.completion_tokens == 5  # type: ignore[attr-defined]


def test_generate_content_logging_model_response_passthrough():
    """INV-1: when result is already a ModelResponse, return it as-is."""
    logging_obj = _generate_content_logging_obj()
    model_response = ModelResponse()

    returned = logging_obj._handle_non_streaming_google_genai_generate_content_response_logging(
        result=model_response
    )

    assert returned is model_response


def test_generate_content_logging_native_httpx_path():
    """INV-2: native path with a real httpx.Response must still return a valid ModelResponse
    (same transform, same content/usage from the dict)."""
    logging_obj = _generate_content_logging_obj()
    real_httpx = httpx.Response(
        status_code=200,
        content=json.dumps(_GENAI_DICT).encode(),
        headers={"content-type": "application/json"},
    )
    logging_obj.model_call_details["httpx_response"] = real_httpx

    result = logging_obj._handle_non_streaming_google_genai_generate_content_response_logging(
        result=None
    )

    assert isinstance(result, ModelResponse)
    assert result.choices[0].message.content == "hello world"  # type: ignore[union-attr]
    assert result.usage.prompt_tokens == 10  # type: ignore[attr-defined]


def test_generate_content_logging_unusable_result_raises():
    """INV-3: genuinely unusable result (None, no httpx_response) must raise ValueError.

    The match string "no usable response to log" is specific to the fixed error message.
    The original buggy code raised 'httpx_response is None', which does NOT match, so this
    test correctly fails on the original code and passes only after the fix.
    """
    logging_obj = _generate_content_logging_obj()

    with pytest.raises(ValueError, match="no usable response to log"):
        logging_obj._handle_non_streaming_google_genai_generate_content_response_logging(
            result=None
        )
