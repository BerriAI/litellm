import os
import sys
import time
import pytest

# Ensure repo root is on the path so tests can import the package during pytest runs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.types.llms.openai import ErrorEvent
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


def test_transform_streaming_response_coalesces_null_error_code():
    """Ensure that when a streaming error event contains error.code=None,
    transform_streaming_response coalesces it to 'unknown_error' and returns
    an ErrorEvent instance without raising a ValidationError.
    """
    config = OpenAIResponsesAPIConfig()

    parsed_chunk = {
        "type": "error",
        "sequence_number": 1,
        "error": {
            "type": "invalid_request_error",
            "code": None,
            "message": "Something went wrong",
            "param": None,
        },
    }

    # Provide a minimal Logging object since the transformer type-hints it
    logging_obj = LiteLLMLoggingObj(
        model="gpt-4o",
        messages=[],
        stream=False,
        call_type="responses",
        start_time=time.time(),
        litellm_call_id="test-call-id",
        function_id="test-function-id",
    )

    # Should not raise
    event = config.transform_streaming_response(
        model="gpt-4o", parsed_chunk=parsed_chunk, logging_obj=logging_obj
    )

    # Validate returned type and coalesced code
    assert isinstance(event, ErrorEvent)
    assert event.error.code == "unknown_error"
    assert event.error.message == "Something went wrong"
