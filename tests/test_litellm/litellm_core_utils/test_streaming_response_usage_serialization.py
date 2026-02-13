"""
Tests for pydantic-clean serialization of ResponseAPIUsage in streaming logging.

When _get_assembled_streaming_response processes a ResponseCompletedEvent, it
transforms usage from ResponseAPIUsage → Chat Completion format → back to
ResponseAPIUsage.  The result must be a proper ResponseAPIUsage instance so that
model_dump() on ResponsesAPIResponse doesn't emit PydanticSerializationUnexpectedValue.
"""

import warnings

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.types.llms.openai import (
    ResponseAPIUsage,
    ResponseCompletedEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)


def _make_response_completed_event(
    input_tokens: int = 10,
    output_tokens: int = 20,
    total_tokens: int = 30,
) -> ResponseCompletedEvent:
    usage = ResponseAPIUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )
    response = ResponsesAPIResponse(
        id="resp_test",
        created_at=1700000000,
        output=[],
        usage=usage,
        status="completed",
    )
    return ResponseCompletedEvent(
        type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
        response=response,
    )


def test_assembled_streaming_response_usage_is_response_api_usage():
    """After _get_assembled_streaming_response, the usage field should be
    a ResponseAPIUsage instance (not a raw dict)."""
    event = _make_response_completed_event(input_tokens=5, output_tokens=15, total_tokens=20)

    logging_obj = Logging.__new__(Logging)
    result = logging_obj._get_assembled_streaming_response(
        result=event,
        start_time=None,
        end_time=None,
        is_async=False,
        streaming_chunks=[],
    )

    assert result is not None
    assert isinstance(result, ResponsesAPIResponse)
    assert isinstance(result.usage, ResponseAPIUsage)


def test_assembled_streaming_response_usage_model_dump_no_warnings():
    """model_dump() on the response should not emit any pydantic
    serialization warnings."""
    event = _make_response_completed_event()

    logging_obj = Logging.__new__(Logging)
    result = logging_obj._get_assembled_streaming_response(
        result=event,
        start_time=None,
        end_time=None,
        is_async=False,
        streaming_chunks=[],
    )

    assert result is not None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result.model_dump()

    pydantic_warnings = [
        w for w in caught if "PydanticSerializationUnexpectedValue" in str(w.message)
    ]
    assert len(pydantic_warnings) == 0, (
        f"Got unexpected pydantic warnings: {pydantic_warnings}"
    )


def test_assembled_streaming_response_preserves_token_counts():
    """The transformed usage should correctly map token counts from the
    Chat Completion format back to ResponseAPIUsage fields."""
    event = _make_response_completed_event(input_tokens=10, output_tokens=20, total_tokens=30)

    logging_obj = Logging.__new__(Logging)
    result = logging_obj._get_assembled_streaming_response(
        result=event,
        start_time=None,
        end_time=None,
        is_async=False,
        streaming_chunks=[],
    )

    assert result is not None
    usage = result.usage
    # Core ResponseAPIUsage fields should be populated with the correct values
    assert usage.input_tokens == 10
    assert usage.output_tokens == 20
    assert usage.total_tokens == 30

    # Extra fields from the Chat Completion transform (like prompt_tokens_details)
    # should be carried through as extras without causing serialization issues
    usage_dict = usage.model_dump()
    assert usage_dict["input_tokens"] == 10
    assert usage_dict["output_tokens"] == 20
    assert usage_dict["total_tokens"] == 30
