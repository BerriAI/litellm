"""
Regression tests for GitHub issue #34754.

Providers whose Responses API payloads fail strict validation used to be built with bare
``model_construct``, leaving ``event.response`` as a raw dict and ``response.usage`` unparsed. Consumers
read those as declared, so streaming requests died with ``AttributeError: 'dict' object has no attribute
'usage'`` (HTTP 500 mid-stream) or silently dropped the SpendLogs entry
"""

import datetime

import pytest

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.responses.response_construction import (
    construct_responses_api_response,
    construct_responses_api_stream_event,
)
from litellm.router import Router
from litellm.types.llms.openai import (
    ResponseAPIUsage,
    ResponseCompletedEvent,
    ResponsesAPIResponse,
)

DICT_FORMAT_COMPLETED_CHUNK = {
    "type": "response.completed",
    "response": {
        "id": "resp_34754",
        "created_at": 1700000000,
        "status": "completed",
        "usage": {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
    },
}


class _StubStreamingIterator:
    def __init__(self, completed_response: object) -> None:
        self.completed_response = completed_response


def _transform_completed_chunk() -> ResponseCompletedEvent:
    event = OpenAIResponsesAPIConfig().transform_streaming_response(
        model="gpt-5",
        parsed_chunk=DICT_FORMAT_COMPLETED_CHUNK,
        logging_obj=None,
    )
    assert isinstance(event, ResponseCompletedEvent)
    return event


def test_unvalidatable_payload_keeps_declared_shape():
    response = construct_responses_api_response(DICT_FORMAT_COMPLETED_CHUNK["response"])

    assert isinstance(response, ResponsesAPIResponse)
    assert isinstance(response.usage, ResponseAPIUsage)
    assert response.usage.input_tokens == 11
    assert response.usage.total_tokens == 18
    assert response.output == []
    assert response.id == "resp_34754"


def test_usage_total_tokens_derived_when_missing():
    response = construct_responses_api_response(
        {"id": "resp_1", "created_at": 1, "usage": {"input_tokens": 3, "output_tokens": 4}}
    )

    assert isinstance(response.usage, ResponseAPIUsage)
    assert response.usage.total_tokens == 7


@pytest.mark.parametrize(
    "usage_value",
    ["not-a-usage-block", {"input_tokens": 1, "output_tokens": 2, "input_tokens_details": "not-details"}],
)
def test_unparseable_usage_is_dropped_not_left_as_dict(usage_value):
    response = construct_responses_api_response({"id": "resp_1", "created_at": 1, "usage": usage_value})

    assert response.usage is None


@pytest.mark.parametrize(
    ("token_value", "expected"),
    [("9", 9), (True, 0), ("nine", 0), (None, 0)],
)
def test_token_counts_coerced_from_loose_provider_values(token_value, expected):
    response = construct_responses_api_response(
        {"id": "resp_1", "created_at": 1, "usage": {"input_tokens": token_value, "output_tokens": 2}}
    )

    assert isinstance(response.usage, ResponseAPIUsage)
    assert response.usage.input_tokens == expected
    assert response.usage.total_tokens == expected + 2


def test_valid_payload_is_validated_not_constructed():
    response = construct_responses_api_response(
        {
            "id": "resp_1",
            "created_at": 1,
            "output": [],
            "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
        }
    )

    assert isinstance(response.usage, ResponseAPIUsage)
    assert response.model_fields_set


def test_streaming_event_response_is_not_a_dict():
    event = _transform_completed_chunk()

    assert isinstance(event.response, ResponsesAPIResponse)
    assert isinstance(event.response.usage, ResponseAPIUsage)


def test_streaming_event_without_nested_response_is_still_constructed():
    event = construct_responses_api_stream_event(
        ResponseCompletedEvent,
        {"type": "response.completed", "response": "not-a-payload"},
    )

    assert isinstance(event, ResponseCompletedEvent)
    assert event.response == "not-a-payload"


def test_router_extracts_partial_usage_from_dict_format_response():
    """Path 1: router fallback usage extraction returned .response.usage -> HTTP 500."""
    event = _transform_completed_chunk()

    usage = Router._extract_partial_responses_usage(_StubStreamingIterator(event))

    assert isinstance(usage, ResponseAPIUsage)
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (11, 7, 18)


def test_streaming_logging_transforms_usage_for_dict_format_response():
    """Path 2: logging silently dropped the SpendLogs entry for these responses."""
    event = _transform_completed_chunk()
    logging_obj = LiteLLMLoggingObj(
        model="gpt-5",
        messages=[],
        stream=True,
        call_type="aresponses",
        start_time=datetime.datetime.now(),
        litellm_call_id="34754",
        function_id="34754",
    )

    assembled = logging_obj._get_assembled_streaming_response(
        result=event,
        start_time=datetime.datetime.now(),
        end_time=datetime.datetime.now(),
        is_async=False,
        streaming_chunks=[],
    )

    assert assembled is not None
    chat_usage = assembled.usage  # pyright: ignore[reportAttributeAccessIssue]  # set as a dict for serialization
    assert (chat_usage["prompt_tokens"], chat_usage["completion_tokens"], chat_usage["total_tokens"]) == (11, 7, 18)


@pytest.mark.parametrize("output_value", [None, "not-a-list"])
def test_output_always_iterable_for_chat_bridge(output_value):
    """Path 3: the responses -> chat bridge iterates response.output directly."""
    response = construct_responses_api_response({"id": "resp_1", "created_at": 1, "output": output_value})

    assert response.output == []
    assert list(response.output) == []
