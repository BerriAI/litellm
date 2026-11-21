import json
from types import MappingProxyType
from typing import Final

from litellm.proxy.common_utils.responses_stream_errors import (
    ResponsesStreamErrorState,
    _failure_details,
    responses_error_fields,
)
from litellm.types.llms.openai import ResponseFailedEvent


class _DummyUpstreamError(Exception):
    def __init__(
        self,
        message: str = "Upstream error",
        body: object = None,
        status_code: object = 400,
        code: object = None,
        type: object = None,
    ) -> None:
        super().__init__(message)
        self.message: Final = message
        self.body: Final = body
        self.status_code: Final = status_code
        self.code: Final = code
        self.type: Final = type


def test_failure_details_extracts_upstream_direct_message():
    error: Final = _DummyUpstreamError(
        message="litellm generic message",
        status_code=400,
        body={"message": "Configuration updates are not supported with this model."},
    )

    details: Final = _failure_details(error)

    assert details.message == "Configuration updates are not supported with this model."
    assert details.status_code == 400


def test_failure_details_extracts_nested_error_mapping():
    error: Final = _DummyUpstreamError(
        message="litellm generic message",
        status_code=400,
        body={
            "error": {
                "message": "Invalid prompt parameter",
                "code": "invalid_prompt",
                "type": "invalid_request_error",
            }
        },
    )

    details: Final = _failure_details(error)

    assert details.message == "Invalid prompt parameter"
    assert details.code == "invalid_prompt"
    assert details.type == "invalid_request_error"
    assert details.status_code == 400


def test_failure_details_extracts_string_error():
    error: Final = _DummyUpstreamError(
        message="litellm generic message",
        status_code=400,
        body={"error": "Direct string error"},
    )

    details: Final = _failure_details(error)

    assert details.message == "Direct string error"
    assert details.status_code == 400


def test_failure_details_handles_malformed_upstream_body():
    error: Final = _DummyUpstreamError(
        message="Fallback error message",
        status_code=400,
        body={
            "unexpected_list": [1, 2, 3],
            "message": {"nested": "not a string"},
            "code": ["not", "an", "int_or_str"],
            "status_code": "not_an_int",
            "type": 12345,
        },
    )

    details: Final = _failure_details(error)

    assert details.message == "Fallback error message"
    assert details.status_code == 400


def test_failure_details_handles_non_mapping_body():
    error: Final = _DummyUpstreamError(
        message="Fallback error message",
        status_code=400,
        body="Plain string body",
    )

    details: Final = _failure_details(error)

    assert details.message == "Fallback error message"
    assert details.status_code == 400


def test_format_failure_emits_valid_failed_event_on_upstream_400():
    error: Final = _DummyUpstreamError(
        message="litellm generic message",
        status_code=400,
        body={"message": "Configuration updates are not supported with this model."},
    )

    state: Final = ResponsesStreamErrorState()
    frame: Final = state.format_failure(error)

    assert frame is not None
    assert frame.startswith("event: response.failed\n")

    event_payload: Final = json.loads(frame.split("data: ", 1)[1])
    failed_event: Final = ResponseFailedEvent.model_validate(event_payload)
    assert failed_event.response.status == "failed"
    assert failed_event.response.error is not None
    assert failed_event.response.error["code"] == "invalid_request_error"
    assert failed_event.response.error["message"] == "Configuration updates are not supported with this model."


def test_responses_error_fields_handles_malformed_mapping():
    malformed: Final = MappingProxyType(
        {
            "code": ["invalid"],
            "status_code": "invalid",
            "message": {"nested": "dict"},
        }
    )

    fields: Final = responses_error_fields(malformed)

    assert fields["code"] == "server_error"
    assert fields["message"] == "Unknown error"
