from collections import OrderedDict
from collections.abc import Mapping
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Final
from uuid import UUID

import pytest
from pydantic import ValidationError

from litellm.integrations.opik.opik_payload_builder import build_opik_payload
from litellm.integrations.opik.opik_payload_builder.types import SpanPayload, TracePayload
from litellm.types.utils import ModelResponse, Usage

_START: Final = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
_END: Final = datetime(2026, 1, 2, 3, 4, 6, tzinfo=timezone.utc)
_MESSAGES: Final = [{"role": "user", "content": "hi"}]
_RESPONSE: Final = {"id": "chatcmpl-1", "choices": []}
_HIDDEN_PARAMS: Final = {"model_id": "deployment-1"}
_LOGGING_METADATA: Final = {
    "user_api_key_alias": "team-key",
    "requester_metadata": {"opik": {"thread_id": "thread-1"}},
}
_STANDARD_LOGGING_OBJECT: Final = {
    "call_type": "acompletion",
    "status": "success",
    "model": "gpt-4o",
    "metadata": _LOGGING_METADATA,
    "messages": _MESSAGES,
    "response": _RESPONSE,
    "hidden_params": _HIDDEN_PARAMS,
    "trace_id": "not-forwarded-to-opik",
}
_RESPONSE_OBJ: Final = ModelResponse(
    model="gpt-4o",
    created=1767323045,
    usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
)
_OPIK_FIELDS_OF_THE_LOGGING_OBJECT: Final = {
    "type": "acompletion",
    "status": "success",
    "model": "gpt-4o",
    "hidden_params": _HIDDEN_PARAMS,
}
_EXISTING_TRACE: Final = {"metadata": {"opik": {"current_span_data": {"trace_id": "trace-1", "id": "span-0"}}}}


def _payloads_attached_to_trace_1(standard_logging_object: object) -> tuple[TracePayload | None, SpanPayload]:
    return build_opik_payload(
        kwargs={"standard_logging_object": standard_logging_object, "litellm_params": _EXISTING_TRACE},
        response_obj=_RESPONSE_OBJ,
        start_time=_START,
        end_time=_END,
        project_name="default-project",
    )


def _span_attached_to_trace_1(span_id: str, metadata: Mapping[str, object]) -> SpanPayload:
    return SpanPayload(
        id=span_id,
        project_name="default-project",
        trace_id="trace-1",
        parent_span_id="span-0",
        name="gpt-4o_chat.completion_1767323045",
        type="llm",
        model="gpt-4o",
        start_time="2026-01-02T03:04:05Z",
        end_time="2026-01-02T03:04:06Z",
        input=_MESSAGES,
        output=_RESPONSE,
        metadata=metadata,
        tags=[],
        usage={"completion_tokens": 2, "prompt_tokens": 1, "total_tokens": 3},
    )


def test_build_opik_payload_creates_a_trace_and_its_span_from_the_standard_logging_object() -> None:
    trace, span = build_opik_payload(
        kwargs={
            "standard_logging_object": _STANDARD_LOGGING_OBJECT,
            "custom_llm_provider": "openai",
            "response_cost": 0.25,
        },
        response_obj=_RESPONSE_OBJ,
        start_time=_START,
        end_time=_END,
        project_name="default-project",
    )
    metadata: Final = {
        "thread_id": "thread-1",
        "created_from": "litellm",
        **_LOGGING_METADATA,
        **_OPIK_FIELDS_OF_THE_LOGGING_OBJECT,
        "cost": {"total_tokens": 0.25, "currency": "USD"},
    }

    assert trace is not None
    assert (trace, span) == (
        TracePayload(
            project_name="default-project",
            id=trace.id,
            name="chat.completion",
            start_time="2026-01-02T03:04:05Z",
            end_time="2026-01-02T03:04:06Z",
            input=_MESSAGES,
            output=_RESPONSE,
            metadata=metadata,
            tags=["openai"],
            thread_id="thread-1",
        ),
        SpanPayload(
            id=span.id,
            project_name="default-project",
            trace_id=trace.id,
            name="gpt-4o_chat.completion_1767323045",
            type="llm",
            model="gpt-4o",
            start_time="2026-01-02T03:04:05Z",
            end_time="2026-01-02T03:04:06Z",
            input=_MESSAGES,
            output=_RESPONSE,
            metadata=metadata,
            tags=["openai"],
            usage={"completion_tokens": 2, "prompt_tokens": 1, "total_tokens": 3},
            provider="openai",
            total_cost=0.25,
        ),
    )
    assert [UUID(trace.id).version, UUID(span.id).version, trace.id != span.id] == [7, 7, True]
    assert [span.input is _MESSAGES, span.output is _RESPONSE, span.metadata["hidden_params"] is _HIDDEN_PARAMS] == [
        True,
        True,
        True,
    ]
    assert list(span.metadata) == [
        "thread_id",
        "created_from",
        "user_api_key_alias",
        "requester_metadata",
        "type",
        "status",
        "model",
        "hidden_params",
        "cost",
    ]


@pytest.mark.parametrize(
    "standard_logging_object",
    [
        _STANDARD_LOGGING_OBJECT,
        OrderedDict(_STANDARD_LOGGING_OBJECT),
        MappingProxyType(_STANDARD_LOGGING_OBJECT),
        {**_STANDARD_LOGGING_OBJECT, "metadata": MappingProxyType(_LOGGING_METADATA)},
    ],
)
def test_build_opik_payload_reads_any_string_keyed_mapping_as_the_standard_logging_object(
    standard_logging_object: object,
) -> None:
    trace, span = _payloads_attached_to_trace_1(standard_logging_object)

    assert (trace, span) == (
        None,
        _span_attached_to_trace_1(
            span.id,
            {
                "thread_id": "thread-1",
                "created_from": "litellm",
                **_LOGGING_METADATA,
                **_OPIK_FIELDS_OF_THE_LOGGING_OBJECT,
            },
        ),
    )


@pytest.mark.parametrize(
    "standard_logging_object",
    [
        {key: value for key, value in _STANDARD_LOGGING_OBJECT.items() if key != "metadata"},
        {**_STANDARD_LOGGING_OBJECT, "metadata": None},
        {**_STANDARD_LOGGING_OBJECT, "metadata": {}},
        {**_STANDARD_LOGGING_OBJECT, "metadata": ""},
    ],
)
def test_build_opik_payload_without_standard_logging_metadata_keeps_only_the_logging_object_fields(
    standard_logging_object: object,
) -> None:
    trace, span = _payloads_attached_to_trace_1(standard_logging_object)

    assert (trace, span) == (
        None,
        _span_attached_to_trace_1(span.id, {"created_from": "litellm", **_OPIK_FIELDS_OF_THE_LOGGING_OBJECT}),
    )


def test_build_opik_payload_of_an_empty_standard_logging_object_has_empty_input_and_output() -> None:
    _, span = _payloads_attached_to_trace_1({})

    assert (span.input, span.output, span.metadata) == ({}, {}, {"created_from": "litellm"})


@pytest.mark.parametrize(
    "standard_logging_object",
    [
        None,
        "standard_logging_object",
        ["messages", "response"],
        list(_STANDARD_LOGGING_OBJECT.items()),
        7,
        {**_STANDARD_LOGGING_OBJECT, 7: "keys must be strings"},
        {**_STANDARD_LOGGING_OBJECT, "metadata": "metadata"},
        {**_STANDARD_LOGGING_OBJECT, "metadata": ["user_api_key_alias"]},
        {**_STANDARD_LOGGING_OBJECT, "metadata": 7},
        {**_STANDARD_LOGGING_OBJECT, "metadata": {7: "keys must be strings"}},
    ],
)
def test_build_opik_payload_rejects_a_standard_logging_object_that_is_not_a_string_keyed_mapping(
    standard_logging_object: object,
) -> None:
    with pytest.raises(ValidationError) as raised:
        _payloads_attached_to_trace_1(standard_logging_object)

    assert "input_value" not in str(raised.value)


def test_build_opik_payload_without_a_standard_logging_object_raises_a_key_error() -> None:
    with pytest.raises(KeyError, match="standard_logging_object"):
        build_opik_payload(
            kwargs={}, response_obj=_RESPONSE_OBJ, start_time=_START, end_time=_END, project_name="default-project"
        )
