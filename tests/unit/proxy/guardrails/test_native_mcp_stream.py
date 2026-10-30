import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterable, Mapping
from typing import Final, cast

import pytest
from pydantic import TypeAdapter

from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.guardrails.native_mcp_stream import validate_native_mcp_stream
from litellm.proxy.utils import ProxyLogging
from litellm.types.llms.openai import OutputItemAddedEvent

_OBJECT: Final = TypeAdapter(dict[str, object])
_BYTES: Final = TypeAdapter(tuple[bytes, ...])

URL: Final = "https://mcp.example.com/github"


def caller() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        metadata={"native_mcp_policy": {"mcp_servers": [{"name": "github", "url": URL, "tools": ["search_code"]}]}}
    )


def body() -> dict[str, object]:
    return {"tools": [{"type": "mcp", "server_label": "github", "server_url": URL, "allowed_tools": ["search_code"]}]}


def call_event(name: str) -> dict[str, object]:
    return {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {"id": "call-1", "type": "mcp_call", "server_label": "github", "name": name, "arguments": ""},
    }


def sse(event: dict[str, object], newline: str = "\n") -> bytes:
    return ("data: " + json.dumps(event) + newline + newline).encode()


async def chunks(*items: object) -> AsyncGenerator[object, None]:
    for item in items:
        yield item


@pytest.mark.asyncio
@pytest.mark.parametrize("wire_format", ["dict", "pydantic", "sse", "split_sse"])
async def test_stream_rejects_call_before_releasing_its_name(wire_format: str) -> None:
    event: Final = call_event("delete_repo")
    encoded: Final = sse(event)
    versions: Final = {
        "dict": (event,),
        "pydantic": (OutputItemAddedEvent.model_validate(event),),
        "sse": (encoded,),
        "split_sse": (encoded[:27], encoded[27:]),
    }
    stream: Final = validate_native_mcp_stream(chunks(*versions[wire_format]), caller(), body())
    with pytest.raises(ProxyException, match="delete_repo"):
        await anext(stream)


@pytest.mark.asyncio
@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
async def test_stream_reassembles_sse_without_changing_allowed_payload(newline: str) -> None:
    first: Final = sse(call_event("search_code"), newline)
    arguments: Final = sse({"type": "response.mcp_call_arguments.delta", "item_id": "call-1", "delta": "{}"}, newline)
    wire: Final = first + arguments
    source: Final = chunks(wire[:3], wire[3:37], wire[37:-1], wire[-1:])
    result: Final = [item async for item in validate_native_mcp_stream(source, caller(), body())]
    assert b"".join(_BYTES.validate_python(tuple(result), strict=True)) == wire


@pytest.mark.asyncio
async def test_text_is_released_before_upstream_finishes_and_denial_closes_upstream() -> None:
    closed: Final = asyncio.Event()
    text: Final = {"type": "response.output_text.delta", "delta": "working"}

    async def upstream() -> AsyncGenerator[object, None]:
        try:
            yield text
            yield call_event("delete_repo")
            pytest.fail("blocked stream consumed later output")
        finally:
            closed.set()

    stream: Final = validate_native_mcp_stream(upstream(), caller(), body())
    assert await anext(stream) is text
    assert not closed.is_set()
    with pytest.raises(ProxyException):
        await anext(stream)
    assert closed.is_set()


@pytest.mark.asyncio
async def test_anthropic_stream_checks_server_and_tool_before_content_block() -> None:
    event: Final[dict[str, object]] = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "mcp_tool_use",
            "id": "call-1",
            "server_name": "github",
            "name": "delete_repo",
            "input": {},
        },
    }
    with pytest.raises(ProxyException, match="delete_repo"):
        await anext(validate_native_mcp_stream(chunks(sse(event)), caller(), body()))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event",
    [
        {"type": "response.mcp_call_arguments.delta", "item_id": "unknown", "delta": "{}"},
        {
            "type": "response.completed",
            "response": {"output": [{"type": "mcp_call", "server_label": "github", "name": "delete_repo"}]},
        },
    ],
)
async def test_stream_checks_referenced_calls_and_final_response(event: dict[str, object]) -> None:
    with pytest.raises(ProxyException):
        await anext(validate_native_mcp_stream(chunks(event), caller(), body()))


@pytest.mark.asyncio
async def test_unfinished_sse_frame_is_never_released() -> None:
    with pytest.raises(ProxyException, match="inside an SSE frame"):
        await anext(validate_native_mcp_stream(chunks(b'data: {"type":'), caller(), body()))


@pytest.mark.asyncio
async def test_lifecycle_events_are_held_until_the_call_identity_arrives() -> None:
    progress: Final = {"type": "response.mcp_call.in_progress", "item_id": "call-1"}
    partial: Final = {"type": "response.output_item.added", "item": {"type": "mcp_call", "id": "call-1"}}
    named: Final = call_event("search_code")
    stream: Final = validate_native_mcp_stream(chunks(progress, partial, named), caller(), body())
    assert [item async for item in stream] == [progress, partial, named]


@pytest.mark.asyncio
async def test_late_denied_identity_releases_none_of_its_earlier_events() -> None:
    progress: Final = {"type": "response.mcp_call.in_progress", "item_id": "call-1"}
    with pytest.raises(ProxyException, match="delete_repo"):
        await anext(validate_native_mcp_stream(chunks(progress, call_event("delete_repo")), caller(), body()))


@pytest.mark.asyncio
async def test_gateway_generated_events_keep_existing_handling() -> None:
    from litellm.responses.mcp.mcp_streaming_iterator import create_mcp_call_events

    events: Final = create_mcp_call_events("search_code", "call-1", "{}", result="found", base_item_id="mcp-1")
    request: Final = {"tools": [{"type": "mcp", "server_url": "litellm_proxy/mcp/github"}]}
    stream: Final = validate_native_mcp_stream(chunks(*events), caller(), request)
    assert [item async for item in stream] == events


@pytest.mark.asyncio
async def test_existing_proxy_iterator_hook_applies_native_policy_without_guardrail_registration() -> None:
    logging: Final = ProxyLogging(user_api_key_cache=UserApiKeyCache())
    stream: Final = logging.async_post_call_streaming_iterator_hook(
        response=chunks(call_event("delete_repo")), user_api_key_dict=caller(), request_data=body()
    )
    with pytest.raises(ProxyException, match="delete_repo"):
        await anext(stream)


@pytest.mark.asyncio
async def test_messages_pipeline_emits_endpoint_error_instead_of_denied_tool_event() -> None:
    from openai._streaming import SSEDecoder

    logging: Final = ProxyLogging(user_api_key_cache=UserApiKeyCache())
    event: Final[dict[str, object]] = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "mcp_tool_use",
            "id": "call-1",
            "server_name": "github",
            "name": "delete_repo",
            "input": {},
        },
    }
    stream: Final = ProxyBaseLLMRequestProcessing.async_sse_data_generator(  # pyright: ignore[reportUnknownMemberType]  # legacy endpoint declares an unparameterized request dict
        response=chunks(sse(event)), user_api_key_dict=caller(), request_data=body(), proxy_logging_obj=logging
    )
    wire: Final = "".join([item async for item in stream]).encode()
    events: Final = tuple(SSEDecoder().iter_bytes(iter((wire,))))
    assert len(events) == 1
    payload: Final = _OBJECT.validate_json(events[0].data)
    assert payload["type"] == "error"
    error: Final = _OBJECT.validate_python(payload["error"])
    assert error["type"] == "invalid_request_error"
    assert "delete_repo" in str(error["message"])


@pytest.mark.asyncio
async def test_responses_pipeline_emits_failed_response_instead_of_denied_tool() -> None:
    from openai._streaming import SSEDecoder

    from litellm.proxy.proxy_server import (
        async_data_generator,  # pyright: ignore[reportUnknownVariableType]  # legacy endpoint declares untyped response parameters
    )

    stream: Final = async_data_generator(
        response=chunks(call_event("delete_repo")),
        user_api_key_dict=caller().model_copy(update={"request_route": "/v1/responses"}),
        request_data=body(),
        responses_stream_errors=True,
    )
    wire: Final = b"".join([item.encode() if isinstance(item, str) else item async for item in stream])
    events: Final = tuple(SSEDecoder().iter_bytes(iter((wire,))))
    assert len(events) == 2
    payload: Final = _OBJECT.validate_json(events[0].data)
    assert payload["type"] == "response.failed"
    failed_response: Final = _OBJECT.validate_python(payload["response"])
    assert failed_response["output"] == []
    assert "delete_repo" in str(_OBJECT.validate_python(failed_response["error"])["message"])
    assert events[1].data == "[DONE]"


def translated_chunk(*calls: Mapping[str, object], results: tuple[Mapping[str, object], ...] = ()) -> dict[str, object]:
    return {"choices": [{"delta": {"provider_specific_fields": {"mcp_tool_uses": calls, "tool_results": results}}}]}


@pytest.mark.asyncio
async def test_translated_second_call_is_checked_before_release() -> None:
    allowed: Final = {"type": "mcp_tool_use", "id": "first", "server_name": "github", "name": "search_code"}
    forbidden: Final = {"type": "mcp_tool_use", "id": "second", "server_name": "github", "name": "delete_repo"}
    first: Final = translated_chunk(allowed)
    stream: Final = validate_native_mcp_stream(chunks(first, translated_chunk(allowed, forbidden)), caller(), body())
    assert await anext(stream) is first
    with pytest.raises(ProxyException, match="delete_repo"):
        await anext(stream)


@pytest.mark.asyncio
async def test_translated_result_can_reference_a_previously_validated_chunk() -> None:
    call: Final = {"type": "mcp_tool_use", "id": "first", "server_name": "github", "name": "search_code"}
    result: Final = {"type": "mcp_tool_result", "tool_use_id": "first", "content": "found"}
    events: Final = (translated_chunk(call), translated_chunk(results=(result,)))
    assert [event async for event in validate_native_mcp_stream(chunks(*events), caller(), body())] == list(events)


@pytest.mark.asyncio
async def test_translated_result_without_call_identity_is_never_released() -> None:
    result: Final = {"type": "mcp_tool_result", "tool_use_id": "missing", "content": "secret"}
    stream: Final = validate_native_mcp_stream(chunks(translated_chunk(results=(result,))), caller(), body())
    with pytest.raises(ProxyException, match="unvalidated tool calls"):
        await anext(stream)


@pytest.mark.asyncio
async def test_configured_attachment_context_is_used_for_streamed_response() -> None:
    from litellm.proxy.auth.native_mcp_policy import native_mcp_deployment_validator

    validate: Final = native_mcp_deployment_validator(caller())
    assert validate is not None
    source: Final = validate(body())(chunks(call_event("search_code")))
    assert isinstance(source, AsyncIterable)
    assert [event async for event in validate_native_mcp_stream(cast(AsyncIterable[object], source), caller(), {})] == [
        call_event("search_code")
    ]


@pytest.mark.asyncio
async def test_fallback_cannot_reuse_previous_attempt_call_identity() -> None:
    from litellm.proxy.auth.native_mcp_policy import native_mcp_deployment_validator

    validate: Final = native_mcp_deployment_validator(caller())
    assert validate is not None
    first_binding: Final = validate(body())
    second_binding: Final = validate(body())
    first: Final = first_binding(call_event("search_code"))
    orphan: Final = second_binding({"type": "response.mcp_call_arguments.delta", "item_id": "call-1", "delta": "{}"})
    stream: Final = validate_native_mcp_stream(chunks(first, orphan), caller(), body())
    assert await anext(stream) is first
    with pytest.raises(ProxyException, match="unvalidated tool calls"):
        await anext(stream)


@pytest.mark.asyncio
async def test_unresolved_stream_identity_has_a_bounded_pending_queue() -> None:
    pending: Final = {"type": "response.mcp_call.in_progress", "item_id": "unknown"}
    with pytest.raises(ProxyException, match="buffer limit"):
        await anext(validate_native_mcp_stream(chunks(*(pending for _ in range(1025))), caller(), body()))


@pytest.mark.asyncio
async def test_validation_precedes_buffering_callback_across_raw_stream_fallback() -> None:
    import litellm
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.proxy.auth.native_mcp_policy import native_mcp_deployment_validator

    class BufferingCallback(CustomLogger):
        async def async_post_call_streaming_iterator_hook(  # pyright: ignore[reportIncompatibleMethodOverride]  # legacy base return type only describes Chat chunks, this hook also handles native SSE
            self, user_api_key_dict: UserAPIKeyAuth, response: AsyncIterable[object], request_data: dict[str, object]
        ) -> AsyncGenerator[object, None]:
            buffered: Final = tuple([chunk async for chunk in response])
            for chunk in buffered:
                yield chunk

    identity: Final = UserAPIKeyAuth(
        metadata={
            "native_mcp_policy": {
                "mcp_servers": [
                    {"name": "github", "url": URL, "tools": ["search_code"]},
                    {"name": "github", "url": URL + "/fallback", "tools": ["get_file"]},
                ]
            }
        }
    )
    validate: Final = native_mcp_deployment_validator(identity)
    assert validate is not None
    first: Final = validate(body())
    fallback: Final = validate(
        {
            "tools": [
                {
                    "type": "mcp",
                    "server_label": "github",
                    "server_url": URL + "/fallback",
                    "allowed_tools": ["get_file"],
                }
            ]
        }
    )
    events: Final = (sse(call_event("search_code")), sse(call_event("get_file")))

    async def switching_source() -> AsyncGenerator[object, None]:
        yield events[0]
        fallback(source)
        yield events[1]

    source: Final = first(switching_source())
    assert isinstance(source, AsyncIterable)
    litellm.callbacks = [BufferingCallback()]
    logging: Final = ProxyLogging(user_api_key_cache=UserApiKeyCache())
    result: Final = [
        chunk
        async for chunk in logging.async_post_call_streaming_iterator_hook(
            response=cast(AsyncIterable[object], source), user_api_key_dict=identity, request_data={}
        )
    ]
    assert result == list(events)
