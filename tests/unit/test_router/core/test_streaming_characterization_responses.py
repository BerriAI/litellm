"""
Characterization of Responses API streaming through a router core: pre-output lifecycle events held
back while a fallback can still take over, mid-stream errors and their fallback, the continuation a
fallback receives after text was streamed, usage combined across a stream fallback, and the hidden
params the returned stream and its events carry.

Scripted streams are ``BaseResponsesAPIStreamingIterator`` subclasses
(``streaming_helpers.responses_stream``) that raise a ``MidStreamFallbackError`` the way the
completion-bridge iterator does.
"""

from collections.abc import Callable, Mapping
from typing import Final

import pytest

import litellm
from litellm.exceptions import MidStreamFallbackError
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.router_core import RouterCore
from litellm.types.llms.openai import ResponseAPIUsage, ResponseCompletedEvent, ResponsesAPIResponse
from tests.unit.test_router.core.scripted_host import (
    ScriptedHost,
    content_policy,
    context_window,
    deployment,
    rate_limit,
    server_error,
)
from tests.unit.test_router.core.streaming_helpers import (
    Event,
    ScriptedResponsesStream,
    collect_event_types,
    drain_into,
    responses_stream,
)

MakeCore = Callable[..., RouterCore]
GROUPS_AB: Final = [deployment("a", "a"), deployment("b", "b")]
CREATED: Final = "response.created"
IN_PROGRESS: Final = "response.in_progress"
TEXT_DELTA: Final = "response.output_text.delta"
COMPLETED: Final = "response.completed"


def cut(generated_content: str = "", original: Exception | None = None) -> MidStreamFallbackError:
    """The error the bridge iterator raises when its stream breaks (before output when nothing was generated)."""
    return MidStreamFallbackError(
        message="stream cut",
        model="d1",
        llm_provider="anthropic",
        original_exception=original if original is not None else rate_limit(message="stream cut"),
        generated_content=generated_content,
        is_pre_first_chunk=not generated_content,
    )


def fallback_ok(
    *events: object, opened: list[ScriptedResponsesStream] | None = None
) -> Callable[..., ScriptedResponsesStream]:
    return responses_stream(
        *(events or (Event(CREATED), Event(TEXT_DELTA), Event(COMPLETED))),
        hidden_params={"model_id": "b-stream", "additional_headers": {"x-provider-request-id": "req-b"}},
        opened=opened,
    )


def build(make_core: MakeCore, **arguments: object) -> RouterCore:  # kwargs-ok: the router's constructor arguments
    return make_core(**{"model_list": GROUPS_AB, "fallbacks": [{"a": ["b"]}], "num_retries": 0, **arguments})


async def open_stream(core: RouterCore, **request: object) -> object:  # kwargs-ok: request kwargs
    return await core.aresponses(model="a", input="hi", stream=True, **request)  # pyright: ignore[reportAttributeAccessIssue]  # endpoint method


def hidden(stream: object) -> Mapping[str, object]:
    return getattr(stream, "_hidden_params")


def completed_event(input_tokens: int, output_tokens: int) -> ResponseCompletedEvent:
    response = ResponsesAPIResponse.model_construct(
        id="resp",
        background=None,
        usage=ResponseAPIUsage(
            input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=input_tokens + output_tokens
        ),
    )
    return ResponseCompletedEvent(type=COMPLETED, response=response)  # pyright: ignore[reportArgumentType]  # the enum accepts its value


# --- the returned stream ------------------------------------------------------------------------


async def test_a_streaming_response_is_returned_as_a_responses_streaming_iterator(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", responses_stream(Event(CREATED), Event(TEXT_DELTA), Event(COMPLETED)))
    core = build(make_core)
    stream = await open_stream(core)
    assert isinstance(stream, BaseResponsesAPIStreamingIterator)
    assert await collect_event_types(stream) == [CREATED, TEXT_DELTA, COMPLETED]
    assert hidden(stream)["model_id"] == "a"


async def test_the_returned_stream_latches_the_terminal_event_it_forwarded(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    completed = Event(COMPLETED)
    host.answer("openai/a", responses_stream(Event(CREATED), Event(TEXT_DELTA), completed))
    core = build(make_core)
    stream = await open_stream(core)
    await collect_event_types(stream)
    assert getattr(stream, "completed_response") is completed


async def test_lifecycle_events_are_held_until_output_and_then_forwarded_in_order(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", responses_stream(Event(CREATED), Event(IN_PROGRESS), Event(TEXT_DELTA), Event(COMPLETED)))
    core = build(make_core)
    assert await collect_event_types(await open_stream(core)) == [CREATED, IN_PROGRESS, TEXT_DELTA, COMPLETED]


async def test_held_lifecycle_events_are_flushed_when_the_stream_ends_without_output(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", responses_stream(Event(CREATED), Event(IN_PROGRESS)))
    core = build(make_core)
    assert await collect_event_types(await open_stream(core)) == [CREATED, IN_PROGRESS]


async def test_the_source_stream_is_closed_when_the_stream_ends(make_core: MakeCore, host: ScriptedHost) -> None:
    opened: list[ScriptedResponsesStream] = []
    host.answer("openai/a", responses_stream(Event(CREATED), Event(COMPLETED), opened=opened))
    core = build(make_core)
    await collect_event_types(await open_stream(core))
    assert [stream.closed for stream in opened] == [True]


# --- mid-stream errors before output -----------------------------------------------------------


async def test_a_mid_stream_error_before_output_falls_back_and_drops_the_primarys_held_events(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", responses_stream(Event(CREATED), Event(IN_PROGRESS), error=cut()))
    host.answer("openai/b", fallback_ok())
    core = build(make_core)
    stream = await open_stream(core)
    assert await collect_event_types(stream) == [CREATED, TEXT_DELTA, COMPLETED]
    assert host.deployments_called() == ["a", "b"]
    assert host.calls[1].kwargs["input"] == "hi"
    assert host.calls[1].model_group == "b"


async def test_a_mid_stream_error_before_output_without_fallbacks_flushes_held_events_then_raises(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", responses_stream(Event(CREATED), Event(IN_PROGRESS), error=cut()))
    core = make_core(model_list=GROUPS_AB, num_retries=0)
    stream = await open_stream(core)
    seen: list[object] = []
    with pytest.raises(litellm.RateLimitError, match="stream cut") as raised:
        await drain_into(stream, seen)
    assert not isinstance(raised.value, MidStreamFallbackError)
    assert [getattr(event, "type") for event in seen] == [CREATED, IN_PROGRESS]
    assert host.deployments_called() == ["a"]


async def test_disable_fallbacks_raises_a_mid_stream_error_before_output(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", responses_stream(Event(CREATED), error=cut()))
    host.answer("openai/b", fallback_ok())
    core = build(make_core)
    stream = await open_stream(core, disable_fallbacks=True)
    with pytest.raises(litellm.RateLimitError, match="stream cut"):
        await collect_event_types(stream)
    assert host.deployments_called() == ["a"]


async def test_when_the_fallback_fails_the_held_events_are_replayed_before_the_primary_error(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", responses_stream(Event(CREATED), error=cut()))
    host.answer("openai/b", server_error())
    core = build(make_core)
    stream = await open_stream(core)
    seen: list[object] = []
    with pytest.raises(litellm.RateLimitError, match="stream cut"):
        await drain_into(stream, seen)
    assert [getattr(event, "type") for event in seen] == [CREATED]
    assert host.deployments_called() == ["a", "b"]


async def test_more_than_two_hundred_lifecycle_events_are_forwarded_before_a_fallback(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", responses_stream(*(Event(IN_PROGRESS) for _ in range(201)), error=cut()))
    host.answer("openai/b", fallback_ok())
    core = build(make_core)
    types = await collect_event_types(await open_stream(core))
    assert types == [*([IN_PROGRESS] * 201), CREATED, TEXT_DELTA, COMPLETED]


async def test_a_context_window_error_before_output_uses_the_context_window_chain(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", responses_stream(Event(CREATED), error=context_window()))
    host.answer("openai/b", fallback_ok())
    core = build(make_core, fallbacks=None, context_window_fallbacks=[{"a": ["b"]}])
    assert await collect_event_types(await open_stream(core)) == [CREATED, TEXT_DELTA, COMPLETED]
    assert host.deployments_called() == ["a", "b"]


async def test_a_context_window_error_after_output_is_raised(make_core: MakeCore, host: ScriptedHost) -> None:
    host.answer("openai/a", responses_stream(Event(CREATED), Event(TEXT_DELTA), error=context_window()))
    host.answer("openai/b", fallback_ok())
    core = build(make_core, context_window_fallbacks=[{"a": ["b"]}])
    stream = await open_stream(core)
    with pytest.raises(litellm.ContextWindowExceededError):
        await collect_event_types(stream)
    assert host.deployments_called() == ["a"]


async def test_a_content_policy_cut_before_output_uses_the_content_policy_chain(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", responses_stream(Event(CREATED), error=cut(original=content_policy())))
    host.answer("openai/b", fallback_ok())
    core = build(make_core, fallbacks=None, content_policy_fallbacks=[{"a": ["b"]}])
    assert await collect_event_types(await open_stream(core)) == [CREATED, TEXT_DELTA, COMPLETED]
    assert host.deployments_called() == ["a", "b"]


# --- continuation after output -----------------------------------------------------------------


async def test_after_text_the_fallback_receives_the_partial_output_and_a_request_to_continue(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", responses_stream(Event(CREATED), Event(TEXT_DELTA), error=cut("hel")))
    host.answer("openai/b", fallback_ok())
    core = build(make_core)
    await collect_event_types(await open_stream(core))
    sent = host.calls[1].kwargs["input"]
    assert isinstance(sent, list)
    assert sent[0] == {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
    assert sent[1] == {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hel"}]}
    assert sent[2]["role"] == "user"
    assert 'It ended with "hel"' in sent[2]["content"][0]["text"]


async def test_after_text_the_output_ceiling_is_reduced_by_the_emitted_tokens(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", responses_stream(Event(CREATED), Event(TEXT_DELTA), error=cut("hel")))
    host.answer("openai/b", fallback_ok())
    core = build(make_core)
    await collect_event_types(await open_stream(core, max_output_tokens=100))
    assert [call.kwargs["max_output_tokens"] for call in host.calls] == [100, 99]


async def test_after_text_a_spent_output_ceiling_raises_the_primary_error(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", responses_stream(Event(CREATED), Event(TEXT_DELTA), error=cut("hello there friend")))
    host.answer("openai/b", fallback_ok())
    core = build(make_core)
    stream = await open_stream(core, max_output_tokens=1)
    with pytest.raises(litellm.RateLimitError, match="stream cut"):
        await collect_event_types(stream)
    assert host.deployments_called() == ["a"]


async def test_after_text_a_content_policy_cut_raises_instead_of_continuing(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", responses_stream(Event(CREATED), Event(TEXT_DELTA), error=cut("hel", content_policy())))
    host.answer("openai/b", fallback_ok())
    core = build(make_core, content_policy_fallbacks=[{"a": ["b"]}])
    stream = await open_stream(core)
    with pytest.raises(litellm.ContentPolicyViolationError):
        await collect_event_types(stream)
    assert host.deployments_called() == ["a"]


async def test_after_a_tool_call_began_the_primary_error_is_raised(make_core: MakeCore, host: ScriptedHost) -> None:
    host.answer(
        "openai/a",
        responses_stream(Event(CREATED), Event(TEXT_DELTA), error=cut("hel"), started_tool_call=True),
    )
    host.answer("openai/b", fallback_ok())
    core = build(make_core)
    stream = await open_stream(core)
    with pytest.raises(litellm.RateLimitError, match="stream cut"):
        await collect_event_types(stream)
    assert host.deployments_called() == ["a"]


async def test_after_text_a_json_schema_output_format_raises_the_primary_error(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", responses_stream(Event(CREATED), Event(TEXT_DELTA), error=cut("hel")))
    host.answer("openai/b", fallback_ok())
    core = build(make_core)
    stream = await open_stream(core, text={"format": {"type": "json_schema", "name": "x", "schema": {}}})
    with pytest.raises(litellm.RateLimitError, match="stream cut"):
        await collect_event_types(stream)
    assert host.deployments_called() == ["a"]


async def test_after_text_the_primarys_events_stay_and_the_fallbacks_follow(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", responses_stream(Event(CREATED), Event(TEXT_DELTA), error=cut("hel")))
    host.answer("openai/b", fallback_ok())
    core = build(make_core)
    types = await collect_event_types(await open_stream(core))
    assert types[:2] == [CREATED, TEXT_DELTA]
    assert types[-1] == COMPLETED
    assert host.deployments_called() == ["a", "b"]


# --- usage across a stream fallback --------------------------------------------------------------


async def test_the_fallbacks_completed_usage_adds_the_usage_the_primary_latched(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", responses_stream(Event(CREATED), error=cut(), completed_response=completed_event(3, 4)))
    host.answer("openai/b", fallback_ok(Event(CREATED), completed_event(10, 20)))
    core = build(make_core)
    stream = await open_stream(core)
    events = [event async for event in stream]  # pyright: ignore[reportGeneralTypeIssues]  # an async stream
    usage = events[-1].response.usage
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (13, 24, 37)


async def test_without_primary_usage_the_fallbacks_completed_usage_is_unchanged(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", responses_stream(Event(CREATED), error=cut()))
    host.answer("openai/b", fallback_ok(Event(CREATED), completed_event(10, 20)))
    core = build(make_core)
    stream = await open_stream(core)
    events = [event async for event in stream]  # pyright: ignore[reportGeneralTypeIssues]  # an async stream
    usage = events[-1].response.usage
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (10, 20, 30)


# --- hidden params after a stream fallback -------------------------------------------------------


async def test_after_a_stream_fallback_the_stream_carries_the_fallbacks_hidden_params_and_headers(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", responses_stream(Event(CREATED), error=cut()))
    host.answer("openai/b", fallback_ok())
    core = build(make_core)
    stream = await open_stream(core)
    await collect_event_types(stream)
    assert hidden(stream)["model_id"] == "b-stream"
    assert hidden(stream)["additional_headers"] == {
        "x-provider-request-id": "req-b",
        "x-litellm-model-group": "b",
        "x-litellm-attempted-retries": 0,
        "x-litellm-attempted-fallbacks": 1,
    }


async def test_after_a_stream_fallback_each_fallback_event_carries_the_fallbacks_hidden_params(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", responses_stream(Event(CREATED), error=cut()))
    host.answer("openai/b", fallback_ok())
    core = build(make_core)
    stream = await open_stream(core)
    events = [event async for event in stream]  # pyright: ignore[reportGeneralTypeIssues]  # an async stream
    assert all(event._hidden_params["model_id"] == "b-stream" for event in events)
    assert all(event._hidden_params["additional_headers"]["x-litellm-model-group"] == "b" for event in events)


async def test_when_the_stream_fallback_fails_the_stream_still_reports_the_primary(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", responses_stream(Event(CREATED), error=cut()))
    host.answer("openai/b", server_error())
    core = build(make_core)
    stream = await open_stream(core)
    with pytest.raises(litellm.RateLimitError):
        await collect_event_types(stream)
    assert hidden(stream)["model_id"] == "a"
    assert hidden(stream)["additional_headers"]["x-litellm-model-group"] == "a"


async def test_the_primary_and_fallback_streams_are_closed(make_core: MakeCore, host: ScriptedHost) -> None:
    opened: list[ScriptedResponsesStream] = []
    host.answer("openai/a", responses_stream(Event(CREATED), error=cut(), opened=opened))
    host.answer("openai/b", fallback_ok(opened=opened))
    core = build(make_core)
    await collect_event_types(await open_stream(core))
    assert [stream.closed for stream in opened] == [True, True]
