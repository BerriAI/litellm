"""
Characterization of Responses API streaming through a router core: pre-output lifecycle events held
back while a fallback can still take over, mid-stream errors and their fallback, the continuation a
fallback receives after text was streamed, usage combined across a stream fallback, and the hidden
params the returned stream and its events carry.

Scripted streams are ``BaseResponsesAPIStreamingIterator`` subclasses
(``streaming_helpers.responses_stream``) that raise a ``MidStreamFallbackError`` the way the
completion-bridge iterator does.
"""

import asyncio
import gc
import weakref
from collections.abc import Callable, Mapping
from typing import Final, Literal, Protocol, TypeAlias, get_args, runtime_checkable

import pytest

import litellm
from litellm.exceptions import MidStreamFallbackError
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.router_core import RouterCore
from litellm.types.llms.openai import (
    OutputTextDeltaEvent,
    ResponseAPIUsage,
    ResponseCompletedEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)
from litellm.types.router import AllowedFailsPolicy, RetryPolicy
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
OUTPUT_ITEM_ADDED: Final = "response.output_item.added"
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


@pytest.mark.parametrize("fallback", (False, True))
async def test_overflowed_lifecycle_events_prevent_retry_or_fresh_fallback(
    make_core: MakeCore, host: ScriptedHost, fallback: bool
) -> None:
    host.answer("openai/a", responses_stream(*(Event(IN_PROGRESS) for _ in range(201)), error=cut()), fallback_ok())
    host.answer("openai/b", fallback_ok())
    core: Final = build(make_core, num_retries=0 if fallback else 2, fallbacks=[{"a": ["b"]}] if fallback else [])
    stream: Final = await open_stream(core)
    assert isinstance(stream, BaseResponsesAPIStreamingIterator)
    assert [getattr(await anext(stream), "type", None) for _ in range(201)] == [IN_PROGRESS] * 201
    with pytest.raises(litellm.RateLimitError, match="stream cut"):
        await anext(stream)
    assert host.deployments_called() == ["a"]


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


@pytest.mark.parametrize(
    "event_type",
    ("response.output_item.added", "response.reasoning_summary_text.delta", "response.content_part.added", TEXT_DELTA),
)
@pytest.mark.parametrize("fallback", (False, True))
async def test_non_text_output_prevents_a_fresh_response_on_retry_or_fallback(
    make_core: MakeCore, host: ScriptedHost, event_type: str, fallback: bool
) -> None:
    host.answer("openai/a", responses_stream(Event(CREATED), Event(event_type), error=cut()), fallback_ok())
    host.answer("openai/b", fallback_ok())
    core: Final = build(make_core, num_retries=0 if fallback else 2, fallbacks=[{"a": ["b"]}] if fallback else [])
    stream: Final = await open_stream(core)
    assert isinstance(stream, BaseResponsesAPIStreamingIterator)
    assert getattr(await anext(stream), "type", None) == CREATED
    assert getattr(await anext(stream), "type", None) == event_type
    with pytest.raises(litellm.RateLimitError, match="stream cut"):
        await anext(stream)
    assert host.deployments_called() == ["a"]


async def test_lifecycle_events_after_output_are_forwarded_before_the_stream_error(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", responses_stream(Event(CREATED), Event(OUTPUT_ITEM_ADDED), Event(IN_PROGRESS), error=cut()))
    host.answer("openai/b", fallback_ok())
    stream: Final = await open_stream(build(make_core, num_retries=2))
    assert isinstance(stream, BaseResponsesAPIStreamingIterator)
    assert getattr(await anext(stream), "type", None) == CREATED
    assert getattr(await anext(stream), "type", None) == OUTPUT_ITEM_ADDED
    assert getattr(await anext(stream), "type", None) == IN_PROGRESS
    with pytest.raises(litellm.RateLimitError, match="stream cut"):
        await anext(stream)
    assert host.deployments_called() == ["a"]


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


# --- sequence numbers across a stream fallback ---------------------------------------------------


def delta(text: str, sequence_number: int | None = None) -> OutputTextDeltaEvent:
    event: Final = OutputTextDeltaEvent(
        type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA, item_id="msg_1", output_index=0, content_index=0, delta=text
    )
    return event if sequence_number is None else event.model_copy(update={"sequence_number": sequence_number})


async def numbered(stream: object) -> list[tuple[str, object]]:  # mutable-ok: a fresh list
    return [(event.delta, event.get("sequence_number")) async for event in stream]  # pyright: ignore[reportGeneralTypeIssues, reportUnknownMemberType, reportUnknownVariableType]  # an async stream of deltas


async def test_sequence_numbers_keep_increasing_across_a_stream_fallback(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", responses_stream(delta("a", 0), delta("b", 1), error=cut("ab")))
    host.answer("openai/b", fallback_ok(delta("c", 0), delta("d", 1)))
    core = build(make_core)
    assert await numbered(await open_stream(core)) == [("a", 0), ("b", 1), ("c", 2), ("d", 3)]


async def test_a_stream_fallback_continues_above_the_highest_number_the_primary_sent(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", responses_stream(delta("a", 0), delta("b", 3), delta("c", 1), error=cut("abc")))
    host.answer("openai/b", fallback_ok(delta("d", 0)))
    core = build(make_core)
    assert await numbered(await open_stream(core)) == [("a", 0), ("b", 3), ("c", 1), ("d", 4)]


async def test_a_stream_fallback_keeps_higher_numbers_and_passes_unnumbered_events(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", responses_stream(delta("a", 0), delta("b", 1), error=cut("ab")))
    host.answer("openai/b", fallback_ok(delta("c", 7), delta("d"), delta("e", 8)))
    core = build(make_core)
    assert await numbered(await open_stream(core)) == [("a", 0), ("b", 1), ("c", 7), ("d", None), ("e", 8)]


async def test_sequence_numbers_pass_through_untouched_without_a_fallback(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", responses_stream(delta("a", 0), delta("b", 1), delta("c", 1), delta("d", 0)))
    core = build(make_core)
    assert await numbered(await open_stream(core)) == [("a", 0), ("b", 1), ("c", 1), ("d", 0)]


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


GROUP_A_TWICE: Final = [deployment("a", "a1"), deployment("a", "a2")]


def retried_ok() -> Callable[..., ScriptedResponsesStream]:
    return responses_stream(
        Event(CREATED), Event(TEXT_DELTA), Event(COMPLETED), hidden_params={"model_id": "retry-stream"}
    )


async def test_a_rate_limit_before_output_is_retried_on_the_other_deployment_of_the_group(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("*", responses_stream(Event(CREATED), Event(IN_PROGRESS), error=cut()), retried_ok())
    core = make_core(model_list=GROUP_A_TWICE, num_retries=1)
    stream = await open_stream(core)
    assert await collect_event_types(stream) == [CREATED, TEXT_DELTA, COMPLETED]
    first, retried = host.deployments_called()
    assert retried != first
    assert hidden(stream)["model_id"] == "retry-stream"
    assert hidden(stream)["additional_headers"]["x-litellm-attempted-retries"] == 1


async def test_a_rate_limit_retry_policy_retries_a_responses_stream_when_num_retries_is_zero(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("*", responses_stream(Event(CREATED), error=cut()), retried_ok())
    core = make_core(model_list=GROUP_A_TWICE, num_retries=0, retry_policy=RetryPolicy(RateLimitErrorRetries=2))
    stream = await open_stream(core)
    assert await collect_event_types(stream) == [CREATED, TEXT_DELTA, COMPLETED]
    assert len(host.deployments_called()) == 2
    assert hidden(stream)["additional_headers"]["x-litellm-max-retries"] == 2


async def test_once_the_stream_retries_are_spent_the_fallback_answers(make_core: MakeCore, host: ScriptedHost) -> None:
    host.answer("openai/a", responses_stream(Event(CREATED), error=cut()))
    host.answer("openai/b", fallback_ok())
    core = build(make_core, num_retries=1)
    stream = await open_stream(core)
    events = [event async for event in stream]  # pyright: ignore[reportGeneralTypeIssues]  # an async stream
    assert [event.type for event in events] == [CREATED, TEXT_DELTA, COMPLETED]
    called = host.deployments_called()
    assert called[-1] == "b"
    assert called.count("a") == 2
    assert hidden(stream)["model_id"] == "b-stream"
    assert all(event._hidden_params["model_id"] == "b-stream" for event in events)


async def test_after_text_the_stream_is_not_retried_in_the_group(make_core: MakeCore, host: ScriptedHost) -> None:
    host.answer("*", responses_stream(Event(CREATED), Event(TEXT_DELTA), error=cut("Hel")), retried_ok())
    core = make_core(model_list=GROUP_A_TWICE, num_retries=1)
    stream = await open_stream(core)
    seen: list[object] = []
    with pytest.raises(litellm.RateLimitError, match="stream cut"):
        await drain_into(stream, seen)
    assert [getattr(event, "type") for event in seen] == [CREATED, TEXT_DELTA]
    assert len(host.deployments_called()) == 1


async def test_after_an_output_item_reached_the_client_the_stream_is_not_retried_in_the_group(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer(
        "*", responses_stream(Event(CREATED), Event(IN_PROGRESS), Event(OUTPUT_ITEM_ADDED), error=cut()), retried_ok()
    )
    core = make_core(model_list=GROUP_A_TWICE, num_retries=1)
    stream = await open_stream(core)
    seen: list[object] = []
    with pytest.raises(litellm.RateLimitError, match="stream cut"):
        await drain_into(stream, seen)
    assert [getattr(event, "type") for event in seen] == [CREATED, IN_PROGRESS, OUTPUT_ITEM_ADDED]
    assert len(host.deployments_called()) == 1


async def test_a_second_failed_stream_counts_as_a_second_retry(make_core: MakeCore, host: ScriptedHost) -> None:
    broken: Final = responses_stream(Event(CREATED), error=cut())
    host.answer("*", broken, broken, retried_ok())
    core = make_core(
        model_list=[*GROUP_A_TWICE, deployment("a", "a3")],
        num_retries=2,
        allowed_fails_policy=AllowedFailsPolicy(RateLimitErrorAllowedFails=1000),
    )
    stream = await open_stream(core)
    assert await collect_event_types(stream) == [CREATED, TEXT_DELTA, COMPLETED]
    assert len(host.deployments_called()) == 3
    assert hidden(stream)["additional_headers"]["x-litellm-attempted-retries"] == 2


async def test_a_retried_stream_keeps_the_model_group_header(make_core: MakeCore, host: ScriptedHost) -> None:
    host.answer("*", responses_stream(Event(CREATED), error=cut()), retried_ok())
    stream = await open_stream(make_core(model_list=GROUP_A_TWICE, num_retries=1))
    served_as: Final = hidden(stream)["additional_headers"]["x-litellm-model-group"]
    assert await collect_event_types(stream) == [CREATED, TEXT_DELTA, COMPLETED]
    assert hidden(stream)["additional_headers"]["x-litellm-model-group"] == served_as


async def test_the_failed_stream_is_closed_before_the_retry_is_opened(make_core: MakeCore, host: ScriptedHost) -> None:
    opened: list[ScriptedResponsesStream] = []
    closed_when_retry_opens: list[bool] = []

    def retry(**request: object) -> ScriptedResponsesStream:  # kwargs-ok: the host passes the request kwargs
        closed_when_retry_opens.append(opened[0].closed)
        return retried_ok()(**request)

    host.answer("*", responses_stream(Event(CREATED), error=cut(), opened=opened), retry)
    core = make_core(model_list=GROUP_A_TWICE, num_retries=1)
    assert await collect_event_types(await open_stream(core)) == [CREATED, TEXT_DELTA, COMPLETED]
    assert closed_when_retry_opens == [True]


class _RequestState:
    pass


@runtime_checkable
class _Stream(Protocol):
    def __aiter__(self) -> "_Stream": ...

    async def __anext__(self) -> object: ...

    async def aclose(self) -> None: ...


Ending: TypeAlias = Literal[
    "read to the end",
    "fell back mid-stream",
    "retried in its group mid-stream",
    "closed mid-stream",
    "dropped mid-stream without closing",
]


class _FreshlyCutStream(ScriptedResponsesStream):
    """Raises a new cut: a stream holding one error would pin the request through that error's traceback."""

    async def __anext__(self) -> object:
        try:
            return await super().__anext__()
        except StopAsyncIteration:
            raise cut() from None


def _cut_after_created(**_request: object) -> ScriptedResponsesStream:  # kwargs-ok: the host passes the request kwargs
    return _FreshlyCutStream([Event(CREATED)], None, "d1", "anthropic", None)


def _answer(host: ScriptedHost, ending: Ending) -> None:
    match ending:
        case "fell back mid-stream":
            host.answer("openai/a", _cut_after_created)
            host.answer("openai/b", fallback_ok())
        case "retried in its group mid-stream":
            host.answer("openai/a", _cut_after_created, retried_ok())
        case _:
            host.answer("openai/a", responses_stream(Event(CREATED), Event(TEXT_DELTA), Event(COMPLETED)))


async def _request_after_a_stream(
    core: RouterCore, host: ScriptedHost, ending: Ending
) -> "tuple[weakref.ref[_RequestState], weakref.ref[LiteLLMLogging]]":
    state: Final = _RequestState()
    stream: Final = await open_stream(core, metadata={"state": state})
    assert isinstance(stream, BaseResponsesAPIStreamingIterator) and isinstance(stream, _Stream)
    logging_obj: Final[object] = stream.logging_obj
    assert isinstance(logging_obj, LiteLLMLogging)
    if ending in ("closed mid-stream", "dropped mid-stream without closing"):
        assert getattr(await anext(stream), "type") == CREATED
        if ending == "closed mid-stream":
            await stream.aclose()
    else:
        assert (await collect_event_types(stream))[-1] == COMPLETED
    host.calls.clear()
    host.raised.clear()
    return weakref.ref(state), weakref.ref(logging_obj)


async def _run_pending_loop_callbacks() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.mark.parametrize("ending", get_args(Ending))
async def test_a_finished_or_dropped_stream_frees_its_request_and_logging_object_without_a_gc_pass(
    make_core: MakeCore, host: ScriptedHost, ending: Ending
) -> None:
    _answer(host, ending)
    core: Final = build(make_core, num_retries=1)
    gc.collect()
    gc.disable()
    try:
        state, logging_obj = await _request_after_a_stream(core, host, ending)
        await _run_pending_loop_callbacks()
        assert state() is None, f"the {ending} stream's request stays in a reference cycle until the next gc pass"
        assert logging_obj() is None, f"the {ending} stream's logging object waits for the next gc pass"
    finally:
        gc.enable()


OUTCOME: Final = "\n\nLiteLLM: model group 'a' failed with the error above. Fallback to b also failed: "
FALLBACK_BOOM: Final = "litellm.InternalServerError: boom"


def with_router_debug(make_core: MakeCore, host: ScriptedHost) -> RouterCore:
    host.set_dynamic(expose_router_debug_in_errors=True)
    return build(make_core)


async def test_a_fallback_stream_failing_before_output_carries_the_primarys_error_and_the_outcome(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", rate_limit())
    host.answer("openai/b", responses_stream(Event(CREATED), error=cut(original=server_error())))
    core = with_router_debug(make_core, host)
    seen: list[object] = []
    with pytest.raises(litellm.InternalServerError) as raised:
        await drain_into(await open_stream(core), seen)
    assert [getattr(event, "type") for event in seen] == [CREATED]
    assert raised.value.message == f"litellm.RateLimitError: slow down{OUTCOME}{FALLBACK_BOOM}"
    assert host.deployments_called() == ["a", "b"]


async def test_a_fallback_stream_raising_a_provider_error_before_output_carries_the_outcome(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", rate_limit())
    host.answer("openai/b", responses_stream(Event(CREATED), error=server_error()))
    core = with_router_debug(make_core, host)
    with pytest.raises(litellm.InternalServerError) as raised:
        await collect_event_types(await open_stream(core))
    assert raised.value.message == f"litellm.RateLimitError: slow down{OUTCOME}{FALLBACK_BOOM}"


async def test_a_primary_cut_whose_fallback_fails_to_open_carries_the_outcome(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", responses_stream(Event(CREATED), error=cut()))
    host.answer("openai/b", server_error())
    core = with_router_debug(make_core, host)
    with pytest.raises(litellm.RateLimitError) as raised:
        await collect_event_types(await open_stream(core))
    assert raised.value.message == f"litellm.RateLimitError: stream cut{OUTCOME}{FALLBACK_BOOM}"


async def test_a_primary_cut_whose_fallback_stream_fails_before_output_carries_both_errors(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", responses_stream(Event(CREATED), error=cut()))
    host.answer("openai/b", responses_stream(Event(CREATED), error=cut(original=server_error())))
    core = with_router_debug(make_core, host)
    with pytest.raises(litellm.InternalServerError) as raised:
        await collect_event_types(await open_stream(core))
    assert raised.value.message == f"litellm.RateLimitError: stream cut{OUTCOME}{FALLBACK_BOOM}"


async def test_a_fallback_stream_failing_after_output_raises_its_own_error(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", rate_limit())
    host.answer("openai/b", responses_stream(Event(CREATED), Event(TEXT_DELTA), error=server_error()))
    core = with_router_debug(make_core, host)
    with pytest.raises(litellm.InternalServerError) as raised:
        await collect_event_types(await open_stream(core))
    assert raised.value.message == FALLBACK_BOOM


async def test_without_router_debug_a_fallback_stream_failing_before_output_raises_its_own_error(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", rate_limit())
    host.answer("openai/b", responses_stream(Event(CREATED), error=cut(original=server_error())))
    core = build(make_core)
    with pytest.raises(litellm.InternalServerError) as raised:
        await collect_event_types(await open_stream(core))
    assert raised.value.message == FALLBACK_BOOM


@pytest.mark.parametrize("output", [TEXT_DELTA, "response.function_call_arguments.delta"])
async def test_a_fallback_stream_cut_after_output_without_generated_text_raises_its_own_error(
    make_core: MakeCore, host: ScriptedHost, output: str
) -> None:
    host.answer("openai/a", rate_limit())
    host.answer("openai/b", responses_stream(Event(CREATED), Event(output), error=cut(original=server_error())))
    core = with_router_debug(make_core, host)
    with pytest.raises(litellm.InternalServerError) as raised:
        await collect_event_types(await open_stream(core))
    assert raised.value.message == FALLBACK_BOOM


async def test_a_primary_cut_after_text_whose_continuation_fallback_fails_raises_its_own_error(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", responses_stream(Event(CREATED), Event(TEXT_DELTA, delta="hi"), error=cut("hi")))
    host.answer("openai/b", server_error())
    core = with_router_debug(make_core, host)
    with pytest.raises(litellm.RateLimitError) as raised:
        await collect_event_types(await open_stream(core))
    assert raised.value.message == "litellm.RateLimitError: stream cut"
    assert host.deployments_called() == ["a", "b"]


async def test_a_primary_cut_after_output_whose_fallback_stream_fails_before_output_raises_the_fallbacks_error(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", responses_stream(Event(CREATED), Event(TEXT_DELTA), error=cut("partial", rate_limit())))
    host.answer("openai/b", responses_stream(Event(CREATED), error=server_error()))
    core = with_router_debug(make_core, host)
    with pytest.raises(litellm.InternalServerError) as raised:
        await collect_event_types(await open_stream(core))
    assert raised.value.message == FALLBACK_BOOM


async def test_a_fallback_groups_stream_retry_failing_before_output_carries_the_primarys_error(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", rate_limit())
    host.answer("openai/b", responses_stream(Event(CREATED), error=cut(original=server_error())))
    host.set_dynamic(expose_router_debug_in_errors=True)
    core = make_core(model_list=GROUPS_AB, fallbacks=[{"a": ["b"]}], num_retries=1)
    with pytest.raises(litellm.InternalServerError) as raised:
        await collect_event_types(await open_stream(core))
    assert host.deployments_called().count("b") == 2
    assert raised.value.message.startswith(f"litellm.RateLimitError: slow down{OUTCOME}")
    assert "boom" in raised.value.message.removeprefix(f"litellm.RateLimitError: slow down{OUTCOME}")


async def test_a_traceback_in_the_primarys_error_keeps_the_fallback_outcome(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", rate_limit(message="slow down\nTraceback (most recent call last):\n  File secret.py"))
    host.answer("openai/b", responses_stream(Event(CREATED), error=server_error()))
    core = with_router_debug(make_core, host)
    with pytest.raises(litellm.InternalServerError) as raised:
        await collect_event_types(await open_stream(core))
    assert raised.value.message == f"litellm.RateLimitError: slow down{OUTCOME}{FALLBACK_BOOM}"
