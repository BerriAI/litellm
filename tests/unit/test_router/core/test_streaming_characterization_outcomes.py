"""
Characterization of what a streamed attempt records against its deployment: the success, tokens and
request a stream counts once it is fully consumed, and the failure and cooldown a stream that breaks
after its first chunk counts, for chat, Responses and /v1/messages streams.

Every scripted stream here logs through the host when it ends (``logged=True``), as litellm's stream
logging does, so a core that learns outcomes from logging callbacks sees the stream's outcome only
then, with the assembled response's usage.
"""

import asyncio
import time
from collections.abc import Callable
from typing import Final

import pytest

import litellm
from litellm.router_core import RouterCore
from litellm.types.llms.openai import ResponseAPIUsage, ResponseCompletedEvent, ResponsesAPIResponse
from tests.unit.test_router.core.scripted_host import ScriptedHost, deployment, rate_limit
from tests.unit.test_router.core.streaming_helpers import (
    MESSAGE_STOP,
    MESSAGES,
    Event,
    anthropic_stream,
    chat_stream,
    collect_bytes,
    collect_event_types,
    collect_texts,
    collect_texts_sync,
    install_orjson_stub_if_missing,
    responses_stream,
    sse,
    text_delta,
)

install_orjson_stub_if_missing()

MakeCore = Callable[..., RouterCore]
USAGE: Final = {"prompt_tokens": 4, "completion_tokens": 8, "total_tokens": 12}


def group(
    *ids: str, **params: object
) -> list[dict[str, object]]:  # kwargs-ok: litellm_params  # mutable-ok: a model_list
    return [deployment("g", deployment_id, None, **params) for deployment_id in ids]


async def open_chat(core: RouterCore, model: str = "g") -> object:
    stream: Final[object] = await core.acompletion(model=model, messages=MESSAGES, stream=True)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]  # endpoint method
    return stream  # pyright: ignore[reportUnknownVariableType]  # see above


def open_chat_sync(core: RouterCore) -> object:
    stream: Final[object] = core.completion(model="g", messages=MESSAGES, stream=True)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]  # endpoint method
    return stream  # pyright: ignore[reportUnknownVariableType]  # see above


async def open_responses(core: RouterCore) -> object:
    stream: Final[object] = await core.aresponses(model="g", input="hi", stream=True)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]  # endpoint method
    return stream  # pyright: ignore[reportUnknownVariableType]  # see above


async def open_messages(core: RouterCore) -> object:
    stream: Final[object] = await core.aanthropic_messages(model="g", messages=MESSAGES, max_tokens=5, stream=True)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]  # endpoint method
    return stream  # pyright: ignore[reportUnknownVariableType]  # see above


async def settle() -> None:
    """Wait for the logging and recording tasks a finished stream scheduled, and any tasks they schedule."""
    while pending := asyncio.all_tasks() - {asyncio.current_task()}:
        _, still_running = await asyncio.wait(pending, timeout=5)
        assert not still_running, still_running


def eventually(check: Callable[[], bool]) -> None:
    """Wait for ``check``: litellm logs a sync stream's outcome from a worker thread."""
    deadline: Final = time.monotonic() + 5.0
    while not check() and time.monotonic() < deadline:
        time.sleep(0.01)


def successes(core: RouterCore, deployment_id: str) -> object:
    count: Final[object] = core.health().store.get_cache(f"{deployment_id}:successes", local_only=True)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # DualCache's reads are untyped
    return count  # pyright: ignore[reportUnknownVariableType]  # see above


def cooled(core: RouterCore, *ids: str) -> list[str]:  # mutable-ok: a fresh list
    return sorted(dict(core.health().cooldowns.get_active_cooldowns(list(ids), None)))


# --- a stream consumed to its end ----------------------------------------------------------------


async def test_a_consumed_stream_counts_its_final_tokens_and_one_request(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream("hello", " world", usage=USAGE, logged=True))
    core: Final = make_core(model_list=group("a", rpm=100))

    stream: Final = await open_chat(core)
    assert await collect_texts(stream) == "hello world"
    await settle()

    assert await core.health().usage("g") == (12, 1)
    assert successes(core, "a") == 1


async def test_a_stream_counts_only_its_request_before_it_is_consumed(make_core: MakeCore, host: ScriptedHost) -> None:
    host.answer("openai/a", chat_stream("hello", usage=USAGE, logged=True))
    core: Final = make_core(model_list=group("a", rpm=100))

    await open_chat(core)
    await settle()

    # The request counts when the stream opens, with the tokens it has read so far (none).
    assert await core.health().usage("g") == (0, 1)
    assert successes(core, "a") is None


def test_a_sync_stream_counts_its_success_only_once_consumed(make_core: MakeCore, host: ScriptedHost) -> None:
    host.answer("openai/a", chat_stream("hello", usage=USAGE, logged=True, sync=True))
    core: Final = make_core(model_list=group("a", rpm=100))

    stream: Final = open_chat_sync(core)
    assert successes(core, "a") is None
    assert collect_texts_sync(stream) == "hello"
    eventually(lambda: successes(core, "a") is not None)

    assert successes(core, "a") == 1
    # A sync call's success callback counts the success alone: the request counted when the stream
    # opened, and no tokens.
    assert asyncio.run(core.health().usage("g")) == (0, 1)


def test_a_sync_stream_that_breaks_after_content_counts_one_failure_and_cools_down(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream("partial", error=rate_limit(message="mid stream"), logged=True, sync=True))
    core: Final = make_core(
        model_list=[deployment("g", "a", rpm=100, order=1), deployment("g", "b", order=2)],
        num_retries=0,
        allowed_fails=0,
    )

    stream: Final = open_chat_sync(core)
    with pytest.raises(litellm.RateLimitError, match="mid stream"):
        collect_texts_sync(stream)
    eventually(lambda: bool(cooled(core, "a", "b")))

    assert core.health().failures_this_minute("a") == 1
    assert successes(core, "a") is None
    assert cooled(core, "a", "b") == ["a"]
    assert asyncio.run(core.health().usage("g")) == (0, 1)


# --- a stream that breaks after its first chunk --------------------------------------------------


async def test_a_stream_that_breaks_after_content_counts_one_failure_and_cools_down(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream("partial", error=rate_limit(message="mid stream"), logged=True))
    core: Final = make_core(
        model_list=[deployment("g", "a", rpm=100, order=1), deployment("g", "b", order=2)],
        num_retries=0,
        allowed_fails=0,
    )

    stream: Final = await open_chat(core)
    with pytest.raises(litellm.RateLimitError, match="mid stream"):
        await collect_texts(stream)
    await settle()

    assert host.deployments_called() == ["a"]
    assert core.health().failures_this_minute("a") == 1
    assert successes(core, "a") is None
    assert cooled(core, "a", "b") == ["a"]
    # One request when the stream opened, one more for the failure.
    assert await core.health().usage("g") == (0, 2)


async def test_a_stream_that_breaks_after_content_within_the_allowed_fails_does_not_cool_down(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream("partial", error=rate_limit(message="mid stream"), logged=True))
    core: Final = make_core(model_list=group("a"), num_retries=0, allowed_fails=3)

    stream: Final = await open_chat(core)
    with pytest.raises(litellm.RateLimitError, match="mid stream"):
        await collect_texts(stream)
    await settle()

    assert core.health().failures_this_minute("a") == 1
    assert successes(core, "a") is None
    assert cooled(core, "a") == []


async def test_a_stream_that_breaks_before_content_and_falls_back_counts_one_failure_and_one_success(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer(
        "openai/a",
        chat_stream("x", error=rate_limit(message="cut"), fail_after=1, leading_role_chunk=True, logged=True),
    )
    host.answer("openai/b", chat_stream("B", usage=USAGE, logged=True))
    core: Final = make_core(
        model_list=[deployment("a", "a", rpm=100), deployment("b", "b", rpm=100)],
        fallbacks=[{"a": ["b"]}],
        num_retries=0,
        allowed_fails=3,
    )

    stream: Final = await open_chat(core, model="a")
    assert await collect_texts(stream) == "B"
    await settle()

    assert host.deployments_called() == ["a", "b"]
    assert core.health().failures_this_minute("a") == 1
    assert successes(core, "a") is None
    assert await core.health().usage("a") == (0, 2)
    assert successes(core, "b") == 1
    assert await core.health().usage("b") == (12, 1)


async def test_a_stream_that_fails_on_its_first_chunk_counts_its_failure_once(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer(
        "openai/a",
        chat_stream("x", error=rate_limit(), fail_after=0, logged=True),
        chat_stream("ok", usage=USAGE, logged=True),
    )
    core: Final = make_core(model_list=group("a", rpm=100), num_retries=1, allowed_fails=3)

    stream: Final = await open_chat(core)
    assert await collect_texts(stream) == "ok"
    await settle()

    assert host.deployments_called() == ["a", "a"]
    assert core.health().failures_this_minute("a") == 1
    assert successes(core, "a") == 1
    assert await core.health().usage("g") == (12, 2)


# --- Responses streams ---------------------------------------------------------------------------


def completed_event(input_tokens: int, output_tokens: int) -> ResponseCompletedEvent:
    response: Final = ResponsesAPIResponse.model_construct(
        id="resp",
        background=None,
        usage=ResponseAPIUsage(
            input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=input_tokens + output_tokens
        ),
    )
    return ResponseCompletedEvent(type="response.completed", response=response)  # pyright: ignore[reportArgumentType]  # the enum accepts its value


async def test_a_consumed_responses_stream_counts_its_final_tokens_and_one_request(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer(
        "openai/a",
        responses_stream(
            Event("response.created"),
            Event("response.output_text.delta"),
            completed_event(4, 8),
            logged=True,
            total_tokens=12,
        ),
    )
    core: Final = make_core(model_list=group("a", rpm=100))

    stream: Final = await open_responses(core)
    assert (await collect_event_types(stream))[-1] == "response.completed"
    await settle()

    assert await core.health().usage("g") == (12, 1)
    assert successes(core, "a") == 1


async def test_a_responses_stream_that_breaks_after_output_counts_one_failure_and_cools_down(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer(
        "openai/a",
        responses_stream(
            Event("response.created"),
            Event("response.output_text.delta", delta="partial"),
            error=rate_limit(message="mid stream"),
            logged=True,
        ),
    )
    core: Final = make_core(
        model_list=[deployment("g", "a", rpm=100, order=1), deployment("g", "b", order=2)],
        num_retries=0,
        allowed_fails=0,
    )

    stream: Final = await open_responses(core)
    with pytest.raises(litellm.RateLimitError, match="mid stream"):
        await collect_event_types(stream)
    await settle()

    assert core.health().failures_this_minute("a") == 1
    assert successes(core, "a") is None
    assert cooled(core, "a", "b") == ["a"]
    assert await core.health().usage("g") == (0, 2)


# --- /v1/messages streams ------------------------------------------------------------------------

MESSAGE_START_WITH_USAGE: Final = sse(
    "message_start",
    {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [],
            "usage": {"input_tokens": 3, "cache_read_input_tokens": 1, "output_tokens": 1},
        },
    },
)
MESSAGE_DELTA_WITH_USAGE: Final = sse(
    "message_delta",
    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 8}},
)


async def test_a_consumed_messages_stream_counts_its_final_tokens_and_one_request(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer(
        "openai/a",
        anthropic_stream(
            MESSAGE_START_WITH_USAGE,
            text_delta("hello"),
            # the delta frame split across two byte chunks
            MESSAGE_DELTA_WITH_USAGE[:30],
            MESSAGE_DELTA_WITH_USAGE[30:],
            MESSAGE_STOP,
            logged=True,
            total_tokens=12,
        ),
    )
    core: Final = make_core(model_list=group("a", rpm=100))

    stream: Final = await open_messages(core)
    assert len(await collect_bytes(stream)) == 5
    await settle()

    assert await core.health().usage("g") == (12, 1)
    assert successes(core, "a") == 1


async def test_a_messages_stream_that_breaks_after_content_counts_one_failure_and_cools_down(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer(
        "openai/a",
        anthropic_stream(
            MESSAGE_START_WITH_USAGE, text_delta("x"), error=rate_limit(message="mid stream"), logged=True
        ),
    )
    core: Final = make_core(
        model_list=[deployment("g", "a", rpm=100, order=1), deployment("g", "b", order=2)],
        num_retries=0,
        allowed_fails=0,
    )

    stream: Final = await open_messages(core)
    with pytest.raises(litellm.RateLimitError, match="mid stream"):
        await collect_bytes(stream)
    await settle()

    assert core.health().failures_this_minute("a") == 1
    assert successes(core, "a") is None
    assert cooled(core, "a", "b") == ["a"]
    assert await core.health().usage("g") == (0, 2)
