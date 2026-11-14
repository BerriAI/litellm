"""
Characterization of /v1/messages streaming through a router core: lifecycle frames held back while a
retry or fallback can still take over, except ``message_start``, sent live while only a retry can,
provider ``event: error`` frames, raised stream errors, safeguard refusals, same-group stream retries and
their budget, and the hidden params the returned stream carries.

Scripted streams are async iterators of raw SSE byte frames (``streaming_helpers.anthropic_stream``).
"""

from collections.abc import AsyncIterator, Callable, Mapping
from typing import Final

import pytest

import litellm
from litellm.exceptions import MidStreamFallbackError
from litellm.router_core import RouterCore
from litellm.router_rust.invoke.streaming import FallbackAwareAnthropicMessagesStream
from litellm.types.router import RetryPolicy
from litellm.types.utils import Usage
from tests.unit.test_router.core.scripted_host import (
    ScriptedHost,
    agentic_followup_failure,
    bad_request,
    context_window,
    deployment,
    rate_limit,
    server_error,
)
from tests.unit.test_router.core.streaming_helpers import (
    BLOCK_START,
    MESSAGE_START,
    MESSAGE_STOP,
    MESSAGES,
    PING,
    ByteStream,
    anthropic_stream,
    collect_bytes,
    error_frame,
    install_orjson_stub_if_missing,
    refusal_delta,
    request_logging,
    sse,
    text_delta,
)

install_orjson_stub_if_missing()

MakeCore = Callable[..., RouterCore]
GROUPS_AB: Final = [deployment("a", "a"), deployment("b", "b")]
REFUSAL_DETAILS: Final = {"type": "refusal", "category": "cyber"}


THINKING_START: Final = sse(
    "content_block_start",
    {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
)
SIGNATURE_DELTA: Final = sse(
    "content_block_delta",
    {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}},
)
BLOCK_STOP: Final = sse("content_block_stop", {"type": "content_block_stop", "index": 0})
SSE_COMMENT: Final = b": upstream keepalive\n\n"
DATA_ONLY_PING: Final = b'data: {"type": "ping"}\n\n'
RETRY_MESSAGE_START: Final = sse(
    "message_start",
    {
        "type": "message_start",
        "message": {
            "id": "msg_2",
            "type": "message",
            "role": "assistant",
            "content": [],
            "usage": {"input_tokens": 1, "cache_read_input_tokens": 9},
        },
    },
)


def tool_use_start(tool_use_id: str) -> bytes:
    block: Final = {"type": "tool_use", "id": tool_use_id, "name": "lookup", "input": {}}
    return sse("content_block_start", {"type": "content_block_start", "index": 0, "content_block": block})


def redacted_thinking_start(data: str) -> bytes:
    block: Final = {"type": "redacted_thinking", "data": data}
    return sse("content_block_start", {"type": "content_block_start", "index": 0, "content_block": block})


def input_json_delta(partial_json: str) -> bytes:
    delta: Final = {"type": "input_json_delta", "partial_json": partial_json}
    return sse("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": delta})


def recording_stream(
    received: list[bytes], reads: list[tuple[bytes, ...]], *frames: bytes
) -> Callable[..., AsyncIterator[bytes]]:
    async def stream() -> AsyncIterator[bytes]:
        for frame in frames:
            reads.append(tuple(received))
            yield frame

    def answer(**_request: object) -> AsyncIterator[bytes]:  # kwargs-ok: the host passes the request kwargs
        return stream()

    return answer


async def read_into(stream: object, received: list[bytes]) -> None:
    assert isinstance(stream, FallbackAwareAnthropicMessagesStream)
    async for frame in stream:
        received.append(frame)


def event_names(frames: list[bytes]) -> list[bytes]:
    return [frame.split(b"\n", 1)[0].removeprefix(b"event: ") for frame in frames]


def fallback_ok(opened: list[ByteStream] | None = None) -> Callable[..., ByteStream]:
    return anthropic_stream(
        MESSAGE_START,
        text_delta("B"),
        MESSAGE_STOP,
        hidden_params={"model_id": "b-stream", "additional_headers": {"x-provider-request-id": "req-b"}},
        opened=opened,
    )


def answered_ok(opened: list[ByteStream] | None = None) -> Callable[..., ByteStream]:
    return anthropic_stream(MESSAGE_START, text_delta("A"), MESSAGE_STOP, opened=opened)


async def open_stream(core: RouterCore, **request: object) -> object:  # kwargs-ok: request kwargs
    return await core.aanthropic_messages(model="a", messages=MESSAGES, max_tokens=5, stream=True, **request)  # pyright: ignore[reportAttributeAccessIssue]  # endpoint method


def hidden(stream: object) -> Mapping[str, object]:
    return getattr(stream, "_hidden_params")


def headers(stream: object) -> Mapping[str, object]:
    return hidden(stream)["additional_headers"]  # pyright: ignore[reportReturnType]  # a dict of headers


# --- provider error frames before content ------------------------------------------------------


async def test_an_error_frame_before_content_falls_back_and_drops_the_primarys_held_frames(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", anthropic_stream(MESSAGE_START, BLOCK_START, error_frame()))
    host.answer("openai/b", fallback_ok())
    core = make_core(model_list=GROUPS_AB, fallbacks=[{"a": ["b"]}], num_retries=0)
    frames = await collect_bytes(await open_stream(core))
    assert frames == [MESSAGE_START, text_delta("B"), MESSAGE_STOP]
    assert host.deployments_called() == ["a", "b"]


async def test_an_error_frame_before_content_with_nothing_to_take_over_is_forwarded_verbatim(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", anthropic_stream(MESSAGE_START, error_frame()))
    core = make_core(model_list=GROUPS_AB, num_retries=0)
    frames = await collect_bytes(await open_stream(core))
    assert frames == [MESSAGE_START, error_frame()]
    assert host.deployments_called() == ["a"]


async def test_an_error_frame_split_across_byte_chunks_is_still_recognised(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    frame = error_frame()
    host.answer("openai/a", anthropic_stream(MESSAGE_START, frame[:25], frame[25:]))
    host.answer("openai/b", fallback_ok())
    core = make_core(model_list=GROUPS_AB, fallbacks=[{"a": ["b"]}], num_retries=0)
    frames = await collect_bytes(await open_stream(core))
    assert frames == [MESSAGE_START, text_delta("B"), MESSAGE_STOP]
    assert host.deployments_called() == ["a", "b"]


async def test_an_error_frame_after_content_is_forwarded_without_fallback(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", anthropic_stream(MESSAGE_START, text_delta("x"), error_frame()))
    host.answer("openai/b", fallback_ok())
    core = make_core(model_list=GROUPS_AB, fallbacks=[{"a": ["b"]}], num_retries=0)
    frames = await collect_bytes(await open_stream(core))
    assert frames == [MESSAGE_START, text_delta("x"), error_frame()]
    assert host.deployments_called() == ["a"]


async def test_a_non_retriable_error_frame_is_forwarded_even_with_a_fallback(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    frame = error_frame("invalid_request_error", "bad request")
    host.answer("openai/a", anthropic_stream(MESSAGE_START, frame))
    host.answer("openai/b", fallback_ok())
    core = make_core(model_list=GROUPS_AB, fallbacks=[{"a": ["b"]}], num_retries=0)
    frames = await collect_bytes(await open_stream(core))
    assert frames == [MESSAGE_START, frame]
    assert host.deployments_called() == ["a"]


async def test_disable_fallbacks_forwards_a_retriable_error_frame(make_core: MakeCore, host: ScriptedHost) -> None:
    host.answer("openai/a", anthropic_stream(MESSAGE_START, error_frame()))
    host.answer("openai/b", fallback_ok())
    core = make_core(model_list=GROUPS_AB, fallbacks=[{"a": ["b"]}], num_retries=0)
    frames = await collect_bytes(await open_stream(core, disable_fallbacks=True))
    assert frames == [MESSAGE_START, error_frame()]
    assert host.deployments_called() == ["a"]


async def test_when_the_fallback_after_an_error_frame_fails_the_frames_error_is_raised(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", anthropic_stream(MESSAGE_START, error_frame("overloaded_error", "primary overloaded")))
    host.answer("openai/b", server_error())
    core = make_core(model_list=GROUPS_AB, fallbacks=[{"a": ["b"]}], num_retries=0)
    stream = await open_stream(core)
    with pytest.raises(litellm.InternalServerError, match="primary overloaded"):
        await collect_bytes(stream)
    assert host.deployments_called() == ["a", "b"]


# --- held lifecycle frames --------------------------------------------------------------------


async def test_a_ping_is_forwarded_live_while_lifecycle_frames_are_held(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", anthropic_stream(MESSAGE_START, PING, error_frame()))
    host.answer("openai/b", fallback_ok())
    core = make_core(model_list=GROUPS_AB, fallbacks=[{"a": ["b"]}], num_retries=0)
    frames = await collect_bytes(await open_stream(core))
    assert frames == [PING, MESSAGE_START, text_delta("B"), MESSAGE_STOP]


async def test_with_nothing_to_take_over_every_frame_is_forwarded_live_in_order(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", anthropic_stream(MESSAGE_START, PING, BLOCK_START, error_frame()))
    core = make_core(model_list=GROUPS_AB, num_retries=0)
    frames = await collect_bytes(await open_stream(core))
    assert frames == [MESSAGE_START, PING, BLOCK_START, error_frame()]


async def test_held_frames_are_flushed_in_order_when_content_arrives(make_core: MakeCore, host: ScriptedHost) -> None:
    host.answer("openai/a", anthropic_stream(MESSAGE_START, BLOCK_START, text_delta("A"), MESSAGE_STOP))
    core = make_core(model_list=GROUPS_AB, fallbacks=[{"a": ["b"]}], num_retries=0)
    frames = await collect_bytes(await open_stream(core))
    assert frames == [MESSAGE_START, BLOCK_START, text_delta("A"), MESSAGE_STOP]
    assert host.deployments_called() == ["a"]


@pytest.mark.parametrize(
    "router_arguments",
    [{"fallbacks": [{"a": ["b"]}], "num_retries": 0}, {"num_retries": 2}],
    ids=["held-for-a-fallback", "message-start-live-for-a-retry"],
)
async def test_lifecycle_frames_reach_the_client_in_order_when_the_stream_ends_without_content(
    make_core: MakeCore, host: ScriptedHost, router_arguments: dict[str, object]
) -> None:
    host.answer("openai/a", anthropic_stream(MESSAGE_START, BLOCK_START, MESSAGE_STOP))
    core = make_core(model_list=GROUPS_AB, **router_arguments)
    frames = await collect_bytes(await open_stream(core))
    assert frames == [MESSAGE_START, BLOCK_START, MESSAGE_STOP]
    assert host.deployments_called() == ["a"]


@pytest.mark.parametrize(
    "router_arguments",
    [{"fallbacks": [{"a": ["b"]}], "num_retries": 0}, {"num_retries": 2}],
    ids=["held-for-a-fallback", "message-start-live-for-a-retry"],
)
async def test_two_hundred_frames_before_content_commit_the_stream_so_a_later_error_frame_is_forwarded(
    make_core: MakeCore, host: ScriptedHost, router_arguments: dict[str, object]
) -> None:
    host.answer("openai/a", anthropic_stream(MESSAGE_START, *([BLOCK_START] * 200), error_frame()))
    host.answer("openai/b", fallback_ok())
    core = make_core(model_list=GROUPS_AB, **router_arguments)
    frames = await collect_bytes(await open_stream(core))
    assert frames == [MESSAGE_START, *([BLOCK_START] * 200), error_frame()]
    assert host.deployments_called() == ["a"]


# --- raised stream errors ----------------------------------------------------------------------


async def test_a_raised_rate_limit_before_content_falls_back(make_core: MakeCore, host: ScriptedHost) -> None:
    host.answer("openai/a", anthropic_stream(MESSAGE_START, error=rate_limit()))
    host.answer("openai/b", fallback_ok())
    core = make_core(model_list=GROUPS_AB, fallbacks=[{"a": ["b"]}], num_retries=0)
    frames = await collect_bytes(await open_stream(core))
    assert frames == [MESSAGE_START, text_delta("B"), MESSAGE_STOP]
    assert host.deployments_called() == ["a", "b"]


async def test_a_raised_bad_request_before_content_is_raised_without_fallback(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", anthropic_stream(MESSAGE_START, error=bad_request(message="raised 400")))
    host.answer("openai/b", fallback_ok())
    core = make_core(model_list=GROUPS_AB, fallbacks=[{"a": ["b"]}], num_retries=0)
    stream = await open_stream(core)
    with pytest.raises(litellm.BadRequestError, match="raised 400"):
        await collect_bytes(stream)
    assert host.deployments_called() == ["a"]


async def test_a_raised_error_after_content_is_raised_without_fallback(make_core: MakeCore, host: ScriptedHost) -> None:
    host.answer("openai/a", anthropic_stream(MESSAGE_START, text_delta("x"), error=rate_limit(message="after")))
    host.answer("openai/b", fallback_ok())
    core = make_core(model_list=GROUPS_AB, fallbacks=[{"a": ["b"]}], num_retries=0)
    stream = await open_stream(core)
    with pytest.raises(litellm.RateLimitError, match="after"):
        await collect_bytes(stream)
    assert host.deployments_called() == ["a"]


async def test_a_raised_server_error_before_content_is_retried_within_the_group(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("*", anthropic_stream(MESSAGE_START, error=server_error()), answered_ok())
    core = make_core(model_list=[deployment("a", "a1"), deployment("a", "a2")], num_retries=1)
    frames = await collect_bytes(await open_stream(core))
    assert frames == [MESSAGE_START, text_delta("A"), MESSAGE_STOP]
    assert len(host.deployments_called()) == 2
    assert len(host.sleeps) == 1


# --- same-group stream retries -----------------------------------------------------------------


async def test_a_failed_agentic_followup_before_content_skips_the_retries_for_the_fallback(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    failed: Final = agentic_followup_failure(server_error())
    host.answer("openai/a", anthropic_stream(MESSAGE_START, error=failed), answered_ok())
    host.answer("openai/b", fallback_ok())
    core = make_core(model_list=GROUPS_AB, fallbacks=[{"a": ["b"]}], num_retries=2)
    frames = await collect_bytes(await open_stream(core))
    assert frames == [MESSAGE_START, text_delta("B"), MESSAGE_STOP]
    assert host.deployments_called() == ["a", "b"]
    assert host.sleeps == []


async def test_a_stream_retry_carries_the_superseded_attempt_usage(make_core: MakeCore, host: ScriptedHost) -> None:
    logging_obj: Final = request_logging()
    usage: Final = Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    failing: Final = anthropic_stream(MESSAGE_START, error=server_error())

    def fails_after_usage(**kwargs: object) -> ByteStream:  # kwargs-ok: the request
        logging_obj.model_call_details["combined_usage_object"] = usage
        return failing(**kwargs)

    host.answer("openai/a", fails_after_usage, answered_ok())
    core = make_core(model_list=[deployment("a", "a")], num_retries=1)
    await collect_bytes(await open_stream(core, litellm_logging_obj=logging_obj))
    assert host.deployments_called() == ["a", "a"]
    assert logging_obj.model_call_details["failed_attempts_usage"] == ((usage, 0.0),)


async def test_an_error_frame_before_content_is_retried_within_the_group(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", anthropic_stream(MESSAGE_START, error_frame()), answered_ok())
    core = make_core(model_list=[deployment("a", "a")], num_retries=2)
    stream = await open_stream(core)
    frames = await collect_bytes(stream)
    assert frames == [MESSAGE_START, text_delta("A"), MESSAGE_STOP]
    assert host.deployments_called() == ["a", "a"]
    assert len(host.sleeps) == 1
    assert headers(stream)["x-litellm-attempted-retries"] == 1
    assert headers(stream)["x-litellm-max-retries"] == 2


async def test_a_retry_policy_grants_stream_retries_when_num_retries_is_zero(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", anthropic_stream(MESSAGE_START, error_frame("rate_limit_error", "slow")), answered_ok())
    core = make_core(
        model_list=[deployment("a", "a")], num_retries=0, retry_policy=RetryPolicy(RateLimitErrorRetries=2)
    )
    stream = await open_stream(core)
    frames = await collect_bytes(stream)
    assert frames == [MESSAGE_START, text_delta("A"), MESSAGE_STOP]
    assert host.deployments_called() == ["a", "a"]
    assert headers(stream)["x-litellm-max-retries"] == 2


async def test_request_num_retries_zero_opts_out_of_the_retry_policy_for_a_stream(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    frame = error_frame("rate_limit_error", "slow")
    host.answer("openai/a", anthropic_stream(MESSAGE_START, frame), answered_ok())
    core = make_core(
        model_list=[deployment("a", "a")], num_retries=0, retry_policy=RetryPolicy(RateLimitErrorRetries=2)
    )
    frames = await collect_bytes(await open_stream(core, num_retries=0))
    assert frames == [MESSAGE_START, frame]
    assert host.deployments_called() == ["a"]


async def test_the_deployment_num_retries_is_the_stream_retry_budget(make_core: MakeCore, host: ScriptedHost) -> None:
    host.answer("openai/a", anthropic_stream(MESSAGE_START, error_frame()), answered_ok())
    core = make_core(model_list=[deployment("a", "a", num_retries=1)], num_retries=0)
    stream = await open_stream(core)
    frames = await collect_bytes(stream)
    assert frames == [MESSAGE_START, text_delta("A"), MESSAGE_STOP]
    assert headers(stream)["x-litellm-max-retries"] == 1


async def test_stream_retries_stop_at_the_budget_and_forward_the_last_error_frame(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", anthropic_stream(MESSAGE_START, error_frame()))
    core = make_core(model_list=[deployment("a", "a")], num_retries=2)
    frames = await collect_bytes(await open_stream(core))
    assert frames == [MESSAGE_START, error_frame()]
    assert host.deployments_called() == ["a", "a", "a"]
    assert len(host.sleeps) == 2


async def test_spent_stream_retries_hand_over_to_the_fallback_chain(make_core: MakeCore, host: ScriptedHost) -> None:
    host.answer("openai/a", anthropic_stream(MESSAGE_START, error_frame()))
    host.answer("openai/b", fallback_ok())
    core = make_core(model_list=GROUPS_AB, fallbacks=[{"a": ["b"]}], num_retries=1)
    stream = await open_stream(core)
    frames = await collect_bytes(stream)
    assert frames == [MESSAGE_START, text_delta("B"), MESSAGE_STOP]
    assert host.deployments_called() == ["a", "a", "b"]
    assert len(host.sleeps) == 1


async def test_a_superseded_attempt_and_every_later_stream_are_closed(make_core: MakeCore, host: ScriptedHost) -> None:
    opened: list[ByteStream] = []
    host.answer("openai/a", anthropic_stream(MESSAGE_START, error_frame(), opened=opened))
    host.answer("openai/b", fallback_ok(opened))
    core = make_core(model_list=GROUPS_AB, fallbacks=[{"a": ["b"]}], num_retries=1)
    await collect_bytes(await open_stream(core))
    assert len(opened) == 3
    assert all(stream.closed for stream in opened)


# --- message_start sent live while only a retry can take over --------------------------------


@pytest.mark.parametrize(
    ("frames", "before_content"),
    [
        ((MESSAGE_START, THINKING_START, SIGNATURE_DELTA, BLOCK_STOP, MESSAGE_STOP), (MESSAGE_START,)),
        (
            (SSE_COMMENT, MESSAGE_START, THINKING_START, SIGNATURE_DELTA, BLOCK_STOP, MESSAGE_STOP),
            (SSE_COMMENT, MESSAGE_START),
        ),
        (
            (
                MESSAGE_START[:20],
                MESSAGE_START[20:] + THINKING_START[:15],
                THINKING_START[15:],
                SIGNATURE_DELTA,
                BLOCK_STOP,
                MESSAGE_STOP,
            ),
            (MESSAGE_START,),
        ),
        (
            (SSE_COMMENT + MESSAGE_START + THINKING_START, SIGNATURE_DELTA, BLOCK_STOP, MESSAGE_STOP),
            (SSE_COMMENT + MESSAGE_START,),
        ),
    ],
    ids=["whole-frames", "after-an-sse-comment", "split-across-chunks", "sharing-a-chunk-with-a-block-start"],
)
async def test_with_only_a_retry_left_message_start_reaches_the_client_before_content_and_block_starts_wait(
    make_core: MakeCore, host: ScriptedHost, frames: tuple[bytes, ...], before_content: tuple[bytes, ...]
) -> None:
    received: list[bytes] = []
    reads: list[tuple[bytes, ...]] = []
    host.answer("openai/a", recording_stream(received, reads, *frames))
    core = make_core(model_list=[deployment("a", "a")], num_retries=2)
    await read_into(await open_stream(core), received)
    assert reads[frames.index(SIGNATURE_DELTA)] == before_content
    assert b"".join(received) == b"".join(frames)


async def test_with_only_a_retry_left_decoded_event_chunks_reach_the_client_without_a_retry(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    events: Final = (
        {"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant", "content": []}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "A"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    )

    async def decoded_events() -> AsyncIterator[Mapping[str, object]]:
        for event in events:
            yield event

    def answer(
        **_request: object,
    ) -> AsyncIterator[Mapping[str, object]]:  # kwargs-ok: the host passes the request kwargs
        return decoded_events()

    host.answer("openai/a", answer)
    core = make_core(model_list=[deployment("a", "a")], num_retries=2)
    stream: Final = await open_stream(core)
    assert isinstance(stream, FallbackAwareAnthropicMessagesStream)
    assert [event async for event in stream] == list(events)
    assert host.deployments_called() == ["a"]


@pytest.mark.parametrize(
    ("router_arguments", "request_kwargs"),
    [
        ({"fallbacks": [{"a": ["b"]}]}, {}),
        ({"content_policy_fallbacks": [{"a": ["b"]}]}, {}),
        ({"default_fallbacks": ["b"]}, {"fallbacks": []}),
        ({"enable_weighted_failover": True}, {}),
    ],
    ids=["fallbacks", "content-policy-fallbacks", "refusal-fallback-only", "weighted-failover"],
)
async def test_with_a_fallback_left_lifecycle_frames_are_held_until_content_even_with_retries(
    make_core: MakeCore, host: ScriptedHost, router_arguments: dict[str, object], request_kwargs: dict[str, object]
) -> None:
    received: list[bytes] = []
    reads: list[tuple[bytes, ...]] = []
    frames: Final = (MESSAGE_START, THINKING_START, PING, SIGNATURE_DELTA, BLOCK_STOP, MESSAGE_STOP)
    host.answer("openai/a", recording_stream(received, reads, *frames))
    core = make_core(model_list=GROUPS_AB, num_retries=2, **router_arguments)
    await read_into(await open_stream(core, **request_kwargs), received)
    assert reads[frames.index(SIGNATURE_DELTA)] == (PING,)
    assert received == [PING, MESSAGE_START, THINKING_START, SIGNATURE_DELTA, BLOCK_STOP, MESSAGE_STOP]


@pytest.mark.parametrize(
    ("first_block", "retry_block", "delta"),
    [
        (THINKING_START, THINKING_START, SIGNATURE_DELTA),
        (THINKING_START, BLOCK_START, text_delta("A")),
        (redacted_thinking_start("abc") + BLOCK_STOP, redacted_thinking_start("xyz") + BLOCK_STOP, text_delta("A")),
        (tool_use_start("toolu_1"), tool_use_start("toolu_2"), input_json_delta("{}")),
    ],
    ids=["thinking", "text-after-thinking", "redacted-thinking-with-new-data", "tool-use-with-a-new-id"],
)
async def test_a_retry_after_a_live_message_start_drops_its_own_and_sends_its_blocks_in_place_of_the_held_ones(
    make_core: MakeCore, host: ScriptedHost, first_block: bytes, retry_block: bytes, delta: bytes
) -> None:
    host.answer(
        "openai/a",
        anthropic_stream(MESSAGE_START, first_block, PING, error_frame()),
        anthropic_stream(
            PING, RETRY_MESSAGE_START[:25], RETRY_MESSAGE_START[25:] + retry_block, delta, BLOCK_STOP, MESSAGE_STOP
        ),
    )
    core = make_core(model_list=[deployment("a", "a")], num_retries=1)
    frames = await collect_bytes(await open_stream(core))
    assert frames == [MESSAGE_START, PING, PING, retry_block, delta, BLOCK_STOP, MESSAGE_STOP]
    assert host.deployments_called() == ["a", "a"]


async def test_a_keepalive_sharing_a_chunk_with_the_retrys_message_start_still_reaches_the_client(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer(
        "openai/a",
        anthropic_stream(MESSAGE_START, error_frame()),
        anthropic_stream(SSE_COMMENT + RETRY_MESSAGE_START, text_delta("A"), MESSAGE_STOP),
    )
    core = make_core(model_list=[deployment("a", "a")], num_retries=1)
    frames = await collect_bytes(await open_stream(core))
    assert b"".join(frames) == MESSAGE_START + SSE_COMMENT + text_delta("A") + MESSAGE_STOP


async def test_a_context_window_fallback_after_a_live_message_start_drops_the_fallbacks_message_start(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    bridged_cut: Final = MidStreamFallbackError(
        message="too long",
        model="a",
        llm_provider="openai",
        original_exception=context_window("a"),
        is_pre_first_chunk=True,
    )
    host.answer("openai/a", anthropic_stream(MESSAGE_START, error=bridged_cut))
    host.answer("openai/b", fallback_ok())
    core = make_core(model_list=GROUPS_AB, context_window_fallbacks=[{"a": ["b"]}], num_retries=2)
    frames = await collect_bytes(await open_stream(core))
    assert frames == [MESSAGE_START, text_delta("B"), MESSAGE_STOP]
    assert host.deployments_called() == ["a", "b"]


@pytest.mark.parametrize(
    ("first_attempt", "expected"),
    [
        ((DATA_ONLY_PING, MESSAGE_START), [RETRY_MESSAGE_START, text_delta("A"), MESSAGE_STOP]),
        ((MESSAGE_START, DATA_ONLY_PING), [MESSAGE_START, text_delta("A"), MESSAGE_STOP]),
    ],
    ids=["before-message-start", "after-message-start"],
)
async def test_an_unrecognized_frame_before_content_is_held_and_leaves_the_retry_available(
    make_core: MakeCore, host: ScriptedHost, first_attempt: tuple[bytes, ...], expected: list[bytes]
) -> None:
    host.answer(
        "openai/a",
        anthropic_stream(*first_attempt, error_frame()),
        anthropic_stream(RETRY_MESSAGE_START, text_delta("A"), MESSAGE_STOP),
    )
    core = make_core(model_list=[deployment("a", "a")], num_retries=1)
    frames = await collect_bytes(await open_stream(core))
    assert frames == expected
    assert host.deployments_called() == ["a", "a"]


async def test_a_split_error_frame_after_a_live_message_start_is_retried(make_core: MakeCore, host: ScriptedHost) -> None:
    frame = error_frame()
    host.answer("openai/a", anthropic_stream(MESSAGE_START, frame[:30], frame[30:]), answered_ok())
    core = make_core(model_list=[deployment("a", "a")], num_retries=2)
    frames = await collect_bytes(await open_stream(core))
    assert frames == [MESSAGE_START, text_delta("A"), MESSAGE_STOP]
    assert host.deployments_called() == ["a", "a"]


async def test_a_retry_that_errors_before_its_message_start_forwards_its_error_frame(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    retry_error: Final = error_frame("api_error", "retry failed")
    host.answer("openai/a", anthropic_stream(MESSAGE_START, error_frame()), anthropic_stream(retry_error))
    core = make_core(model_list=[deployment("a", "a")], num_retries=1)
    frames = await collect_bytes(await open_stream(core))
    assert frames == [MESSAGE_START, retry_error]
    assert host.deployments_called() == ["a", "a"]


# --- safeguard refusals ------------------------------------------------------------------------


async def test_a_refusal_with_stop_details_before_content_falls_back(make_core: MakeCore, host: ScriptedHost) -> None:
    host.answer("openai/a", anthropic_stream(MESSAGE_START, refusal_delta(REFUSAL_DETAILS), MESSAGE_STOP))
    host.answer("openai/b", fallback_ok())
    core = make_core(model_list=GROUPS_AB, fallbacks=[{"a": ["b"]}], num_retries=0)
    frames = await collect_bytes(await open_stream(core))
    assert frames == [MESSAGE_START, text_delta("B"), MESSAGE_STOP]
    assert host.deployments_called() == ["a", "b"]


async def test_a_refusal_falls_back_through_the_content_policy_chain(make_core: MakeCore, host: ScriptedHost) -> None:
    host.answer("openai/a", anthropic_stream(MESSAGE_START, refusal_delta(REFUSAL_DETAILS), MESSAGE_STOP))
    host.answer("openai/b", fallback_ok())
    core = make_core(model_list=GROUPS_AB, content_policy_fallbacks=[{"a": ["b"]}], num_retries=0)
    frames = await collect_bytes(await open_stream(core))
    assert frames == [MESSAGE_START, text_delta("B"), MESSAGE_STOP]
    assert host.deployments_called() == ["a", "b"]


async def test_a_refusal_without_stop_details_is_forwarded(make_core: MakeCore, host: ScriptedHost) -> None:
    host.answer("openai/a", anthropic_stream(MESSAGE_START, refusal_delta(), MESSAGE_STOP))
    host.answer("openai/b", fallback_ok())
    core = make_core(model_list=GROUPS_AB, fallbacks=[{"a": ["b"]}], num_retries=0)
    frames = await collect_bytes(await open_stream(core))
    assert frames == [MESSAGE_START, refusal_delta(), MESSAGE_STOP]
    assert host.deployments_called() == ["a"]


@pytest.mark.parametrize(
    ("router_arguments", "request_kwargs"),
    [({"fallbacks": [{"a": ["b"]}]}, {"disable_fallbacks": True}), ({}, {})],
    ids=["fallbacks-disabled", "no-fallbacks"],
)
async def test_a_refusal_with_no_reachable_fallback_is_forwarded(
    make_core: MakeCore, host: ScriptedHost, router_arguments: dict[str, object], request_kwargs: dict[str, object]
) -> None:
    host.answer("openai/a", anthropic_stream(MESSAGE_START, refusal_delta(REFUSAL_DETAILS), MESSAGE_STOP))
    host.answer("openai/b", fallback_ok())
    core = make_core(model_list=GROUPS_AB, num_retries=0, **router_arguments)
    frames = await collect_bytes(await open_stream(core, **request_kwargs))
    assert frames == [MESSAGE_START, refusal_delta(REFUSAL_DETAILS), MESSAGE_STOP]
    assert host.deployments_called() == ["a"]


async def test_a_refusal_after_content_is_forwarded(make_core: MakeCore, host: ScriptedHost) -> None:
    host.answer("openai/a", anthropic_stream(MESSAGE_START, text_delta("x"), refusal_delta(REFUSAL_DETAILS)))
    host.answer("openai/b", fallback_ok())
    core = make_core(model_list=GROUPS_AB, fallbacks=[{"a": ["b"]}], num_retries=0)
    frames = await collect_bytes(await open_stream(core))
    assert frames == [MESSAGE_START, text_delta("x"), refusal_delta(REFUSAL_DETAILS)]
    assert host.deployments_called() == ["a"]


# --- what the fallback hands back --------------------------------------------------------------


async def test_a_fallback_that_answers_without_streaming_is_replayed_as_an_sse_lifecycle(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", anthropic_stream(MESSAGE_START, error_frame()))
    host.answer(
        "openai/b",
        {
            "id": "msg_b",
            "type": "message",
            "role": "assistant",
            "model": "b",
            "content": [{"type": "text", "text": "B"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )
    core = make_core(model_list=GROUPS_AB, fallbacks=[{"a": ["b"]}], num_retries=0)
    frames = await collect_bytes(await open_stream(core))
    assert event_names(frames) == [
        b"message_start",
        b"content_block_start",
        b"content_block_delta",
        b"content_block_stop",
        b"message_delta",
        b"message_stop",
    ]
    assert b'"B"' in frames[2]


# --- hidden params on the returned stream ------------------------------------------------------


async def test_a_stream_without_recovery_reports_the_serving_deployment(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", answered_ok())
    core = make_core(model_list=GROUPS_AB, fallbacks=[{"a": ["b"]}], num_retries=0)
    stream = await open_stream(core)
    await collect_bytes(stream)
    assert hidden(stream)["model_id"] == "a"
    assert headers(stream) == {
        "x-litellm-model-group": "a",
        "x-litellm-attempted-retries": 0,
        "x-litellm-attempted-fallbacks": 0,
    }


async def test_after_a_stream_fallback_the_stream_carries_the_fallbacks_hidden_params_and_headers(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", anthropic_stream(MESSAGE_START, error_frame()))
    host.answer("openai/b", fallback_ok())
    core = make_core(model_list=GROUPS_AB, fallbacks=[{"a": ["b"]}], num_retries=0)
    stream = await open_stream(core)
    await collect_bytes(stream)
    assert hidden(stream)["model_id"] == "b-stream"
    assert headers(stream) == {
        "x-litellm-model-group": "b",
        "x-litellm-attempted-retries": 0,
        "x-litellm-attempted-fallbacks": 1,
        "x-provider-request-id": "req-b",
    }


async def test_after_retries_then_a_fallback_the_retry_budget_header_lingers(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", anthropic_stream(MESSAGE_START, error_frame()))
    host.answer("openai/b", answered_ok())
    core = make_core(model_list=GROUPS_AB, fallbacks=[{"a": ["b"]}], num_retries=1)
    stream = await open_stream(core)
    await collect_bytes(stream)
    assert headers(stream) == {
        "x-litellm-model-group": "b",
        "x-litellm-attempted-retries": 0,
        "x-litellm-attempted-fallbacks": 1,
        "x-litellm-max-retries": 1,
    }


async def test_a_stream_retry_keeps_reporting_the_first_deployment_when_the_retry_has_no_hidden_params(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("*", anthropic_stream(MESSAGE_START, error=server_error()), answered_ok())
    core = make_core(model_list=[deployment("a", "a1"), deployment("a", "a2")], num_retries=1)
    stream = await open_stream(core)
    await collect_bytes(stream)
    first, _retried = host.deployments_called()
    assert hidden(stream)["model_id"] == first
    assert headers(stream)["x-litellm-attempted-retries"] == 1
