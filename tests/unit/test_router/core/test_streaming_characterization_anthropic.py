"""
Characterization of /v1/messages streaming through a router core: lifecycle frames held back while a
retry or fallback can still take over, provider ``event: error`` frames, raised stream errors, safeguard
refusals, same-group stream retries and their budget, and the hidden params the returned stream carries.

Scripted streams are async iterators of raw SSE byte frames (``streaming_helpers.anthropic_stream``).
"""

from collections.abc import Callable, Mapping
from typing import Final

import pytest

import litellm
from litellm.router_core import RouterCore
from litellm.types.router import RetryPolicy
from tests.unit.test_router.core.scripted_host import (
    ScriptedHost,
    bad_request,
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
    text_delta,
)

install_orjson_stub_if_missing()

MakeCore = Callable[..., RouterCore]
GROUPS_AB: Final = [deployment("a", "a"), deployment("b", "b")]
REFUSAL_DETAILS: Final = {"type": "refusal", "category": "cyber"}


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


async def test_held_frames_are_flushed_when_the_stream_ends_without_content(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", anthropic_stream(MESSAGE_START, BLOCK_START, MESSAGE_STOP))
    core = make_core(model_list=GROUPS_AB, fallbacks=[{"a": ["b"]}], num_retries=0)
    frames = await collect_bytes(await open_stream(core))
    assert frames == [MESSAGE_START, BLOCK_START, MESSAGE_STOP]
    assert host.deployments_called() == ["a"]


async def test_two_hundred_held_frames_commit_the_stream_so_a_later_error_frame_is_forwarded(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", anthropic_stream(*([BLOCK_START] * 201), error_frame()))
    host.answer("openai/b", fallback_ok())
    core = make_core(model_list=GROUPS_AB, fallbacks=[{"a": ["b"]}], num_retries=0)
    frames = await collect_bytes(await open_stream(core))
    assert frames == [*([BLOCK_START] * 201), error_frame()]
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
