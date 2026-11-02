"""
Characterization of chat-completion streaming through a router core: the first chunk read inside the
attempt, mid-stream errors and their fallback, mid-stream continuation, the sync twin, and the stream
timeout sent to the host.

Each scripted stream is a ``CustomStreamWrapper`` built by ``streaming_helpers.chat_stream``; a stream
that raises before its first chunk fails inside the attempt (retries and fallbacks run as for any error),
while one that raises later fails while the caller iterates.
"""

import asyncio
from collections.abc import Callable, Mapping
from typing import Final

import pytest

import litellm
from litellm.exceptions import MidStreamFallbackError
from litellm.router_core import RouterCore
from tests.unit.test_router.core.scripted_host import (
    ScriptedHost,
    bad_request,
    content_policy,
    context_window,
    deployment,
    rate_limit,
    server_error,
)
from tests.unit.test_router.core.streaming_helpers import (
    MESSAGES,
    chat_stream,
    collect_texts,
    collect_texts_sync,
    drain_into,
)

MakeCore = Callable[..., RouterCore]
PREFILL_MODEL: Final = "anthropic/claude-sonnet-4-5"
"""A model whose deployments accept an assistant prefill, so a continuation may land on them."""
GROUPS_AB: Final = [deployment("a", "a"), deployment("b", "b")]


def build(make_core: MakeCore, **arguments: object) -> RouterCore:  # kwargs-ok: the router's constructor arguments
    return make_core(**{"model_list": GROUPS_AB, "fallbacks": [{"a": ["b"]}], "num_retries": 0, **arguments})


async def open_stream(core: RouterCore, **request: object) -> object:  # kwargs-ok: request kwargs
    return await core.acompletion(model="a", messages=MESSAGES, stream=True, **request)  # pyright: ignore[reportAttributeAccessIssue]  # endpoint method


def headers(response: object) -> Mapping[str, object]:
    return getattr(response, "_hidden_params")["additional_headers"]


def model_id(response: object) -> object:
    return getattr(response, "_hidden_params").get("model_id")


# --- the first chunk is read inside the attempt -----------------------------------------------


async def test_a_stream_that_fails_on_its_first_chunk_is_retried_inside_the_attempt(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream("x", error=rate_limit(), fail_after=0), chat_stream("ok"))
    core = make_core(model_list=[deployment("a", "a")], num_retries=2)
    stream = await open_stream(core)
    assert await collect_texts(stream) == "ok"
    assert host.deployments_called() == ["a", "a"]
    assert len(host.sleeps) == 1
    assert headers(stream)["x-litellm-attempted-retries"] == 1


async def test_a_stream_that_fails_on_its_first_chunk_falls_back_after_its_retries(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream("x", error=rate_limit(), fail_after=0))
    host.answer("openai/b", chat_stream("B"))
    core = build(make_core, num_retries=2)
    stream = await open_stream(core)
    assert await collect_texts(stream) == "B"
    assert host.deployments_called() == ["a", "a", "a", "b"]
    assert len(host.sleeps) == 3
    assert model_id(stream) == "b"
    assert headers(stream)["x-litellm-attempted-fallbacks"] == 1


async def test_a_first_chunk_failure_with_nowhere_to_go_raises_the_provider_error_from_the_call(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream("x", error=rate_limit(message="first chunk"), fail_after=0))
    core = make_core(model_list=[deployment("a", "a")], num_retries=0)
    with pytest.raises(litellm.RateLimitError, match="first chunk"):
        await open_stream(core)


async def test_a_first_chunk_bad_request_is_raised_from_the_call_without_retry(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream("x", error=bad_request(message="nope"), fail_after=0))
    core = make_core(model_list=GROUPS_AB, num_retries=2)
    with pytest.raises(litellm.BadRequestError, match="nope"):
        await open_stream(core)
    assert host.deployments_called() == ["a"]


async def test_the_first_chunk_already_read_is_still_delivered_to_the_caller(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream("first", "second", "third"))
    core = make_core(model_list=[deployment("a", "a")])
    stream = await open_stream(core)
    assert await collect_texts(stream) == "firstsecondthird"


async def test_a_stream_that_ends_without_any_chunk_or_finish_reason_ends_quietly_without_text(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream(finish=False))
    host.answer("openai/b", chat_stream("B"))
    core = build(make_core)
    stream = await open_stream(core)
    assert await collect_texts(stream) == ""
    assert host.deployments_called() == ["a"]


async def test_a_stream_that_ends_after_content_without_a_finish_reason_raises_while_iterating(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream("partial", finish=False))
    core = build(make_core)
    stream = await open_stream(core)
    with pytest.raises(litellm.APIConnectionError, match="Stream ended before a finish_reason was received"):
        await collect_texts(stream)
    assert host.deployments_called() == ["a"]


# --- mid-stream errors -------------------------------------------------------------------------


async def test_a_mid_stream_error_after_content_raises_the_provider_error_not_mid_stream_fallback_error(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream("he", "llo", error=rate_limit(message="cut after content")))
    host.answer("openai/b", chat_stream("B"))
    core = build(make_core)
    stream = await open_stream(core)
    with pytest.raises(litellm.RateLimitError, match="cut after content") as raised:
        await collect_texts(stream)
    assert not isinstance(raised.value, MidStreamFallbackError)
    assert host.deployments_called() == ["a"]


async def test_a_mid_stream_error_after_content_delivers_the_text_before_it(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream("he", "llo", error=rate_limit()))
    core = build(make_core)
    stream = await open_stream(core)
    received: list[object] = []
    with pytest.raises(litellm.RateLimitError):
        await drain_into(stream, received)
    assert "".join(getattr(chunk, "choices")[0].delta.content or "" for chunk in received) == "hello"


async def test_a_mid_stream_error_before_content_falls_back_to_the_next_group(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream(error=rate_limit(), leading_role_chunk=True))
    host.answer("openai/b", chat_stream("B"))
    core = build(make_core)
    stream = await open_stream(core)
    assert await collect_texts(stream) == "B"
    assert host.deployments_called() == ["a", "b"]


async def test_a_mid_stream_error_before_content_does_not_retry_the_same_group(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream(error=rate_limit(), leading_role_chunk=True))
    host.answer("openai/b", chat_stream("B"))
    core = build(make_core, num_retries=2)
    stream = await open_stream(core)
    assert await collect_texts(stream) == "B"
    assert host.deployments_called() == ["a", "b"]
    assert host.sleeps == []


async def test_a_mid_stream_fallback_sends_the_original_messages_to_the_fallback(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream(error=rate_limit(), leading_role_chunk=True))
    host.answer("openai/b", chat_stream("B"))
    core = build(make_core)
    stream = await open_stream(core)
    await collect_texts(stream)
    fallback_call = host.calls_to("acompletion")[1]
    assert fallback_call.kwargs["messages"] == MESSAGES
    assert fallback_call.model_group == "b"
    assert fallback_call.kwargs["stream"] is True


async def test_a_mid_stream_error_before_content_without_fallbacks_raises_the_provider_error(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream(error=rate_limit(message="no fallback"), leading_role_chunk=True))
    core = make_core(model_list=GROUPS_AB, num_retries=0)
    stream = await open_stream(core)
    with pytest.raises(litellm.RateLimitError, match="no fallback") as raised:
        await collect_texts(stream)
    assert not isinstance(raised.value, MidStreamFallbackError)
    assert host.deployments_called() == ["a"]


async def test_disable_fallbacks_on_the_request_raises_a_mid_stream_error_before_content(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream(error=rate_limit(message="disabled"), leading_role_chunk=True))
    host.answer("openai/b", chat_stream("B"))
    core = build(make_core)
    stream = await open_stream(core, disable_fallbacks=True)
    with pytest.raises(litellm.RateLimitError, match="disabled"):
        await collect_texts(stream)
    assert host.deployments_called() == ["a"]


async def test_a_per_request_fallback_list_is_used_for_a_mid_stream_fallback(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream(error=rate_limit(), leading_role_chunk=True))
    host.answer("openai/c", chat_stream("C"))
    core = build(make_core, model_list=[*GROUPS_AB, deployment("c", "c")])
    stream = await open_stream(core, fallbacks=["c"])
    assert await collect_texts(stream) == "C"
    assert host.deployments_called() == ["a", "c"]


async def test_a_mid_stream_fallback_walks_the_chain_until_a_group_answers(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream(error=rate_limit(), leading_role_chunk=True))
    host.answer("openai/b", server_error())
    host.answer("openai/c", chat_stream("C"))
    core = build(make_core, model_list=[*GROUPS_AB, deployment("c", "c")], fallbacks=[{"a": ["b", "c"]}])
    stream = await open_stream(core)
    assert await collect_texts(stream) == "C"
    assert host.deployments_called() == ["a", "b", "c"]


async def test_when_the_mid_stream_fallback_also_fails_the_primary_provider_error_is_raised(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream(error=rate_limit(message="primary"), leading_role_chunk=True))
    host.answer("openai/b", server_error())
    core = build(make_core)
    stream = await open_stream(core)
    with pytest.raises(litellm.RateLimitError, match="primary"):
        await collect_texts(stream)
    assert host.deployments_called() == ["a", "b"]


async def test_a_mid_stream_server_error_before_content_falls_through_order_tiers(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a1", chat_stream(error=server_error(), leading_role_chunk=True))
    host.answer("openai/a2", chat_stream("A2"))
    core = make_core(model_list=[deployment("a", "a1", order=1), deployment("a", "a2", order=2)], num_retries=0)
    stream = await open_stream(core)
    assert await collect_texts(stream) == "A2"
    assert host.deployments_called() == ["a1", "a2"]
    assert model_id(stream) == "a2"
    assert headers(stream)["x-litellm-model-group"] == "a"


async def test_a_mid_stream_bad_request_before_content_is_raised_without_fallback(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream(error=bad_request(message="client error"), leading_role_chunk=True))
    host.answer("openai/b", chat_stream("B"))
    core = build(make_core)
    stream = await open_stream(core)
    with pytest.raises(litellm.BadRequestError, match="client error"):
        await collect_texts(stream)
    assert host.deployments_called() == ["a"]


async def test_a_mid_stream_content_policy_error_before_content_is_raised_without_fallback(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream(error=content_policy(), leading_role_chunk=True))
    host.answer("openai/b", chat_stream("B"))
    core = build(make_core, fallbacks=None, content_policy_fallbacks=[{"a": ["b"]}])
    stream = await open_stream(core)
    with pytest.raises(litellm.ContentPolicyViolationError):
        await collect_texts(stream)
    assert host.deployments_called() == ["a"]


async def test_a_mid_stream_context_window_error_before_content_uses_the_context_window_chain(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream(error=context_window(), leading_role_chunk=True))
    host.answer("openai/b", chat_stream("B"))
    core = build(make_core, fallbacks=None, context_window_fallbacks=[{"a": ["b"]}])
    stream = await open_stream(core)
    assert await collect_texts(stream) == "B"
    assert host.deployments_called() == ["a", "b"]


async def test_a_mid_stream_context_window_error_after_content_is_raised(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream("x", error=context_window()))
    host.answer("openai/b", chat_stream("B"))
    core = build(make_core, context_window_fallbacks=[{"a": ["b"]}])
    stream = await open_stream(core)
    with pytest.raises(litellm.ContextWindowExceededError):
        await collect_texts(stream)
    assert host.deployments_called() == ["a"]


# --- hidden params on a streamed response -----------------------------------------------------


async def test_a_stream_without_fallback_reports_the_serving_deployment(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream("x"))
    core = build(make_core)
    stream = await open_stream(core)
    await collect_texts(stream)
    assert model_id(stream) == "a"
    assert headers(stream)["x-litellm-model-group"] == "a"
    assert headers(stream)["x-litellm-attempted-fallbacks"] == 0


async def test_after_a_mid_stream_fallback_the_stream_reports_the_fallback_deployment(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream(error=rate_limit(), leading_role_chunk=True))
    host.answer("openai/b", chat_stream("B"))
    core = build(make_core)
    stream = await open_stream(core)
    await collect_texts(stream)
    assert model_id(stream) == "b"
    assert headers(stream) == {
        "x-litellm-model-group": "b",
        "x-litellm-attempted-retries": 0,
        "x-litellm-attempted-fallbacks": 1,
    }


async def test_after_a_mid_stream_fallback_each_fallback_chunk_carries_the_fallback_deployment(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream(error=rate_limit(), leading_role_chunk=True))
    host.answer("openai/b", chat_stream("B"))
    core = build(make_core)
    stream = await open_stream(core)
    chunks = [chunk async for chunk in stream]  # pyright: ignore[reportGeneralTypeIssues]  # an async stream
    fallback_chunks = chunks[1:]
    assert fallback_chunks
    assert all(chunk._hidden_params.get("model_id") == "b" for chunk in fallback_chunks)
    assert fallback_chunks[-1]._hidden_params["additional_headers"]["x-litellm-model-group"] == "b"


async def test_when_the_mid_stream_fallback_fails_the_stream_still_reports_the_primary(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream(error=rate_limit(), leading_role_chunk=True))
    host.answer("openai/b", server_error())
    core = build(make_core)
    stream = await open_stream(core)
    with pytest.raises(litellm.RateLimitError):
        await collect_texts(stream)
    assert model_id(stream) == "a"
    assert headers(stream)["x-litellm-model-group"] == "a"


# --- mid-stream continuation -------------------------------------------------------------------


def continuation_core(make_core: MakeCore, **arguments: object) -> RouterCore:  # kwargs-ok: constructor arguments
    return build(
        make_core,
        model_list=[deployment("a", "a"), deployment("b", "b", model=PREFILL_MODEL)],
        enable_mid_stream_fallback_continuation=True,
        **arguments,
    )


async def test_continuation_off_by_default_raises_after_content(make_core: MakeCore, host: ScriptedHost) -> None:
    host.answer("openai/a", chat_stream("he", "llo", error=rate_limit(message="cut")))
    host.answer(PREFILL_MODEL, chat_stream(" world"))
    core = build(make_core, model_list=[deployment("a", "a"), deployment("b", "b", model=PREFILL_MODEL)])
    stream = await open_stream(core)
    with pytest.raises(litellm.RateLimitError, match="cut"):
        await collect_texts(stream)
    assert host.deployments_called() == ["a"]


async def test_continuation_continues_after_content_on_a_prefill_capable_fallback(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream("he", "llo", error=rate_limit()))
    host.answer(PREFILL_MODEL, chat_stream(" world"))
    core = continuation_core(make_core)
    stream = await open_stream(core)
    assert await collect_texts(stream) == "hello world"
    assert host.deployments_called() == ["a", "b"]


async def test_continuation_sends_the_partial_text_as_an_assistant_prefill(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream("he", "llo", error=rate_limit()))
    host.answer(PREFILL_MODEL, chat_stream(" world"))
    core = continuation_core(make_core)
    stream = await open_stream(core)
    await collect_texts(stream)
    assert host.calls[1].kwargs["messages"] == [*MESSAGES, {"role": "assistant", "content": "hello", "prefix": True}]


async def test_continuation_extends_a_trailing_prefill_turn_in_place(make_core: MakeCore, host: ScriptedHost) -> None:
    host.answer("openai/a", chat_stream("Once", " upon", error=rate_limit()))
    host.answer(PREFILL_MODEL, chat_stream(" a time"))
    core = continuation_core(make_core)
    messages = [*MESSAGES, {"role": "assistant", "content": "Once", "prefix": True}]
    stream = await core.acompletion(model="a", messages=messages, stream=True)  # pyright: ignore[reportAttributeAccessIssue]  # endpoint method
    await collect_texts(stream)
    sent = host.calls[1].kwargs["messages"]
    assert sent[-1] == {"role": "assistant", "content": "Once upon", "prefix": True}
    assert len(sent) == 2


async def test_continuation_declines_when_the_request_ends_in_a_plain_assistant_turn(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream("x", error=rate_limit(message="cut")))
    host.answer(PREFILL_MODEL, chat_stream("y"))
    core = continuation_core(make_core)
    messages = [*MESSAGES, {"role": "assistant", "content": "earlier"}]
    stream = await core.acompletion(model="a", messages=messages, stream=True)  # pyright: ignore[reportAttributeAccessIssue]  # endpoint method
    with pytest.raises(litellm.RateLimitError, match="cut"):
        await collect_texts(stream)
    assert host.deployments_called() == ["a"]


async def test_continuation_reduces_the_output_ceiling_by_the_tokens_already_emitted(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream("hello", " there", error=rate_limit()))
    host.answer(PREFILL_MODEL, chat_stream(" friend"))
    core = continuation_core(make_core)
    stream = await open_stream(core, max_tokens=50)
    await collect_texts(stream)
    assert [call.kwargs["max_tokens"] for call in host.calls] == [50, 48]


async def test_continuation_raises_the_provider_error_when_the_output_ceiling_is_spent(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream("hello", " there", error=rate_limit(message="spent")))
    host.answer(PREFILL_MODEL, chat_stream(" friend"))
    core = continuation_core(make_core)
    stream = await open_stream(core, max_tokens=2)
    with pytest.raises(litellm.RateLimitError, match="spent"):
        await collect_texts(stream)
    assert host.deployments_called() == ["a"]


async def test_continuation_never_lands_on_a_fallback_that_cannot_take_a_prefill(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream("he", "llo", error=rate_limit(message="no prefill target")))
    host.answer("openai/b", chat_stream("B"))
    core = build(make_core, enable_mid_stream_fallback_continuation=True)
    stream = await open_stream(core)
    with pytest.raises(litellm.RateLimitError, match="no prefill target"):
        await collect_texts(stream)
    assert host.deployments_called() == ["a"]


@pytest.mark.parametrize(
    "request_kwargs",
    [
        {"response_format": {"type": "json_object"}},
        {"tool_choice": "required", "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}]},
    ],
    ids=["json-response-format", "forced-tool-choice"],
)
async def test_continuation_declines_a_constrained_request(
    make_core: MakeCore, host: ScriptedHost, request_kwargs: dict[str, object]
) -> None:
    host.answer("openai/a", chat_stream("he", "llo", error=rate_limit(message="constrained")))
    host.answer(PREFILL_MODEL, chat_stream(" world"))
    core = continuation_core(make_core)
    stream = await open_stream(core, **request_kwargs)
    with pytest.raises(litellm.RateLimitError, match="constrained"):
        await collect_texts(stream)
    assert host.deployments_called() == ["a"]


async def test_continuation_declines_a_fallback_target_that_overrides_request_params(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream("he", "llo", error=rate_limit(message="override")))
    host.answer(PREFILL_MODEL, chat_stream(" world"))
    core = continuation_core(make_core, fallbacks=[{"a": [{"model": "b", "temperature": 0.1}]}])
    stream = await open_stream(core)
    with pytest.raises(litellm.RateLimitError, match="override"):
        await collect_texts(stream)
    assert host.deployments_called() == ["a"]


async def test_continuation_runs_a_plain_fallback_for_an_error_before_content(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream(error=rate_limit(), leading_role_chunk=True))
    host.answer(PREFILL_MODEL, chat_stream("B"))
    core = continuation_core(make_core)
    stream = await open_stream(core)
    assert await collect_texts(stream) == "B"
    assert host.calls[1].kwargs["messages"] == MESSAGES


# --- the deployment slot held by an open stream -----------------------------------------------


async def test_an_open_stream_holds_its_max_parallel_requests_slot_until_consumed(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream("x"))
    core = make_core(model_list=[deployment("a", "a", max_parallel_requests=1)], num_retries=0)
    first = await open_stream(core)
    with pytest.raises(litellm.RateLimitError):
        await asyncio.wait_for(open_stream(core), 5)
    await collect_texts(first)
    second = await asyncio.wait_for(open_stream(core), 5)
    assert await collect_texts(second) == "x"


# --- the sync twin -----------------------------------------------------------------------------


def test_sync_stream_falls_back_on_a_mid_stream_error_before_content(make_core: MakeCore, host: ScriptedHost) -> None:
    host.answer("openai/a", chat_stream(error=rate_limit(), leading_role_chunk=True, sync=True))
    host.answer("openai/b", chat_stream("B", sync=True))
    core = build(make_core)
    stream = core.completion(model="a", messages=MESSAGES, stream=True)  # pyright: ignore[reportAttributeAccessIssue]  # endpoint method
    assert collect_texts_sync(stream) == "B"
    assert [(call.op, call.deployment_id) for call in host.calls] == [("completion", "a"), ("completion", "b")]
    assert model_id(stream) == "b"


def test_sync_stream_raises_the_provider_error_after_content(make_core: MakeCore, host: ScriptedHost) -> None:
    host.answer("openai/a", chat_stream("x", error=rate_limit(message="sync cut"), sync=True))
    host.answer("openai/b", chat_stream("B", sync=True))
    core = build(make_core)
    stream = core.completion(model="a", messages=MESSAGES, stream=True)  # pyright: ignore[reportAttributeAccessIssue]  # endpoint method
    with pytest.raises(litellm.RateLimitError, match="sync cut"):
        collect_texts_sync(stream)
    assert host.deployments_called() == ["a"]


def test_sync_stream_never_continues_after_content_even_with_continuation_on(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream("he", "llo", error=rate_limit(message="sync no continuation"), sync=True))
    host.answer(PREFILL_MODEL, chat_stream(" world", sync=True))
    core = continuation_core(make_core)
    stream = core.completion(model="a", messages=MESSAGES, stream=True)  # pyright: ignore[reportAttributeAccessIssue]  # endpoint method
    with pytest.raises(litellm.RateLimitError, match="sync no continuation"):
        collect_texts_sync(stream)
    assert host.deployments_called() == ["a"]


def test_sync_stream_disable_fallbacks_raises_a_mid_stream_error_before_content(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream(error=rate_limit(message="sync disabled"), leading_role_chunk=True, sync=True))
    host.answer("openai/b", chat_stream("B", sync=True))
    core = build(make_core)
    stream = core.completion(model="a", messages=MESSAGES, stream=True, disable_fallbacks=True)  # pyright: ignore[reportAttributeAccessIssue]  # endpoint method
    with pytest.raises(litellm.RateLimitError, match="sync disabled"):
        collect_texts_sync(stream)
    assert host.deployments_called() == ["a"]


def test_sync_stream_first_chunk_failure_falls_back_inside_the_call(make_core: MakeCore, host: ScriptedHost) -> None:
    host.answer("openai/a", chat_stream("x", error=rate_limit(), fail_after=0, sync=True))
    host.answer("openai/b", chat_stream("B", sync=True))
    core = build(make_core)
    stream = core.completion(model="a", messages=MESSAGES, stream=True)  # pyright: ignore[reportAttributeAccessIssue]  # endpoint method
    assert collect_texts_sync(stream) == "B"
    assert host.deployments_called() == ["a", "b"]


def test_sync_stream_when_the_fallback_fails_raises_the_primary_error(make_core: MakeCore, host: ScriptedHost) -> None:
    host.answer("openai/a", chat_stream(error=rate_limit(message="sync primary"), leading_role_chunk=True, sync=True))
    host.answer("openai/b", server_error())
    core = build(make_core)
    stream = core.completion(model="a", messages=MESSAGES, stream=True)  # pyright: ignore[reportAttributeAccessIssue]  # endpoint method
    with pytest.raises(litellm.RateLimitError, match="sync primary"):
        collect_texts_sync(stream)
    assert host.deployments_called() == ["a", "b"]


# --- the stream timeout sent to the host -------------------------------------------------------


@pytest.mark.parametrize(
    ("router_arguments", "deployment_params", "request_kwargs", "expected"),
    [
        ({"stream_timeout": 4}, {}, {}, 4),
        ({"stream_timeout": 4}, {"stream_timeout": 3}, {}, 3.0),
        ({"stream_timeout": 4}, {"stream_timeout": 3}, {"stream_timeout": 2}, 2),
        ({}, {"stream_timeout_by_service_tier": {"flex": 9}, "stream_timeout": 3}, {"service_tier": "flex"}, 9.0),
        ({}, {"stream_timeout_by_service_tier": {"flex": 9}, "stream_timeout": 3}, {"service_tier": "auto"}, 3.0),
        (
            {},
            {"stream_timeout_by_service_tier": {"flex": 9}},
            {"service_tier": "flex", "stream_timeout": 2},
            2,
        ),
        ({"default_litellm_params": {"stream_timeout": 8}}, {}, {}, 8),
    ],
    ids=[
        "router",
        "deployment-over-router",
        "request-over-deployment",
        "service-tier-over-deployment",
        "unmatched-service-tier-uses-deployment",
        "request-over-service-tier",
        "default-litellm-params",
    ],
)
async def test_the_stream_timeout_comes_from_request_then_tier_then_deployment_then_router(
    make_core: MakeCore,
    host: ScriptedHost,
    router_arguments: dict[str, object],
    deployment_params: dict[str, object],
    request_kwargs: dict[str, object],
    expected: float,
) -> None:
    host.answer("openai/a", chat_stream("x"))
    core = make_core(model_list=[deployment("a", "a", **deployment_params)], **router_arguments)
    stream = await open_stream(core, **request_kwargs)
    await collect_texts(stream)
    assert host.calls[0].kwargs["stream_timeout"] == expected


@pytest.mark.parametrize("router_arguments", [{}, {"timeout": 42}], ids=["no-router-timeout", "router-timeout"])
async def test_without_any_stream_timeout_none_is_sent_and_the_plain_timeout_applies(
    make_core: MakeCore, host: ScriptedHost, router_arguments: dict[str, object]
) -> None:
    host.answer("openai/a", chat_stream("x"))
    core = make_core(model_list=[deployment("a", "a")], **router_arguments)
    stream = await open_stream(core)
    await collect_texts(stream)
    sent = host.calls[0].kwargs
    assert "stream_timeout" not in sent
    assert sent["timeout"] == router_arguments.get("timeout", 6000.0)


async def test_a_non_streaming_request_sends_no_stream_timeout(make_core: MakeCore, host: ScriptedHost) -> None:
    core = make_core(model_list=[deployment("a", "a")], stream_timeout=4)
    await core.acompletion(model="a", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]  # endpoint method
    assert "stream_timeout" not in host.calls[0].kwargs
