"""
Parity of the Rust core with ``router_python_core.py`` on the stream causes of the regress5 pack:
request-level fallback lists on a ``/v1/messages`` stream, what a stream's failure records, a
malformed group retry policy, the error a failed recovery raises, and the hint on a stream's failure
event.
"""

import asyncio
from collections.abc import Callable, Mapping
from typing import Final

import pytest
from pydantic import ValidationError

import litellm
from litellm.llms.anthropic.messages.adapters.streaming_iterator import AnthropicSSEStream, AnthropicStreamWrapper
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.router_core import RouterCore
from litellm.types.router import RetryPolicy
from tests.unit.test_router.core.scripted_host import (
    ScriptedHost,
    auth_error,
    content_policy,
    context_window,
    deployment,
    rate_limit,
    server_error,
)
from tests.unit.test_router.core.streaming_helpers import (
    MESSAGE_START,
    MESSAGE_STOP,
    MESSAGES,
    anthropic_stream,
    chat_stream,
    collect_bytes,
    collect_texts,
    collect_texts_sync,
    error_frame,
    install_orjson_stub_if_missing,
    refusal_delta,
    text_delta,
)

install_orjson_stub_if_missing()

MakeCore = Callable[..., RouterCore]
GROUPS_AB: Final = [deployment("a", "a"), deployment("b", "b")]
PREFILL_MODEL: Final = "anthropic/claude-sonnet-4-5"
FALLBACK_FRAMES: Final = [MESSAGE_START, text_delta("B"), MESSAGE_STOP]
REFUSAL_DETAILS: Final = {"type": "refusal", "category": "cyber"}
PER_USER_COPILOT: Final = {
    "model": "github_copilot/gpt-4o",
    "litellm_credential_name": "ghc",
    "github_copilot_auth_type": "per_user_oauth",
}


async def open_messages(core: RouterCore, **request: object) -> object:  # kwargs-ok: request kwargs
    return await core.aanthropic_messages(model="a", messages=MESSAGES, max_tokens=5, stream=True, **request)  # pyright: ignore[reportAttributeAccessIssue]  # endpoint method


async def settle() -> None:
    """Let the logging and recording tasks a failed stream scheduled run."""
    for _ in range(20):
        await asyncio.sleep(0)


def cooled(core: RouterCore, *ids: str) -> Mapping[str, Mapping[str, object]]:
    return dict(core.health().cooldowns.get_active_cooldowns(list(ids), None))  # pyright: ignore[reportReturnType]  # a mapping of entries


class BridgedStream(AnthropicSSEStream):
    """The byte stream the completion bridge returns for ``/v1/messages``: a chat stream's failure is
    logged by that stream and reaches the caller as an ``event: error`` frame."""

    def __init__(self, stream: CustomStreamWrapper) -> None:
        super().__init__(
            AnthropicStreamWrapper(completion_stream=stream, model="d1", litellm_logging_obj=stream.logging_obj)
        )
        self.logging_obj: Final = stream.logging_obj


def bridged(error: Exception) -> Callable[..., BridgedStream]:
    chat: Final = chat_stream(error=error, fail_after=0, logged=True)

    def answer(**request: object) -> BridgedStream:  # kwargs-ok: the host passes the request kwargs
        return BridgedStream(chat(**request))

    return answer


# --- T1: the request's fallback lists on a /v1/messages stream -------------------------------------


@pytest.mark.parametrize("fallbacks", [["b"], [{"a": ["b"]}], [{"*": ["b"]}]], ids=["names", "keyed", "star"])
async def test_request_fallbacks_rescue_a_messages_stream_before_content(
    make_core: MakeCore, host: ScriptedHost, fallbacks: list[object]
) -> None:
    host.answer("openai/a", anthropic_stream(MESSAGE_START, error_frame()))
    host.answer("openai/b", anthropic_stream(*FALLBACK_FRAMES))
    core: Final = make_core(model_list=GROUPS_AB, num_retries=0)
    assert await collect_bytes(await open_messages(core, fallbacks=fallbacks)) == FALLBACK_FRAMES
    assert host.deployments_called() == ["a", "b"]


async def test_request_content_policy_fallbacks_rescue_a_refusal_before_content(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", anthropic_stream(MESSAGE_START, refusal_delta(REFUSAL_DETAILS)))
    host.answer("openai/b", anthropic_stream(*FALLBACK_FRAMES))
    core: Final = make_core(model_list=GROUPS_AB, num_retries=0)
    stream: Final = await open_messages(core, content_policy_fallbacks=[{"a": ["b"]}])
    assert await collect_bytes(stream) == FALLBACK_FRAMES
    assert host.deployments_called() == ["a", "b"]


async def test_a_request_without_fallbacks_replaces_the_router_list_on_a_messages_stream(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", anthropic_stream(MESSAGE_START, error_frame()))
    core: Final = make_core(model_list=GROUPS_AB, num_retries=0, fallbacks=[{"a": ["b"]}])
    assert await collect_bytes(await open_messages(core, fallbacks=None)) == [MESSAGE_START, error_frame()]
    assert host.deployments_called() == ["a"]


# --- T2: what a bridged stream's failure records --------------------------------------------------


async def test_a_bridged_401_before_content_cools_the_deployment_down(make_core: MakeCore, host: ScriptedHost) -> None:
    host.answer("openai/a", bridged(auth_error()))
    host.answer("openai/b", anthropic_stream(*FALLBACK_FRAMES))
    core: Final = make_core(model_list=GROUPS_AB, num_retries=0, fallbacks=[{"a": ["b"]}])
    assert await collect_bytes(await open_messages(core)) == FALLBACK_FRAMES
    await settle()
    assert sorted(cooled(core, "a", "b")) == ["a"]
    assert str(cooled(core, "a")["a"]["status_code"]) == "401"


async def test_a_bridged_rate_limit_cools_with_the_provider_error_not_the_frame(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", bridged(rate_limit(message="provider text")))
    host.answer("openai/b", anthropic_stream(*FALLBACK_FRAMES))
    core: Final = make_core(model_list=GROUPS_AB, num_retries=0, allowed_fails=0, fallbacks=[{"a": ["b"]}])
    assert await collect_bytes(await open_messages(core)) == FALLBACK_FRAMES
    await settle()
    received: Final = str(cooled(core, "a")["a"]["exception_received"])
    assert "provider text" in received
    assert "AnthropicException" not in received


async def test_a_caller_scoped_401_on_a_bridged_stream_never_cools_the_deployment(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer(
        "github_copilot/gpt-4o",
        bridged(
            litellm.CallerCredentialAuthenticationError(
                message="not connected", llm_provider="github_copilot", model="gpt-4o"
            )
        ),
    )
    host.answer("openai/b", anthropic_stream(*FALLBACK_FRAMES))
    core: Final = make_core(
        model_list=[deployment("a", "a", **PER_USER_COPILOT), deployment("b", "b")],
        num_retries=0,
        allowed_fails=0,
        fallbacks=[{"a": ["b"]}],
    )
    assert await collect_bytes(await open_messages(core)) == FALLBACK_FRAMES
    await settle()
    assert cooled(core, "a", "b") == {}


# --- T3: a malformed group retry policy on a messages stream ---------------------------------------


async def test_a_malformed_group_policy_raises_on_a_messages_stream_that_never_opened(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("*", rate_limit())
    core: Final = make_core(
        model_list=[deployment("a", "a1"), deployment("a", "a2")],
        num_retries=1,
        model_group_retry_policy={"a": {"RateLimitErrorRetries": "many"}},
    )
    with pytest.raises(ValidationError):
        await open_messages(core)
    assert len(host.deployments_called()) == 1


@pytest.mark.parametrize("retry_policy", [None, RetryPolicy(InternalServerErrorRetries=0)], ids=["alone", "router"])
async def test_a_malformed_group_policy_governs_nothing_on_a_stream_that_opened(
    make_core: MakeCore, host: ScriptedHost, retry_policy: RetryPolicy | None
) -> None:
    host.answer("openai/a", anthropic_stream(MESSAGE_START, error=server_error()), anthropic_stream(*FALLBACK_FRAMES))
    core: Final = make_core(
        model_list=[deployment("a", "a")],
        num_retries=1,
        retry_policy=retry_policy,
        model_group_retry_policy={"a": {"RateLimitErrorRetries": "many"}},
    )
    assert await collect_bytes(await open_messages(core)) == FALLBACK_FRAMES
    assert host.deployments_called() == ["a", "a"]


# --- S8: the error a failed recovery raises --------------------------------------------------------


async def test_a_continuation_whose_fallback_stream_fails_raises_the_fallback_error(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream("he", error=rate_limit(message="primary")))
    host.answer(PREFILL_MODEL, chat_stream(" wor", "ld", error=server_error(), fail_after=1))
    core: Final = make_core(
        model_list=[deployment("a", "a"), deployment("b", "b", model=PREFILL_MODEL)],
        num_retries=0,
        fallbacks=[{"a": ["b"]}],
        enable_mid_stream_fallback_continuation=True,
    )
    stream: Final = await core.acompletion(model="a", messages=MESSAGES, stream=True)  # pyright: ignore[reportAttributeAccessIssue]  # endpoint method
    with pytest.raises(litellm.InternalServerError, match="boom"):
        await collect_texts(stream)
    assert host.deployments_called() == ["a", "b"]


async def test_a_fallback_stream_that_fails_raises_its_own_error(make_core: MakeCore, host: ScriptedHost) -> None:
    host.answer("openai/a", chat_stream(error=rate_limit(message="primary"), leading_role_chunk=True))
    host.answer("openai/b", chat_stream("B", error=server_error()))
    core: Final = make_core(model_list=GROUPS_AB, num_retries=0, fallbacks=[{"a": ["b"]}])
    stream: Final = await core.acompletion(model="a", messages=MESSAGES, stream=True)  # pyright: ignore[reportAttributeAccessIssue]  # endpoint method
    with pytest.raises(litellm.InternalServerError, match="boom"):
        await collect_texts(stream)


def test_a_sync_fallback_stream_that_fails_raises_its_own_error(make_core: MakeCore, host: ScriptedHost) -> None:
    host.answer("openai/a", chat_stream(error=rate_limit(message="primary"), leading_role_chunk=True, sync=True))
    host.answer("openai/b", chat_stream("B", error=server_error(), sync=True))
    core: Final = make_core(model_list=GROUPS_AB, num_retries=0, fallbacks=[{"a": ["b"]}])
    stream: Final = core.completion(model="a", messages=MESSAGES, stream=True)  # pyright: ignore[reportAttributeAccessIssue]  # endpoint method
    with pytest.raises(litellm.InternalServerError, match="boom"):
        collect_texts_sync(stream)


async def test_a_messages_fallback_stream_that_fails_raises_its_own_error(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", anthropic_stream(MESSAGE_START, error_frame()))
    host.answer("openai/b", anthropic_stream(MESSAGE_START, text_delta("B"), error=server_error()))
    core: Final = make_core(model_list=GROUPS_AB, num_retries=0, fallbacks=[{"a": ["b"]}])
    stream: Final = await open_messages(core)
    with pytest.raises(litellm.InternalServerError, match="boom"):
        await collect_bytes(stream)


async def test_an_order_chain_that_serves_nothing_raises_its_last_hop_error_from_a_stream(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    """The order path of the fallback walk raises its last hop's error, not the failure it was handed."""
    host.answer("openai/a1", chat_stream(error=rate_limit(message="primary"), leading_role_chunk=True))
    host.answer("openai/a2", server_error())
    core: Final = make_core(model_list=[deployment("a", "a1", order=1), deployment("a", "a2", order=2)], num_retries=0)
    stream: Final = await core.acompletion(model="a", messages=MESSAGES, stream=True)  # pyright: ignore[reportAttributeAccessIssue]  # endpoint method
    with pytest.raises(litellm.InternalServerError, match="boom"):
        await collect_texts(stream)
    assert host.deployments_called() == ["a1", "a2"]


# --- V3: the hint on a stream's failure event ------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "raised", "hint"),
    [
        (context_window, litellm.ContextWindowExceededError, "Set 'context_window_fallback'"),
        (content_policy, litellm.ContentPolicyViolationError, "content_policy_fallback"),
    ],
    ids=["context_window", "content_policy"],
)
async def test_a_failure_event_of_a_stream_failing_before_its_first_chunk_carries_the_hint(
    make_core: MakeCore, host: ScriptedHost, error: Callable[[], Exception], raised: type[Exception], hint: str
) -> None:
    host.set_dynamic(expose_router_debug_in_errors=True)
    host.answer("openai/a", chat_stream(error=error(), fail_after=0, logged=True))
    host.answer("openai/b", server_error())
    core: Final = make_core(model_list=GROUPS_AB, num_retries=0, fallbacks=[{"a": ["b"]}])
    logged: Final[list[str]] = []

    def record(kwargs: Mapping[str, object], *_rest: object) -> None:
        logged.append(str(getattr(kwargs.get("exception"), "message", "")))

    host.callback_list("failure_callback").append(record)
    with pytest.raises(raised):
        await core.acompletion(model="a", messages=MESSAGES, stream=True)  # pyright: ignore[reportAttributeAccessIssue]  # endpoint method
    await settle()
    # The fallback's own failure may be logged first.
    assert any(hint in message for message in logged)
