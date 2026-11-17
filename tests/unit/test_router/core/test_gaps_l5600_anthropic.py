"""
Gap characterization for /v1/messages streaming recovery (``router_python_core.py`` 7106-7455): a
``MidStreamFallbackError`` raised by the source stream itself, a same-group stream retry whose own
attempt raises before opening, and a request-level retry policy that is not a ``RetryPolicy``.
"""

from collections.abc import Callable
from typing import Final

import pytest

import litellm
from litellm.exceptions import MidStreamFallbackError
from litellm.router_core import RouterCore
from tests.unit.test_router.core.scripted_host import ScriptedHost, deployment, rate_limit, server_error
from tests.unit.test_router.core.streaming_helpers import (
    MESSAGE_START,
    MESSAGE_STOP,
    MESSAGES,
    anthropic_stream,
    collect_bytes,
    drain_into,
    error_frame,
    install_orjson_stub_if_missing,
    text_delta,
)

install_orjson_stub_if_missing()

MakeCore = Callable[..., RouterCore]
GROUPS_AB: Final = [deployment("a", "a"), deployment("b", "b")]


async def _open(core: RouterCore, **request: object) -> object:  # kwargs-ok: request kwargs
    return await core.aanthropic_messages(model="a", messages=MESSAGES, max_tokens=5, stream=True, **request)  # pyright: ignore[reportAttributeAccessIssue]  # endpoint method


def _mid_stream(*, pre_first_chunk: bool) -> MidStreamFallbackError:
    return MidStreamFallbackError(
        message="bridge dropped",
        model="a",
        llm_provider="openai",
        original_exception=rate_limit("a", "bridge original"),
        is_pre_first_chunk=pre_first_chunk,
    )


async def test_a_raised_mid_stream_error_not_before_the_first_chunk_flushes_held_frames_and_raises_the_original(
    make_core: MakeCore, host: ScriptedHost
):
    host.answer("openai/a", anthropic_stream(MESSAGE_START, error=_mid_stream(pre_first_chunk=False)))
    host.answer("openai/b", anthropic_stream(MESSAGE_START, text_delta("B"), MESSAGE_STOP))
    core: Final = make_core(model_list=GROUPS_AB, fallbacks=[{"a": ["b"]}], num_retries=0)
    stream: Final = await _open(core)
    seen: Final[list[object]] = []
    with pytest.raises(litellm.RateLimitError, match="bridge original"):
        await drain_into(stream, seen)
    assert seen == [MESSAGE_START]
    assert host.deployments_called() == ["a"]


async def test_a_raised_mid_stream_error_before_the_first_chunk_falls_back(make_core: MakeCore, host: ScriptedHost):
    host.answer("openai/a", anthropic_stream(MESSAGE_START, error=_mid_stream(pre_first_chunk=True)))
    host.answer("openai/b", anthropic_stream(MESSAGE_START, text_delta("B"), MESSAGE_STOP))
    core: Final = make_core(model_list=GROUPS_AB, fallbacks=[{"a": ["b"]}], num_retries=0)
    frames: Final = await collect_bytes(await _open(core))
    assert frames == [MESSAGE_START, text_delta("B"), MESSAGE_STOP]
    assert host.deployments_called() == ["a", "b"]


async def test_a_stream_retry_that_raises_before_opening_spends_a_retry_and_the_next_one_serves(
    make_core: MakeCore, host: ScriptedHost
):
    host.answer(
        "openai/a",
        anthropic_stream(MESSAGE_START, error_frame()),
        server_error("a"),
        anthropic_stream(MESSAGE_START, text_delta("A"), MESSAGE_STOP),
    )
    core: Final = make_core(model_list=[deployment("a", "a")], num_retries=2)
    stream: Final = await _open(core)
    frames: Final = await collect_bytes(stream)
    assert frames == [MESSAGE_START, text_delta("A"), MESSAGE_STOP]
    assert host.deployments_called() == ["a", "a", "a"]
    assert len(host.sleeps) == 2
    headers: Final = getattr(stream, "_hidden_params")["additional_headers"]
    assert (headers["x-litellm-attempted-retries"], headers["x-litellm-max-retries"]) == (2, 2)


async def test_a_request_retry_policy_that_is_not_a_retry_policy_streams_with_the_plain_budget(
    make_core: MakeCore, host: ScriptedHost
):
    host.answer(
        "openai/a",
        anthropic_stream(MESSAGE_START, error_frame("rate_limit_error", "slow")),
        anthropic_stream(MESSAGE_START, text_delta("A"), MESSAGE_STOP),
    )
    core: Final = make_core(model_list=[deployment("a", "a")], num_retries=1)
    stream: Final = await _open(core, model_group_retry_policy={"a": 5})
    frames: Final = await collect_bytes(stream)
    assert frames == [MESSAGE_START, text_delta("A"), MESSAGE_STOP]
    assert host.deployments_called() == ["a", "a"]
    assert getattr(stream, "_hidden_params")["additional_headers"]["x-litellm-max-retries"] == 1
