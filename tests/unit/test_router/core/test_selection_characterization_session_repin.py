"""
Characterization of ``session_affinity_repin_on_failover``: a session pinned to a deployment that
fails is served by another deployment of the group, and a completed success there moves the
session's pin to it when the flag is on. With it off, the pin stays where it was.
"""

import asyncio
from collections.abc import Callable

import pytest

from litellm.router_core import RouterCore
from litellm.router_utils.pre_call_checks.deployment_affinity_check import DeploymentAffinityCheck
from tests.unit.test_router.core.scripted_host import ScriptedHost, completion_response, deployment, rate_limit
from tests.unit.test_router.core.streaming_helpers import (
    MESSAGES,
    chat_stream,
    collect_texts,
    install_orjson_stub_if_missing,
)

install_orjson_stub_if_missing()

MakeCore = Callable[..., RouterCore]

_SESSION = "session-1"
_CALLER = "caller-1"
_METADATA = {"session_id": _SESSION, "user_api_key_hash": _CALLER}


def _affinity_check(core: RouterCore) -> DeploymentAffinityCheck:
    return next(
        callback
        for callback in core.optional_callbacks or ()  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType, reportUnknownMemberType]
        if isinstance(callback, DeploymentAffinityCheck)
    )


async def _pinned(check: DeploymentAffinityCheck) -> object:
    key = DeploymentAffinityCheck.get_session_affinity_cache_key("g", _SESSION, _CALLER)
    pinned: object = await check.cache.async_get_cache(key)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # DualCache's reads are untyped
    return pinned  # pyright: ignore[reportUnknownVariableType]  # see above


async def _complete(core: RouterCore) -> object:
    response: object = await core.acompletion(  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]  # endpoint method
        model="g", messages=[{"role": "user", "content": "hi"}], metadata=dict(_METADATA)
    )
    return response  # pyright: ignore[reportUnknownVariableType]  # see above


@pytest.mark.asyncio
@pytest.mark.parametrize(("repin", "next_served_by"), [(True, "b"), (False, "a")])
async def test_a_failover_success_repins_the_session_only_with_repin_on_failover(
    make_core: MakeCore, host: ScriptedHost, repin: bool, next_served_by: str
) -> None:
    core = make_core(
        model_list=[deployment("g", "a"), deployment("g", "b")],
        optional_pre_call_checks=["session_affinity"],
        session_affinity_repin_on_failover=repin,
        num_retries=1,
    )
    check = _affinity_check(core)
    key = DeploymentAffinityCheck.get_session_affinity_cache_key("g", _SESSION, _CALLER)
    await check.cache.async_set_cache(key, {"model_id": "a"})  # pyright: ignore[reportUnknownMemberType]
    host.answer("openai/a", rate_limit(), completion_response())
    host.answer("openai/b", completion_response())

    await _complete(core)

    # The pinned deployment fails and cools down; the group's other deployment serves the retry.
    assert host.deployments_called("acompletion") == ["a", "b"]
    assert await _pinned(check) == {"model_id": "b" if repin else "a"}

    # Once the first deployment is healthy again, the session goes where it is pinned.
    core.health().cooldowns.cooldown_store.evict_local("a")
    await _complete(core)

    assert host.deployments_called("acompletion")[-1] == next_served_by


async def _settle() -> None:
    """Let the logging and recording tasks a finished stream scheduled run."""
    for _ in range(20):
        await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("repin", [True, False])
async def test_a_failover_stream_repins_the_session_when_it_ends(
    make_core: MakeCore, host: ScriptedHost, repin: bool
) -> None:
    core = make_core(
        model_list=[deployment("g", "a"), deployment("g", "b")],
        optional_pre_call_checks=["session_affinity"],
        session_affinity_repin_on_failover=repin,
        num_retries=1,
    )
    check = _affinity_check(core)
    key = DeploymentAffinityCheck.get_session_affinity_cache_key("g", _SESSION, _CALLER)
    await check.cache.async_set_cache(key, {"model_id": "a"})  # pyright: ignore[reportUnknownMemberType]
    host.answer("openai/a", rate_limit())
    host.answer(
        "openai/b",
        chat_stream("hello", usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}, logged=True),
    )

    stream: object = await core.acompletion(  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]  # endpoint method
        model="g", messages=MESSAGES, stream=True, metadata=dict(_METADATA)
    )
    await _settle()
    # The pin moves only once the stream has been read to its end.
    assert await _pinned(check) == {"model_id": "a"}

    assert await collect_texts(stream) == "hello"  # pyright: ignore[reportUnknownArgumentType]  # see above
    await _settle()

    assert host.deployments_called("acompletion") == ["a", "b"]
    assert await _pinned(check) == {"model_id": "b" if repin else "a"}
