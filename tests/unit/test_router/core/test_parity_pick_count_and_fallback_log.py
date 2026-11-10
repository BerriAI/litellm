"""
Parity of the Rust core with ``router_python_core.py`` on two traces a call leaves on the proxy:
the per-deployment pick count in the router cache (``_track_deployment_metrics``, ``_update_usage``)
and the error log of a level whose fallbacks served nothing (the ``except`` around the regular
fallbacks of ``async_function_with_fallbacks_common_utils``).
"""

import logging
from collections.abc import Generator
from typing import Final

import openai
import pytest

import litellm
from litellm._logging import verbose_router_logger
from litellm.router_api import ChangeSet
from litellm.router_core import RouterCore
from tests.unit.test_router.core.fallbacks_helpers import (
    TIERED_A,
    MakeCore,
    build,
    complete,
    complete_sync,
    show_router_debug,
)
from tests.unit.test_router.core.scripted_host import (
    ScriptedHost,
    auth_error,
    content_policy,
    deployment,
    embedding_response,
    rate_limit,
    server_error,
)

LOG_HEAD: Final = "litellm.router.py::async_function_with_fallbacks() - Error occurred while trying to do fallbacks - "
NO_COOLDOWNS: Final = "\nDebug Information:\nCooldown Deployments=[]"


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.ERROR)
        self.messages: Final[list[str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def router_errors() -> Generator[list[str], None, None]:
    """The router logger's error messages, in order."""
    records: Final = _Records()
    verbose_router_logger.addHandler(records)
    yield records.messages
    verbose_router_logger.removeHandler(records)


def _picks(core: RouterCore, deployment_id: object) -> object:
    return core.health().store.in_memory_cache.get_cache(deployment_id)


def _fallback_logs(messages: list[str]) -> list[str]:  # mutable-ok: a fresh list
    return [message for message in messages if message.startswith(LOG_HEAD)]


async def _fails(core: RouterCore, model: str = "a", **request: object) -> BaseException:  # kwargs-ok: request kwargs
    with pytest.raises(openai.APIStatusError) as raised:
        await complete(core, model, **request)
    return raised.value


# --- the pick count -------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_each_async_chat_pick_counts_in_the_router_cache(make_core: MakeCore) -> None:
    core: Final = build(make_core)

    _ = await complete(core)
    _ = await complete(core)

    assert _picks(core, "a") == 2


@pytest.mark.asyncio
async def test_a_pick_counts_before_its_attempt_fails_and_a_retry_picks_again(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", server_error())
    core: Final = build(make_core, num_retries=2, retry_after=0)

    _ = await _fails(core)

    assert _picks(core, "a") == 3


@pytest.mark.asyncio
async def test_every_hop_of_a_fallback_chain_counts_its_pick(make_core: MakeCore, host: ScriptedHost) -> None:
    host.answer("openai/a", server_error())
    core: Final = build(make_core, fallbacks=[{"a": ["b"]}])

    _ = await complete(core)

    assert _picks(core, "a") == 1
    assert _picks(core, "b") == 1
    assert _picks(core, "c") is None


def test_a_sync_chat_pick_does_not_count(make_core: MakeCore) -> None:
    core: Final = build(make_core)

    _ = complete_sync(core)

    assert _picks(core, "a") is None


@pytest.mark.asyncio
async def test_an_embedding_pick_does_not_count(make_core: MakeCore, host: ScriptedHost) -> None:
    host.answer("openai/a", embedding_response())
    core: Final = build(make_core)

    _ = await core.aembedding(model="a", input="hi")  # pyright: ignore[reportAttributeAccessIssue]  # endpoint method

    assert _picks(core, "a") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("assigned", "counted"),
    [("fallback-id", "fallback-id"), (7, 7), (None, None)],
    ids=["string-id", "other-id", "no-id"],
)
async def test_a_default_deployment_pick_counts_under_the_id_it_holds(
    make_core: MakeCore, assigned: object, counted: object
) -> None:
    core: Final = build(make_core)
    served: Final = deployment("*", "x", model="openai/x")
    _ = core.apply(ChangeSet(assign={"default_deployment": {**served, "model_info": {"id": assigned}}}))

    _ = await complete(core, "not-a-group")

    if counted is not None:
        assert _picks(core, counted) == 1
    assert _picks(core, "x") is None


# --- the fallback failure log ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_chain_that_fails_logs_every_level_innermost_first(
    make_core: MakeCore, host: ScriptedHost, router_errors: list[str]
) -> None:
    for name in ("a", "b", "c"):
        host.answer(f"openai/{name}", rate_limit(message=f"{name} down"))
    core: Final = build(make_core, fallbacks=[{"a": ["b"]}, {"b": ["c"]}], disable_cooldowns=True)

    _ = await _fails(core)

    # Each level logs what its `try` raised: `c` has no chain, so its own error, and `b`'s hop to it
    # raised that; `a`'s hop to `b` raised `b`'s own error.
    logged: Final = _fallback_logs(router_errors)
    assert ["c down" in logged[0], "c down" in logged[1], "b down" in logged[2], len(logged)] == [True, True, True, 3]
    assert all(entry.endswith(NO_COOLDOWNS) for entry in logged)


@pytest.mark.asyncio
async def test_a_failed_hop_is_logged_with_the_hops_error_not_the_levels(
    make_core: MakeCore, host: ScriptedHost, router_errors: list[str]
) -> None:
    host.answer("openai/a", server_error())
    host.answer("openai/b", auth_error())
    core: Final = build(make_core, fallbacks=[{"a": ["b"]}], disable_cooldowns=True)

    error: Final = await _fails(core)

    assert type(error).__name__ == "InternalServerError"
    logged: Final = _fallback_logs(router_errors)
    assert len(logged) == 2
    assert all("AuthenticationError" in entry for entry in logged)


@pytest.mark.asyncio
async def test_no_fallback_list_logs_nothing(make_core: MakeCore, host: ScriptedHost, router_errors: list[str]) -> None:
    host.answer("openai/a", server_error())
    core: Final = build(make_core)

    _ = await _fails(core)

    assert _fallback_logs(router_errors) == []


@pytest.mark.asyncio
async def test_a_fallback_list_that_does_not_name_the_group_logs_the_levels_error(
    make_core: MakeCore, host: ScriptedHost, router_errors: list[str]
) -> None:
    host.answer("openai/a", server_error())
    core: Final = build(make_core, fallbacks=[{"b": ["c"]}], disable_cooldowns=True)

    error: Final = await _fails(core)

    assert _fallback_logs(router_errors) == [f"{LOG_HEAD}{error}{NO_COOLDOWNS}"]


@pytest.mark.asyncio
async def test_the_missing_group_explanation_is_in_the_log_and_on_the_error_once(
    make_core: MakeCore, host: ScriptedHost, router_errors: list[str]
) -> None:
    show_router_debug(host)
    host.answer("openai/a", server_error())
    core: Final = build(make_core, fallbacks=[{"b": ["c"]}], disable_cooldowns=True)

    error: Final = await _fails(core)

    assert str(error).count("no fallback model group was found") == 1
    assert _fallback_logs(router_errors) == [f"{LOG_HEAD}{error}{NO_COOLDOWNS}"]


@pytest.mark.asyncio
async def test_the_outcome_message_of_a_failed_chain_stays_out_of_the_log(
    make_core: MakeCore, host: ScriptedHost, router_errors: list[str]
) -> None:
    show_router_debug(host)
    host.answer("openai/a", server_error())
    host.answer("openai/b", server_error())
    core: Final = build(make_core, fallbacks=[{"a": ["b"]}], disable_cooldowns=True)

    error: Final = await _fails(core)

    logged: Final = _fallback_logs(router_errors)
    assert len(logged) == 2
    assert "Fallback to b also failed" in str(error)
    assert not any("also failed" in entry for entry in logged)


@pytest.mark.asyncio
async def test_a_content_policy_list_without_the_group_logs_before_its_outcome_message(
    make_core: MakeCore, host: ScriptedHost, router_errors: list[str]
) -> None:
    show_router_debug(host)
    host.answer("openai/a", content_policy())
    core: Final = build(make_core, content_policy_fallbacks=[{"b": ["c"]}], disable_cooldowns=True)

    error: Final = await _fails(core)

    (logged,) = _fallback_logs(router_errors)
    assert logged.endswith(NO_COOLDOWNS)
    assert len(logged) < len(f"{LOG_HEAD}{error}{NO_COOLDOWNS}")


@pytest.mark.asyncio
async def test_max_fallbacks_of_zero_logs_the_levels_error(
    make_core: MakeCore, host: ScriptedHost, router_errors: list[str]
) -> None:
    host.answer("openai/a", server_error())
    core: Final = build(make_core, fallbacks=[{"a": ["b"]}], max_fallbacks=0, disable_cooldowns=True)

    error: Final = await _fails(core)

    assert _fallback_logs(router_errors) == [f"{LOG_HEAD}{error}{NO_COOLDOWNS}"]


@pytest.mark.asyncio
async def test_the_order_path_logs_only_where_its_last_tier_reaches_the_regular_fallbacks(
    make_core: MakeCore, host: ScriptedHost, router_errors: list[str]
) -> None:
    host.answer("openai/a1", server_error())
    host.answer("openai/a2", server_error())
    core: Final = make_core(model_list=list(TIERED_A), num_retries=0, fallbacks=[{"b": ["c"]}], disable_cooldowns=True)

    _ = await _fails(core)

    assert len(_fallback_logs(router_errors)) == 1


@pytest.mark.asyncio
async def test_disabled_fallbacks_log_nothing(
    make_core: MakeCore, host: ScriptedHost, router_errors: list[str]
) -> None:
    host.answer("openai/a", server_error())
    core: Final = build(make_core, fallbacks=[{"a": ["b"]}])

    _ = await _fails(core, disable_fallbacks=True)

    assert _fallback_logs(router_errors) == []


def test_a_sync_call_logs_through_the_same_handling(
    make_core: MakeCore, host: ScriptedHost, router_errors: list[str]
) -> None:
    host.answer("openai/a", server_error())
    core: Final = build(make_core, fallbacks=[{"b": ["c"]}], disable_cooldowns=True)

    with pytest.raises(litellm.InternalServerError) as raised:
        _ = complete_sync(core)

    assert _fallback_logs(router_errors) == [f"{LOG_HEAD}{raised.value}{NO_COOLDOWNS}"]


@pytest.mark.asyncio
async def test_the_log_lists_the_deployments_cooling_down(
    make_core: MakeCore, host: ScriptedHost, router_errors: list[str]
) -> None:
    host.answer("openai/a1", auth_error())
    host.answer("openai/a2", auth_error())
    core: Final = make_core(
        model_list=[deployment("a", "a1"), deployment("a", "a2"), deployment("b", "b")],
        num_retries=0,
        fallbacks=[{"b": ["c"]}],
    )

    _ = await _fails(core)

    (logged,) = _fallback_logs(router_errors)
    assert "Cooldown Deployments=[('a" in logged
