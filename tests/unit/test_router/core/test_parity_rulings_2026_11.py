"""
Parity of the Rust core with ``router_python_core.py`` on the router differences ruled on
2026-11-05, where both cores now take the side that was ruled correct: team ownership on the sync
and by-id paths, settings lists that change only through ``apply()``, one unreadable health record,
a failed ``set_model_list`` that keeps the old list, odd ``cooldown_time`` and status values,
numbers a deployment cannot be served with, an empty fallback entry, and a ``/v1/messages``
stream that raises before its first frame.
"""

import asyncio
import datetime
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Final, cast

import pytest

import litellm
from litellm.router_core import RouterCore
from litellm.utils import get_deployment_order
from tests.unit.test_router.core.scripted_host import ScriptedHost, deployment, rate_limit
from tests.unit.test_router.core.streaming_helpers import (
    MESSAGE_START,
    MESSAGE_STOP,
    MESSAGES,
    anthropic_stream,
    collect_bytes,
    install_orjson_stub_if_missing,
    text_delta,
)

install_orjson_stub_if_missing()

MakeCore = Callable[..., RouterCore]


def _cooled(core: RouterCore, *ids: str) -> dict[str, Mapping[str, object]]:  # mutable-ok: a fresh dict
    return dict(core.health().cooldowns.get_active_cooldowns(list(ids), None))


def _sync(core: object, name: str) -> Callable[..., object]:
    """A member both cores serve outside the ``RouterCore`` protocol (an endpoint or a deprecated name)."""
    return cast("Callable[..., object]", getattr(core, name))  # cast-ok: the member is a plain function


def _async(core: object, name: str) -> Callable[..., Awaitable[object]]:
    """``_sync`` for a coroutine member."""
    return cast("Callable[..., Awaitable[object]]", getattr(core, name))  # cast-ok: the member is a coroutine function


async def _on_failure(core: RouterCore, logged: Mapping[str, object]) -> None:
    """``deployment_callback_on_failure``, awaited when the core answers a coroutine."""
    now: Final = datetime.datetime(2026, 1, 1)
    outcome: Final = _sync(core, "deployment_callback_on_failure")(logged, None, now, now)
    if asyncio.iscoroutine(outcome):
        _ = cast("object", await outcome)  # cast-ok: the callback's verdict, unused
    await _settle()


def _ids(core: RouterCore) -> list[str]:  # mutable-ok: a fresh list
    return [str(entry["model_info"]["id"]) for entry in core.registry().all()]  # pyright: ignore[reportAny]  # model_list rows


async def _settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


# --- team ownership --------------------------------------------------------------------------------

_TEAM_OWNED: Final = [deployment("own", "o1", model_info={"team_id": "T1"})]


@pytest.mark.parametrize("metadata", [{}, {"user_api_key_team_id": "T2"}], ids=["no-team", "another-team"])
def test_a_sync_call_is_refused_a_deployment_another_team_owns(
    make_core: MakeCore, host: ScriptedHost, metadata: dict[str, str]
) -> None:
    core: Final = make_core(model_list=list(_TEAM_OWNED), num_retries=0)

    with pytest.raises(Exception, match=r"(?i)no (healthy )?deployments"):
        _sync(core, "completion")(model="own", messages=MESSAGES, metadata=dict(metadata))

    assert host.deployments_called("completion") == []


def test_a_sync_pick_skips_a_deployment_another_team_owns(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(_TEAM_OWNED), num_retries=0)

    with pytest.raises(Exception, match=r"(?i)no (healthy )?deployments"):
        _sync(core, "get_available_deployment")(model="own", messages=MESSAGES, request_kwargs={})


@pytest.mark.asyncio
async def test_a_pick_by_id_is_refused_another_teams_deployment(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=[deployment("g", "t1-dep", model_info={"team_id": "T1"})], num_retries=0)

    with pytest.raises(Exception, match="g"):
        await _async(core, "acompletion")(model="t1-dep", messages=MESSAGES, metadata={"user_api_key_team_id": "T2"})

    assert host.deployments_called("acompletion") == []


# --- settings change only through apply() ---------------------------------------------------------


@pytest.mark.asyncio
async def test_an_entry_appended_to_the_fallbacks_list_is_ignored(make_core: MakeCore, host: ScriptedHost) -> None:
    listed: Final[list[object]] = []  # mutable-ok: the caller's list, edited after construction
    core: Final = make_core(model_list=[deployment("g", "d1"), deployment("fb", "f1")], num_retries=0, fallbacks=listed)
    host.answer("openai/d1", rate_limit())
    listed.append({"g": ["fb"]})

    with pytest.raises(litellm.RateLimitError):
        await _async(core, "acompletion")(model="g", messages=MESSAGES)

    assert host.deployments_called("acompletion") == ["d1"]


# --- one unreadable health record ------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("timestamp", ["soon", True, None], ids=["string", "bool", "none"])
async def test_an_unreadable_health_timestamp_drops_only_that_record(
    make_core: MakeCore, host: ScriptedHost, timestamp: object
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "d1", weight=1), deployment("g", "d2", weight=0), deployment("x", "x1")],
        num_retries=0,
        enable_health_check_routing=True,
    )
    states: Final = {
        "d1": {"is_healthy": False, "timestamp": time.time()},
        "x1": {"is_healthy": False, "timestamp": timestamp},
    }
    health_checks: Final = core.health().health_checks
    # As another writer left it in the shared cache, then merged by this router's own writer.
    _sync(health_checks.cache, "set_cache")(key=health_checks.CACHE_KEY, value=states)
    await _async(core, "acompletion")(model="g", messages=MESSAGES)
    health_checks.set_deployment_health_states(states)  # pyright: ignore[reportArgumentType]  # a record with a bad timestamp, on purpose
    await _async(core, "acompletion")(model="g", messages=MESSAGES)

    assert host.deployments_called("acompletion") == ["d2", "d2"]


# --- a failed set_model_list -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_failed_set_model_list_keeps_the_old_list(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=[deployment("a", "d-a"), deployment("b", "d-b")], num_retries=0)
    unservable: Final = deployment("e", "d-e", input_cost_per_token=float("inf"))

    with pytest.raises(ValueError, match="cannot be served"):
        _sync(core, "set_model_list")([deployment("c", "d-c"), unservable])

    assert sorted(_ids(core)) == ["d-a", "d-b"]
    await _async(core, "acompletion")(model="a", messages=MESSAGES)
    assert host.deployments_called("acompletion") == ["d-a"]


# --- odd cooldown_time and status values ------------------------------------------------------------


@pytest.mark.parametrize("value", [True, False], ids=["true", "false"])
def test_a_bool_cooldown_time_is_refused_at_construction(make_core: MakeCore, value: bool) -> None:
    with pytest.raises(TypeError, match="cooldown_time"):
        make_core(model_list=[deployment("g", "d1")], cooldown_time=value)


# A group of one is never cooled down, so these groups hold two.
_TWO: Final = [deployment("g", "d1"), deployment("g", "d2")]


def _failure(
    deployment_id: str, exception: Exception, **litellm_params: object
) -> dict[str, object]:  # mutable-ok: logging kwargs
    return {
        "litellm_params": {"metadata": {"model_group": "g"}, "model_info": {"id": deployment_id}, **litellm_params},
        "call_type": "acompletion",
        "exception": exception,
    }


@pytest.mark.asyncio
async def test_a_bool_cooldown_time_on_the_call_cools_for_the_router_default(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(_TWO), num_retries=0, allowed_fails=0, cooldown_time=60)

    await _on_failure(core, _failure("d1", rate_limit(), cooldown_time=True))

    assert _cooled(core, "d1")["d1"]["cooldown_time"] == 60


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [70000, True, -1], ids=["too-wide", "bool", "negative"])
async def test_a_status_outside_http_is_recorded_as_500(make_core: MakeCore, status: object) -> None:
    core: Final = make_core(model_list=list(_TWO), num_retries=0, allowed_fails=0)
    error: Final = rate_limit()
    error.status_code = status  # pyright: ignore[reportAttributeAccessIssue]  # a provider error with an odd status

    await _on_failure(core, _failure("d1", error))

    assert str(_cooled(core, "d1")["d1"]["status_code"]) == "500"


# --- numbers a deployment cannot be served with -----------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_cost_per_token", float("inf")),
        ("output_cost_per_token", float("nan")),
        ("input_cost_per_token", "1e400"),
    ],
    ids=["inf", "nan", "overflowing-string"],
)
def test_a_non_finite_cost_is_refused_by_name(make_core: MakeCore, field: str, value: object) -> None:
    with pytest.raises(ValueError, match=f"Deployment 'g' cannot be served: .*{field}.*finite costs"):
        make_core(model_list=[deployment("g", "d1", model_info={field: value})])


def test_a_ptu_share_wider_than_64_bits_is_refused_at_build_and_on_add(make_core: MakeCore) -> None:
    wide: Final = deployment("g", "sh", model_info={"db_model": True, "ptu_shares": {"T1": 10**20}})
    with pytest.raises(ValueError, match=r"model_info\.ptu_shares\.T1 is 100000000000000000000, outside"):
        make_core(model_list=[wide])

    core: Final = make_core(model_list=[deployment("g", "free")])
    with pytest.raises(ValueError, match=r"ptu_shares\.T1 .* outside"):
        _sync(core, "add_deployment")(
            litellm.Deployment.model_validate(
                deployment("g", "late", model_info={"db_model": True, "ptu_shares": {"T1": 10**20}})
            )
        )
    assert _ids(core) == ["free"]


# --- an empty fallback entry --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_empty_fallback_entry_is_skipped(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=[deployment("g", "d1"), deployment("fb", "f1")], num_retries=0)
    host.answer("openai/d1", rate_limit())

    await _async(core, "acompletion")(model="g", messages=MESSAGES, fallbacks=[{}, {"g": ["fb"]}])

    assert host.deployments_called("acompletion") == ["d1", "f1"]


# --- a deployment's order --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("order", "expected"),
    [(1, 1), ("2", 2), (" 3", None), (2.0, 2), (2.5, None), (True, 1), ("x", None), (None, None)],
    ids=["int", "digits", "padded", "whole-float", "fraction", "bool", "word", "unset"],
)
def test_a_deployment_order_reads_as_a_whole_number(order: object, expected: int | None) -> None:
    assert get_deployment_order({"litellm_params": {"order": order}}) == expected


# --- a /v1/messages stream that raises before its first frame ----------------------------------------


@pytest.mark.asyncio
async def test_a_messages_stream_raising_before_its_first_frame_cools_and_retries(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    frames: Final = [MESSAGE_START, text_delta("B"), MESSAGE_STOP]
    host.answer("openai/d1", anthropic_stream(error=rate_limit(message="provider text")))
    host.answer("openai/d2", anthropic_stream(*frames))
    core: Final = make_core(
        model_list=[deployment("g", "d1", weight=1), deployment("g", "d2", weight=0)],
        num_retries=1,
        retry_after=0,
        allowed_fails=0,
    )

    stream: Final = await _async(core, "aanthropic_messages")(model="g", messages=MESSAGES, max_tokens=5, stream=True)
    assert await collect_bytes(stream) == frames
    await _settle()

    assert host.deployments_called("anthropic_messages") == ["d1", "d2"]
    assert sorted(_cooled(core, "d1", "d2")) == ["d1"]
    assert "provider text" in str(_cooled(core, "d1")["d1"]["exception_received"])
