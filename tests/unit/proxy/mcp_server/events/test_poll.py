from typing import Final

import pytest

from litellm.proxy.mcp_server.events.access import principal_of
from litellm.proxy.mcp_server.events.config import EventsSettings
from litellm.proxy.mcp_server.events.cursors import CursorBinding, arguments_digest, open_cursor, seal_cursor
from litellm.proxy.mcp_server.events.observability import EventActivity
from litellm.proxy.mcp_server.events.poll import poll
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Ok
from litellm.types.mcp_server.events import EventsErrorCode, PollEventsParams
from litellm.types.mcp_server.mcp_server_manager import MCPServerEventsConfig

from .fakes import CALLER, SETTINGS, FakeUpstream, context, occurrence, runtime, screen_blocking, server

NOW: Final = 1_800_000_000
ARGS: Final = {"repo": "acme/web"}
BINDING: Final = CursorBinding("srv-ci", "build.failed", arguments_digest(ARGS), principal_of(CALLER) or "")


@pytest.fixture(autouse=True)
def salt_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "synthetic-events-poll-key")


def _upstream(
    events: tuple[object, ...] = (), cursor: str | None = "9", next_poll_ms: int = 200, has_more: bool = False
) -> FakeUpstream:
    return FakeUpstream(
        responses={
            "events/poll": {
                "events": [item.model_dump(by_alias=True) for item in events],  # pyright: ignore[reportAttributeAccessIssue]  # occurrences
                "cursor": cursor,
                "truncated": False,
                "hasMore": has_more,
                "nextPollMs": next_poll_ms,
            }
        }
    )


@pytest.mark.asyncio
async def test_a_poll_renames_events_seals_the_cursor_and_strips_upstream_cursors() -> None:
    upstream: Final = _upstream((occurrence("evt_1", cursor="8"),), has_more=True)
    records: Final[list[EventActivity]] = []  # mutable-ok: collects recorded activity
    sealed_in: Final = seal_cursor(BINDING, "5", NOW)
    assert isinstance(sealed_in, Ok)
    result: Final = await poll(
        context(),
        PollEventsParams(name="ci-build.failed", arguments=ARGS, cursor=sealed_in.ok),
        {"mcp-name": "ci-build.failed", "mcp-param-repo": "acme/web"},
        SETTINGS,
        runtime((server(),), upstream=upstream, records=records),
    )
    assert isinstance(result, Ok)
    assert result.ok.has_more
    method, params, headers = upstream.calls[0]
    assert (method, params["name"], params["cursor"]) == ("events/poll", "build.failed", "5")
    folded: Final = {key.lower(): value for key, value in headers.items()}
    assert (folded["mcp-name"], folded["mcp-param-repo"]) == ("build.failed", "acme/web")
    assert [(item.name, item.cursor, item.meta) for item in result.ok.events] == [("ci-build.failed", None, None)]
    reopened: Final = open_cursor(result.ok.cursor, BINDING, NOW)
    assert isinstance(reopened, Ok) and reopened.ok.upstream == "9"
    assert [(item.outcome, item.event_ids) for item in records] == [("delivered", ("evt_1",))]


@pytest.mark.parametrize(("upstream_ms", "floor_ms", "expected"), ((200, 1000, 1000), (5000, 1000, 5000), (0, 0, 0)))
@pytest.mark.asyncio
async def test_next_poll_is_raised_to_the_server_floor(upstream_ms: int, floor_ms: int, expected: int) -> None:
    srv: Final = server(events=MCPServerEventsConfig(enabled=True, min_poll_interval_ms=floor_ms))
    result: Final = await poll(
        context(),
        PollEventsParams(name="ci-build.failed", arguments=ARGS),
        None,
        SETTINGS,
        runtime((srv,), upstream=_upstream(next_poll_ms=upstream_ms)),
    )
    assert isinstance(result, Ok) and result.ok.next_poll_ms == expected


@pytest.mark.asyncio
async def test_events_over_the_per_caller_budget_are_dropped_and_flagged_truncated() -> None:
    records: Final[list[EventActivity]] = []  # mutable-ok: collects recorded activity
    result: Final = await poll(
        context(),
        PollEventsParams(name="ci-build.failed", arguments=ARGS),
        None,
        EventsSettings(enabled=True, max_events_per_minute=1),
        runtime((server(),), upstream=_upstream((occurrence("evt_1"), occurrence("evt_2"))), records=records),
    )
    assert isinstance(result, Ok)
    assert ([item.event_id for item in result.ok.events], result.ok.truncated) == (["evt_1"], True)
    assert [(item.outcome, item.count, item.event_ids) for item in records] == [
        ("delivered", 1, ("evt_1",)),
        ("throttled", 1, ()),
    ]


@pytest.mark.asyncio
async def test_a_guardrail_block_drops_the_event_and_flags_truncated() -> None:
    records: Final[list[EventActivity]] = []  # mutable-ok: collects recorded activity
    result: Final = await poll(
        context(),
        PollEventsParams(name="ci-build.failed", arguments=ARGS),
        None,
        SETTINGS,
        runtime((server(),), upstream=_upstream((occurrence(),)), screen=screen_blocking(), records=records),
    )
    assert isinstance(result, Ok) and (result.ok.events, result.ok.truncated) == ((), True)
    assert [item.outcome for item in records] == ["blocked"]


@pytest.mark.asyncio
async def test_a_swapped_cursor_never_reaches_the_upstream() -> None:
    upstream: Final = _upstream()
    sealed: Final = seal_cursor(BINDING, "5", NOW)
    assert isinstance(sealed, Ok)
    result: Final = await poll(
        context(),
        PollEventsParams(name="ci-build.failed", arguments={"repo": "acme/api"}, cursor=sealed.ok),
        None,
        SETTINGS,
        runtime((server(),), upstream=upstream),
    )
    assert isinstance(result, Error) and result.error.code is EventsErrorCode.INVALID_PARAMS
    assert upstream.calls == []


@pytest.mark.asyncio
async def test_an_invalid_upstream_result_is_an_internal_error() -> None:
    result: Final = await poll(
        context(),
        PollEventsParams(name="ci-build.failed", arguments=ARGS),
        None,
        SETTINGS,
        runtime((server(),), upstream=FakeUpstream(responses={"events/poll": {"events": "nope"}})),
    )
    assert isinstance(result, Error) and result.error.code is EventsErrorCode.INTERNAL_ERROR


@pytest.mark.asyncio
async def test_a_poll_is_refused_before_reading_upstream_when_cursors_cannot_be_sealed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LITELLM_SALT_KEY", raising=False)
    monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
    upstream: Final = _upstream((occurrence(),))
    result: Final = await poll(
        context(),
        PollEventsParams(name="ci-build.failed", arguments=ARGS),
        None,
        SETTINGS,
        runtime((server(),), upstream=upstream),
    )
    assert isinstance(result, Error) and upstream.calls == []
