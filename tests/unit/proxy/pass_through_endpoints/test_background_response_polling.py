import asyncio
import contextlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Final

import httpx
import pytest

from litellm.interactions.background_cost_polling import PollSchedule
from litellm.proxy.pass_through_endpoints.background_response_creator import (
    BackgroundResponseCreateContext,
    BackgroundResponseCreator,
    PrismaBackgroundResponseCreatorStore,
)
from litellm.proxy.pass_through_endpoints.background_response_polling import (
    BACKGROUND_RESPONSE_CREATOR_RETENTION,
    BACKGROUND_RESPONSE_POLL_SCHEDULE,
    BackgroundResponsePoller,
    BackgroundResponsePollRequest,
    Fetched,
    HttpxBackgroundResponseFetch,
    RetryLater,
    StopPolling,
    poll_request_for_create,
    sweep_background_response_polls,
    sweep_periodically,
)

RESUMED_AT: Final = datetime(2026, 10, 17, 16, 0, 0, tzinfo=timezone.utc)
RESPONSES_URL: Final = "https://api.openai.com/v1/responses"
RESPONSE_ID: Final = "resp_080b10ff5d740c69006ad3de732e5087"
COMPLETED: Final = {"id": RESPONSE_ID, "status": "completed", "background": True, "usage": {"total_tokens": 16}}
IN_PROGRESS: Final = {"id": RESPONSE_ID, "status": "in_progress", "background": True, "usage": None}
FAST: Final = PollSchedule(initial_interval_seconds=0.001, max_interval_seconds=0.001, timeout_seconds=0.01)
ProviderAnswer = httpx.Response | httpx.HTTPError
EXPIRED_TOKEN_READ: Final = BackgroundResponsePollRequest(
    url=httpx.URL(f"{RESPONSES_URL}/{RESPONSE_ID}"),
    headers=(("Authorization", "Bearer expired-token"), ("OpenAI-Project", "proj_creator")),
)


async def _no_credential(api_base: str) -> Mapping[str, str] | None:
    return None


async def _fresh_token(api_base: str) -> Mapping[str, str] | None:
    return {"authorization": "Bearer fresh-token", "api-key": "fresh-token"}


async def _the_expired_token(api_base: str) -> Mapping[str, str] | None:
    return {"authorization": "Bearer expired-token"}


async def _failed_token_exchange(api_base: str) -> Mapping[str, str] | None:
    raise RuntimeError("workload identity token exchange failed")


def _creator(response_id: str = RESPONSE_ID, upstream_url: str = RESPONSES_URL) -> BackgroundResponseCreator:
    return BackgroundResponseCreator(
        response_id=response_id,
        upstream_url=upstream_url,
        context=BackgroundResponseCreateContext(
            model="gpt-5.4-mini",
            custom_llm_provider="openai",
            litellm_call_id="0d4b7e19-6a2c-4f83-9e51-2c8a7f3b6d40",
            litellm_trace_id="5f1d2c3b-8e7a-4b69-9c0d-1a2b3c4d5e6f",
            start_time=datetime(2026, 10, 17, 13, 47, 3, tzinfo=timezone.utc),
            metadata={"user_api_key_team_id": "team-creator"},
        ),
    )


class _Provider:
    def __init__(self, answers: tuple[ProviderAnswer, ...]) -> None:
        self.answers: Final = list(answers)
        self.requests: Final[list[httpx.Request]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer: Final = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, httpx.HTTPError):
            raise answer
        return answer

    def poller(
        self, schedule: PollSchedule = FAST, database_configured: bool = True, max_active_polls: int = 1000
    ) -> BackgroundResponsePoller:
        client: Final = httpx.AsyncClient(transport=httpx.MockTransport(self.handle))
        return BackgroundResponsePoller(
            fetch=HttpxBackgroundResponseFetch(client=lambda: client, current_credential=_no_credential),
            schedule=schedule,
            database_configured=lambda: database_configured,
            now=lambda: RESUMED_AT,
            max_active_polls=max_active_polls,
        )


class _Settled:
    def __init__(self) -> None:
        self.bodies: Final[list[tuple[str, Mapping[str, object]]]] = []

    async def __call__(self, creator: BackgroundResponseCreator, body: Mapping[str, object]) -> bool:
        self.bodies.append((creator.response_id, body))
        return True


@dataclass(frozen=True, slots=True)
class _Row:
    response_id: str
    upstream_url: str
    create_context: object
    created_at: datetime
    poll_lease_expires_at: datetime | None = None


def _row(
    creator: BackgroundResponseCreator, created_at: datetime = RESUMED_AT, lease: datetime | None = None
) -> _Row:
    return _Row(creator.response_id, creator.upstream_url, creator.context.model_dump(mode="json"), created_at, lease)


def _lease_matches(lease: datetime | None, expiry: Mapping[str, datetime] | None) -> bool:
    if expiry is None:
        return lease is None
    return lease is not None and lease < expiry["lt"]


def _lease_open(row: _Row, where: Mapping[str, object]) -> bool:
    expiries: Final = where.get("OR")
    if not isinstance(expiries, Sequence):
        return True
    return any(_lease_matches(row.poll_lease_expires_at, expiry["poll_lease_expires_at"]) for expiry in expiries)


class _CreatorTable:
    def __init__(self, *rows: _Row) -> None:
        self.rows: Final = {row.response_id: row for row in rows}

    async def find_many(self, *, where: Mapping[str, Mapping[str, datetime]]) -> list[_Row]:
        recent: Final = (row for row in self.rows.values() if row.created_at > where["created_at"]["gt"])
        return [row for row in recent if _lease_open(row, where)]

    async def update_many(self, *, data: Mapping[str, datetime], where: Mapping[str, object]) -> int:
        row: Final = self.rows.get(str(where["response_id"]))
        if row is None or not _lease_open(row, where):
            return 0
        self.rows[row.response_id] = replace(row, poll_lease_expires_at=data["poll_lease_expires_at"])
        return 1

    async def delete_many(self, *, where: Mapping[str, Mapping[str, datetime]]) -> int:
        pruned: Final = tuple(
            response_id for response_id, row in self.rows.items() if row.created_at < where["created_at"]["lt"]
        )
        for response_id in pruned:
            del self.rows[response_id]
        return len(pruned)


class _UnreachableLeases(_CreatorTable):
    async def update_many(self, *, data: Mapping[str, datetime], where: Mapping[str, object]) -> int:
        raise ConnectionResetError("server closed the connection unexpectedly")


def _store(table: _CreatorTable) -> PrismaBackgroundResponseCreatorStore:
    return PrismaBackgroundResponseCreatorStore(table=lambda: table, timeout_seconds=5)


async def _credential(api_base: str) -> Mapping[str, str]:
    return {"authorization": "Bearer resumed-openai-key"}


async def _no_credential(api_base: str) -> None:
    return None


async def _swept(
    provider: _Provider,
    settled: _Settled,
    table: _CreatorTable,
    poll_headers: Callable[[str], object] = _credential,
    poller: BackgroundResponsePoller | None = None,
) -> int:
    polls: Final = await sweep_background_response_polls(
        creators=_store(table),
        poller=poller or provider.poller(),
        settle=settled,
        api_base="https://api.openai.com/",
        poll_headers=poll_headers,
    )
    await asyncio.gather(*polls)
    return len(polls)


async def _resume(
    provider: _Provider,
    settled: _Settled,
    *rows: tuple[BackgroundResponseCreator, datetime],
    poll_headers: Callable[[str], object] = _credential,
    poller: BackgroundResponsePoller | None = None,
) -> int:
    table: Final = _CreatorTable(*(_row(creator, created_at) for creator, created_at in rows))
    return await _swept(provider, settled, table, poll_headers, poller)


async def _polled(
    provider: _Provider,
    settled: _Settled,
    schedule: PollSchedule = FAST,
    table: _CreatorTable | None = None,
    settle: Callable[[BackgroundResponseCreator, Mapping[str, object]], object] | None = None,
) -> None:
    create: Final = httpx.Request("POST", RESPONSES_URL, headers={"authorization": "Bearer creator-upstream-key"})
    poll: Final = provider.poller(schedule).start(
        _creator(),
        poll_request_for_create(create, RESPONSE_ID),
        settle or settled,
        _store(_CreatorTable(_row(_creator())) if table is None else table),
    )
    assert poll is not None
    await poll


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transient",
    [
        httpx.Response(408),
        httpx.Response(409),
        httpx.Response(429),
        httpx.Response(500),
        httpx.Response(503),
        httpx.Response(200, json=["not", "a", "response"]),
        httpx.Response(200, content=b"<html>gateway</html>"),
        httpx.ConnectError("connection refused"),
        httpx.Response(200, json=IN_PROGRESS),
        httpx.Response(200, json={**IN_PROGRESS, "status": "queued"}),
    ],
)
async def test_a_poll_keeps_going_past_a_transient_answer_and_settles_the_finished_response(transient):
    provider: Final = _Provider((transient, httpx.Response(200, json=COMPLETED)))
    settled: Final = _Settled()

    await _polled(provider, settled)

    assert settled.bodies == [(RESPONSE_ID, COMPLETED)]
    assert len(provider.requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [400, 401, 403, 404])
async def test_a_poll_stops_without_settling_when_the_provider_refuses_the_read(status_code):
    provider: Final = _Provider((httpx.Response(status_code), httpx.Response(200, json=COMPLETED)))
    settled: Final = _Settled()

    await _polled(provider, settled)

    assert settled.bodies == []
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_a_poll_gives_up_without_settling_once_its_schedule_runs_out():
    provider: Final = _Provider((httpx.Response(200, json=IN_PROGRESS),))
    settled: Final = _Settled()
    schedule: Final = PollSchedule(initial_interval_seconds=0.001, max_interval_seconds=0.004, timeout_seconds=0.01)

    await _polled(provider, settled, schedule)

    assert settled.bodies == []
    assert len(provider.requests) == 3


@pytest.mark.asyncio
async def test_a_poll_whose_settlement_fails_ends_without_raising():
    provider: Final = _Provider((httpx.Response(200, json=COMPLETED),))

    async def failing_settlement(creator: BackgroundResponseCreator, body: Mapping[str, object]) -> bool:
        raise RuntimeError("spend log write failed")

    create: Final = httpx.Request("POST", RESPONSES_URL)
    poll: Final = provider.poller().start(
        _creator(),
        poll_request_for_create(create, RESPONSE_ID),
        failing_settlement,
        _store(_CreatorTable(_row(_creator()))),
    )
    assert poll is not None

    await poll

    assert poll.exception() is None
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_a_poll_reads_the_response_with_the_create_credentials_and_query_but_no_body_headers():
    provider: Final = _Provider((httpx.Response(200, json=COMPLETED),))
    create: Final = httpx.Request(
        "POST",
        "https://my-resource.openai.azure.com/openai/v1/responses/?api-version=preview",
        headers={"api-key": "creator-azure-key", "x-request-tag": "nightly"},
        json={"model": "gpt-5.4-mini", "input": "hi", "background": True},
    )

    poll: Final = provider.poller().start(
        _creator(), poll_request_for_create(create, RESPONSE_ID), _Settled(), _store(_CreatorTable(_row(_creator())))
    )
    assert poll is not None
    await poll

    (read,) = provider.requests
    assert read.method == "GET"
    assert read.url == httpx.URL(
        f"https://my-resource.openai.azure.com/openai/v1/responses/{RESPONSE_ID}?api-version=preview"
    )
    assert read.headers["api-key"] == "creator-azure-key"
    assert read.headers["x-request-tag"] == "nightly"
    assert "content-type" not in read.headers
    assert "content-length" not in read.headers
    assert read.content == b""


@pytest.mark.asyncio
async def test_a_poll_does_not_start_without_a_proxy_database():
    provider: Final = _Provider((httpx.Response(200, json=COMPLETED),))
    settled: Final = _Settled()

    resumed: Final = await _resume(
        provider, settled, (_creator(), RESUMED_AT), poller=provider.poller(database_configured=False)
    )

    assert resumed == 0
    assert provider.requests == []
    assert settled.bodies == []


@pytest.mark.asyncio
async def test_a_restart_resumes_only_creators_recorded_within_the_poll_window():
    window: Final = timedelta(seconds=BACKGROUND_RESPONSE_POLL_SCHEDULE.timeout_seconds)
    recent: Final = _creator("resp_recent")
    stale: Final = _creator("resp_stale")
    provider: Final = _Provider((httpx.Response(200, json=COMPLETED),))
    settled: Final = _Settled()
    poller: Final = provider.poller(
        PollSchedule(initial_interval_seconds=0.001, max_interval_seconds=0.001, timeout_seconds=window.total_seconds())
    )

    await _resume(
        provider,
        settled,
        (recent, RESUMED_AT - window + timedelta(seconds=1)),
        (stale, RESUMED_AT - window - timedelta(seconds=1)),
        poller=poller,
    )

    assert [response_id for response_id, _ in settled.bodies] == ["resp_recent"]
    assert [str(request.url) for request in provider.requests] == [f"{RESPONSES_URL}/resp_recent"]
    assert provider.requests[0].headers["authorization"] == "Bearer resumed-openai-key"


@pytest.mark.asyncio
@pytest.mark.parametrize("unsafe_id", ["resp_a/../../files", "resp_a\\b", "resp_a#b", "resp_a?b", "..", ""])
async def test_a_restart_never_polls_a_recorded_id_that_could_leave_the_responses_path(unsafe_id):
    provider: Final = _Provider((httpx.Response(200, json=COMPLETED),))
    settled: Final = _Settled()

    await _resume(provider, settled, (_creator(unsafe_id), RESUMED_AT), (_creator(), RESUMED_AT))

    assert [str(request.url) for request in provider.requests] == [f"{RESPONSES_URL}/{RESPONSE_ID}"]


@pytest.mark.asyncio
async def test_a_restart_without_an_openai_credential_polls_nothing():
    provider: Final = _Provider((httpx.Response(200, json=COMPLETED),))
    settled: Final = _Settled()

    resumed: Final = await _resume(provider, settled, (_creator(), RESUMED_AT), poll_headers=_no_credential)

    assert resumed == 0
    assert provider.requests == []


@pytest.mark.asyncio
async def test_a_restart_survives_a_failing_credential_lookup():
    provider: Final = _Provider((httpx.Response(200, json=COMPLETED),))

    async def failing_credential(api_base: str) -> Mapping[str, str]:
        raise RuntimeError("secret manager unreachable")

    resumed: Final = await _resume(provider, _Settled(), (_creator(), RESUMED_AT), poll_headers=failing_credential)

    assert resumed == 0
    assert provider.requests == []


@pytest.mark.asyncio
async def test_a_restart_with_no_row_on_the_configured_origin_polls_nothing_and_reads_no_credential():
    provider: Final = _Provider((httpx.Response(200, json=COMPLETED),))
    lookups: Final[list[str]] = []

    async def recorded_credential(api_base: str) -> Mapping[str, str]:
        lookups.append(api_base)
        return {"authorization": "Bearer resumed-openai-key"}

    azure_creator: Final = _creator(upstream_url="https://my-resource.openai.azure.com/openai/v1/responses")
    resumed: Final = await _resume(provider, _Settled(), (azure_creator, RESUMED_AT), poll_headers=recorded_credential)

    assert resumed == 0
    assert provider.requests == []
    assert lookups == []


@pytest.mark.asyncio
async def test_a_poll_stops_without_reading_once_another_settler_forgot_the_creator():
    provider: Final = _Provider((httpx.Response(200, json=COMPLETED),))
    settled: Final = _Settled()

    await _polled(provider, settled, table=_CreatorTable())

    assert provider.requests == []
    assert settled.bodies == []


@pytest.mark.asyncio
async def test_a_poll_keeps_going_when_the_database_cannot_renew_its_lease():
    provider: Final = _Provider((httpx.Response(200, json=COMPLETED),))
    settled: Final = _Settled()

    await _polled(provider, settled, table=_UnreachableLeases(_row(_creator())))

    assert settled.bodies == [(RESPONSE_ID, COMPLETED)]


@pytest.mark.asyncio
async def test_two_workers_sweeping_one_store_poll_each_response_once():
    provider: Final = _Provider((httpx.Response(200, json=COMPLETED),))
    table: Final = _CreatorTable(_row(_creator("resp_first")), _row(_creator("resp_second")))

    swept: Final = await asyncio.gather(
        _swept(provider, _Settled(), table, poller=provider.poller()),
        _swept(provider, _Settled(), table, poller=provider.poller()),
    )

    assert sum(swept) == 2
    assert sorted(str(request.url) for request in provider.requests) == [
        f"{RESPONSES_URL}/resp_first",
        f"{RESPONSES_URL}/resp_second",
    ]


@pytest.mark.asyncio
async def test_a_worker_at_its_poll_limit_starts_no_poll_for_a_new_create():
    provider: Final = _Provider((httpx.Response(200, json=COMPLETED),))
    poller: Final = provider.poller(max_active_polls=1)
    table: Final = _CreatorTable(_row(_creator("resp_first")), _row(_creator("resp_second")))
    create: Final = httpx.Request("POST", RESPONSES_URL)

    first: Final = poller.start(
        _creator("resp_first"), poll_request_for_create(create, "resp_first"), _Settled(), _store(table)
    )
    second: Final = poller.start(
        _creator("resp_second"), poll_request_for_create(create, "resp_second"), _Settled(), _store(table)
    )

    assert first is not None
    assert second is None
    await first


@pytest.mark.asyncio
async def test_a_worker_at_its_poll_limit_leaves_the_other_responses_unleased_for_its_next_sweep():
    provider: Final = _Provider((httpx.Response(200, json=COMPLETED),))
    settled: Final = _Settled()
    poller: Final = provider.poller(max_active_polls=1)
    table: Final = _CreatorTable(_row(_creator("resp_first")), _row(_creator("resp_second")))

    first_sweep: Final = await _swept(provider, settled, table, poller=poller)
    second_sweep: Final = await _swept(provider, settled, table, poller=poller)

    assert (first_sweep, second_sweep) == (1, 1)
    assert sorted(response_id for response_id, _ in settled.bodies) == ["resp_first", "resp_second"]


@pytest.mark.asyncio
async def test_a_sweep_takes_an_expired_or_missing_lease_but_never_a_live_one():
    provider: Final = _Provider((httpx.Response(200, json=COMPLETED),))
    table: Final = _CreatorTable(
        _row(_creator("resp_live"), lease=RESUMED_AT + timedelta(seconds=1)),
        _row(_creator("resp_expired"), lease=RESUMED_AT - timedelta(seconds=1)),
        _row(_creator("resp_unleased")),
    )

    await _swept(provider, _Settled(), table)

    assert sorted(str(request.url) for request in provider.requests) == [
        f"{RESPONSES_URL}/resp_expired",
        f"{RESPONSES_URL}/resp_unleased",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer", [httpx.Response(404), httpx.Response(200, json=IN_PROGRESS)], ids=["refused", "gave_up"]
)
async def test_a_poll_that_ends_unsettled_keeps_the_next_sweep_from_polling_it_again(answer):
    provider: Final = _Provider((answer,))
    table: Final = _CreatorTable(_row(_creator()))
    await _polled(provider, _Settled(), table=table)
    reads_before_the_sweep: Final = len(provider.requests)

    resumed: Final = await _swept(provider, _Settled(), table)

    assert resumed == 0
    assert len(provider.requests) == reads_before_the_sweep
    assert set(table.rows) == {RESPONSE_ID}


@pytest.mark.asyncio
async def test_a_sweep_deletes_only_creators_past_their_retention():
    provider: Final = _Provider((httpx.Response(200, json=COMPLETED),))
    table: Final = _CreatorTable(
        _row(_creator("resp_past"), RESUMED_AT - BACKGROUND_RESPONSE_CREATOR_RETENTION - timedelta(seconds=1)),
        _row(_creator("resp_kept"), RESUMED_AT - BACKGROUND_RESPONSE_CREATOR_RETENTION + timedelta(seconds=1)),
    )

    await _swept(provider, _Settled(), table)

    assert set(table.rows) == {"resp_kept"}


@pytest.mark.asyncio
async def test_the_periodic_sweep_sweeps_again_after_each_interval():
    swept_at: Final[list[float]] = []
    third_sweep: Final = asyncio.Event()

    async def sweep() -> None:
        swept_at.append(asyncio.get_running_loop().time())
        if len(swept_at) == 3:
            third_sweep.set()

    sweeping: Final = asyncio.create_task(sweep_periodically(sweep, 0.01))
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(third_sweep.wait(), timeout=5)
    sweeping.cancel()

    assert len(swept_at) >= 3
    assert all(later - earlier >= 0.01 for earlier, later in zip(swept_at, swept_at[1:]))


@pytest.mark.asyncio
async def test_a_fetch_the_provider_never_answers_is_retried_later_instead_of_holding_the_poll(socket_path):
    async def never_answer(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.read()
        writer.close()

    server: Final = await asyncio.start_unix_server(never_answer, path=str(socket_path))
    transport: Final = httpx.AsyncHTTPTransport(uds=str(socket_path))
    async with server, httpx.AsyncClient(transport=transport, timeout=None) as client:
        fetch: Final = HttpxBackgroundResponseFetch(client=lambda: client, timeout_seconds=0.1)
        fetching: Final = asyncio.ensure_future(
            fetch(BackgroundResponsePollRequest(url=f"http://provider/v1/responses/{RESPONSE_ID}", headers=()))
        )
        done, _ = await asyncio.wait({fetching}, timeout=2)
        fetching.cancel()

    assert fetching in done
    assert isinstance(fetching.result(), RetryLater)


@pytest.mark.asyncio
async def test_a_read_whose_token_expired_reads_again_with_the_pass_throughs_current_credential():
    provider: Final = _Provider((httpx.Response(401), httpx.Response(200, json=COMPLETED)))
    client: Final = httpx.AsyncClient(transport=httpx.MockTransport(provider.handle))
    asked_for: Final[list[str]] = []

    async def current_credential(api_base: str) -> Mapping[str, str] | None:
        asked_for.append(api_base)
        return await _fresh_token(api_base)

    outcome: Final = await HttpxBackgroundResponseFetch(client=lambda: client, current_credential=current_credential)(
        EXPIRED_TOKEN_READ
    )

    assert outcome == Fetched(body=COMPLETED)
    assert asked_for == ["https://api.openai.com/"]
    assert [read.headers["authorization"] for read in provider.requests] == [
        "Bearer expired-token",
        "Bearer fresh-token",
    ]
    assert provider.requests[1].headers["api-key"] == "fresh-token"
    assert provider.requests[1].headers["openai-project"] == "proj_creator"
    assert provider.requests[1].url == EXPIRED_TOKEN_READ.url


@pytest.mark.asyncio
@pytest.mark.parametrize("current_credential", [_no_credential, _the_expired_token, _failed_token_exchange])
async def test_a_refused_read_is_not_sent_again_without_a_different_credential(current_credential):
    provider: Final = _Provider((httpx.Response(401), httpx.Response(200, json=COMPLETED)))
    client: Final = httpx.AsyncClient(transport=httpx.MockTransport(provider.handle))

    outcome: Final = await HttpxBackgroundResponseFetch(client=lambda: client, current_credential=current_credential)(
        EXPIRED_TOKEN_READ
    )

    assert outcome == StopPolling(status_code=401)
    assert len(provider.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [403, 404])
async def test_a_read_refused_for_another_reason_keeps_its_credential(status_code):
    provider: Final = _Provider((httpx.Response(status_code), httpx.Response(200, json=COMPLETED)))
    client: Final = httpx.AsyncClient(transport=httpx.MockTransport(provider.handle))

    outcome: Final = await HttpxBackgroundResponseFetch(client=lambda: client, current_credential=_fresh_token)(
        EXPIRED_TOKEN_READ
    )

    assert outcome == StopPolling(status_code=status_code)
    assert len(provider.requests) == 1
