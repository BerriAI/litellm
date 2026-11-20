import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
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
    BACKGROUND_RESPONSE_POLL_SCHEDULE,
    BackgroundResponsePoller,
    HttpxBackgroundResponseFetch,
    poll_request_for_create,
    resume_background_response_polls,
)

RESUMED_AT: Final = datetime(2026, 10, 17, 16, 0, 0, tzinfo=timezone.utc)
RESPONSES_URL: Final = "https://api.openai.com/v1/responses"
RESPONSE_ID: Final = "resp_080b10ff5d740c69006ad3de732e5087"
COMPLETED: Final = {"id": RESPONSE_ID, "status": "completed", "background": True, "usage": {"total_tokens": 16}}
IN_PROGRESS: Final = {"id": RESPONSE_ID, "status": "in_progress", "background": True, "usage": None}
FAST: Final = PollSchedule(initial_interval_seconds=0.001, max_interval_seconds=0.001, timeout_seconds=0.01)
ProviderAnswer = httpx.Response | httpx.HTTPError


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

    def poller(self, schedule: PollSchedule = FAST, database_configured: bool = True) -> BackgroundResponsePoller:
        client: Final = httpx.AsyncClient(transport=httpx.MockTransport(self.handle))
        return BackgroundResponsePoller(
            fetch=HttpxBackgroundResponseFetch(client=lambda: client),
            schedule=schedule,
            database_configured=lambda: database_configured,
            now=lambda: RESUMED_AT,
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


class _CreatorTable:
    def __init__(self, rows: tuple[tuple[BackgroundResponseCreator, datetime], ...]) -> None:
        self.rows: Final = tuple(
            _Row(creator.response_id, creator.upstream_url, creator.context.model_dump(mode="json"), created_at)
            for creator, created_at in rows
        )

    async def find_many(self, *, where: Mapping[str, Mapping[str, datetime]]) -> list[_Row]:
        return [row for row in self.rows if row.created_at > where["created_at"]["gt"]]


def _store(*rows: tuple[BackgroundResponseCreator, datetime]) -> PrismaBackgroundResponseCreatorStore:
    table: Final = _CreatorTable(rows)
    return PrismaBackgroundResponseCreatorStore(table=lambda: table, timeout_seconds=5)


async def _credential(api_base: str) -> Mapping[str, str]:
    return {"authorization": "Bearer resumed-openai-key"}


async def _no_credential(api_base: str) -> None:
    return None


async def _resume(
    provider: _Provider,
    settled: _Settled,
    *rows: tuple[BackgroundResponseCreator, datetime],
    poll_headers: Callable[[str], object] = _credential,
    poller: BackgroundResponsePoller | None = None,
) -> int:
    polls: Final = await resume_background_response_polls(
        creators=_store(*rows),
        poller=poller or provider.poller(),
        settle=settled,
        api_base="https://api.openai.com/",
        poll_headers=poll_headers,
    )
    await asyncio.gather(*polls)
    return len(polls)


async def _polled(provider: _Provider, settled: _Settled, schedule: PollSchedule = FAST) -> None:
    create: Final = httpx.Request("POST", RESPONSES_URL, headers={"authorization": "Bearer creator-upstream-key"})
    poll: Final = provider.poller(schedule).start(_creator(), poll_request_for_create(create, RESPONSE_ID), settled)
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
    poll: Final = provider.poller().start(_creator(), poll_request_for_create(create, RESPONSE_ID), failing_settlement)
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

    poll: Final = provider.poller().start(_creator(), poll_request_for_create(create, RESPONSE_ID), _Settled())
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
