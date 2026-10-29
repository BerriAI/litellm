import asyncio
import json
from datetime import datetime, timezone
from typing import Final, cast

import httpx
import pytest

from litellm.constants import AGENT_TRACING_AGENT_LIST_LIMIT, AGENT_TRACING_LIST_PAGE_SIZE
from litellm.tracing import TraceReceiver
from litellm.tracing.generated.models import TraceAgentRow, TraceAgentsParams
from litellm.tracing.generated.types import TraceScope
from litellm.tracing.remote import RemoteTraceStore
from litellm.tracing.storage import LensTraceStorage
from litellm.tracing.types import TraceAgent


class AgentRowsStorage:
    def __init__(self, rows: tuple[TraceAgentRow, ...]) -> None:
        self.rows: Final = rows
        self.requests: tuple[TraceAgentsParams, ...] = ()

    async def trace_agents(self, parameters: TraceAgentsParams) -> tuple[TraceAgentRow, ...]:
        self.requests = (*self.requests, parameters)
        return self.rows


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "all_teams", "user_id", "team_ids"),
    (
        pytest.param(TraceScope(all_teams=1, user_id="", team_ids=()), 1, "", (), id="all-teams"),
        pytest.param(TraceScope(all_teams=0, user_id="u1", team_ids=("t1", "t2")), 0, "u1", ("t1", "t2"), id="owned"),
    ),
)
async def test_list_agents_queries_the_reader_scope_and_shapes_rows(
    scope: TraceScope, all_teams: int, user_id: str, team_ids: tuple[str, ...]
) -> None:
    storage: Final = AgentRowsStorage(
        (
            TraceAgentRow(
                agent_name="moyai", runs=5, failed_runs=2, last_seen_ms=1_791_405_060_123, frameworks=("pi",)
            ),
            TraceAgentRow(agent_name="research", runs=1, failed_runs=0, last_seen_ms=0),
        )
    )
    receiver: Final = TraceReceiver(storage=cast(LensTraceStorage, storage))

    result: Final = await receiver.list_agents(scope, start_ms=10, end_ms=20)

    assert storage.requests == (
        TraceAgentsParams(
            all_teams=all_teams,
            user_id=user_id,
            team_ids=team_ids,
            start_ms=10,
            end_ms=20,
            limit=AGENT_TRACING_AGENT_LIST_LIMIT,
        ),
    )
    assert result.agents == (
        TraceAgent(
            name="moyai",
            runs=5,
            failed_runs=2,
            last_seen=datetime(2026, 10, 7, 20, 31, 0, 123000, tzinfo=timezone.utc),
            frameworks=("pi",),
        ),
        TraceAgent(
            name="research",
            runs=1,
            failed_runs=0,
            last_seen=datetime(1970, 1, 1, tzinfo=timezone.utc),
        ),
    )


@pytest.mark.asyncio
async def test_list_traces_forwards_agent_filter_to_lens() -> None:
    requests: Final = asyncio.Queue[httpx.Request]()

    def accept(request: httpx.Request) -> httpx.Response:
        requests.put_nowait(request)
        return httpx.Response(200, json={"data": [], "next_cursor": None})

    scope: Final = TraceScope(all_teams=0, user_id="owner", team_ids=("team",))
    async with httpx.AsyncClient(base_url="http://lens", transport=httpx.MockTransport(accept)) as client:
        receiver: Final = TraceReceiver(LensTraceStorage(RemoteTraceStore(client)))
        page: Final = await receiver.list_traces(scope, start_ms=10, end_ms=20, agent="research-agent")

    request: Final = requests.get_nowait()
    assert request.url.path == "/internal/read"
    assert json.loads(request.content) == {
        "operation": "list",
        "scope": {"all_teams": 0, "user_id": "owner", "team_ids": ["team"]},
        "start_ms": 10,
        "end_ms": 20,
        "cursor": None,
        "limit": AGENT_TRACING_LIST_PAGE_SIZE,
        "agent": "research-agent",
    }
    assert page == {"data": (), "next_cursor": None}
