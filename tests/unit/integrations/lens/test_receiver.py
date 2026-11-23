import json
from datetime import datetime, timezone
from typing import Final, cast

import httpx
import pytest
from pydantic import JsonValue

from litellm.constants import AGENT_TRACING_AGENT_LIST_LIMIT
from litellm.integrations.lens import TraceReceiver
from litellm.integrations.lens.generated.models import TraceAgentRow, TraceAgentsParams
from litellm.integrations.lens.generated.types import TraceScope
from litellm.integrations.lens.remote import RemoteTraceStore
from litellm.integrations.lens.storage import LensTraceStorage
from litellm.integrations.lens.types import TraceAgent


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
@pytest.mark.parametrize(
    ("scope", "filters"),
    (
        pytest.param(TraceScope(all_teams=0, user_id="u1", team_ids=("t1",)), {}, id="no-filter"),
        pytest.param(
            TraceScope(all_teams=0, user_id="u1", team_ids=("t1",), filter_team_id="", filter_api_key_hash=""),
            {},
            id="empty-filter",
        ),
        pytest.param(
            TraceScope(all_teams=0, user_id="u1", team_ids=("t1",), filter_team_id="t1", filter_api_key_hash=""),
            {"filter_team_id": "t1"},
            id="team-filter",
        ),
        pytest.param(
            TraceScope(all_teams=0, user_id="u1", team_ids=("t1",), filter_team_id="", filter_api_key_hash="h1"),
            {"filter_api_key_hash": "h1"},
            id="key-filter",
        ),
    ),
)
async def test_list_agents_sends_lens_a_filter_field_only_when_the_filter_is_set(
    scope: TraceScope, filters: dict[str, str]
) -> None:
    bodies: Final[list[JsonValue]] = []  # mutable-ok: records what the scripted Lens received

    def lens(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"meta": [], "data": [], "rows": 0, "statistics": {"elapsed": 0, "rows_read": 0, "bytes_read": 0}},
        )

    async with httpx.AsyncClient(base_url="http://lens", transport=httpx.MockTransport(lens)) as client:
        receiver: Final = TraceReceiver(storage=LensTraceStorage(RemoteTraceStore(client)))
        await receiver.list_agents(scope, start_ms=10, end_ms=20)

    assert bodies == [
        {
            "operation": "query",
            "name": "trace_agents",
            "parameters": {
                "all_teams": 0,
                "user_id": "u1",
                "team_ids": ["t1"],
                "start_ms": 10,
                "end_ms": 20,
                "limit": AGENT_TRACING_AGENT_LIST_LIMIT,
                **filters,
            },
        }
    ]
