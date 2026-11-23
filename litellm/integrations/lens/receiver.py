from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Final

from litellm.constants import AGENT_TRACING_AGENT_LIST_LIMIT, AGENT_TRACING_LIST_PAGE_SIZE
from litellm.integrations.lens.generated.models import TraceAgentsParams
from litellm.integrations.lens.generated.types import (
    SpanDetail,
    SpanErrorPage,
    Trace,
    TraceConversationPage,
    TracePage,
    TraceScope,
)
from litellm.integrations.lens.storage import LensTraceStorage
from litellm.integrations.lens.types import TraceAgent, TraceAgentList


def _agent_filters(scope: TraceScope) -> Mapping[str, str]:
    filters: Final = {
        "filter_team_id": scope.get("filter_team_id", ""),
        "filter_api_key_hash": scope.get("filter_api_key_hash", ""),
    }
    return {name: value for name, value in filters.items() if value}


class TraceReceiver:
    def __init__(self, storage: LensTraceStorage) -> None:
        self.storage: Final = storage

    async def list_traces(self, scope: TraceScope, start_ms: int, end_ms: int, cursor: str | None = None) -> TracePage:
        return await self.storage.list_traces(scope, start_ms, end_ms, cursor, AGENT_TRACING_LIST_PAGE_SIZE)

    async def list_agents(self, scope: TraceScope, start_ms: int, end_ms: int) -> TraceAgentList:
        rows: Final = await self.storage.trace_agents(
            TraceAgentsParams(
                all_teams=scope["all_teams"],
                user_id=scope["user_id"],
                team_ids=tuple(scope["team_ids"]),
                start_ms=start_ms,
                end_ms=end_ms,
                limit=AGENT_TRACING_AGENT_LIST_LIMIT,
                **_agent_filters(scope),
            )
        )
        return TraceAgentList(
            agents=tuple(
                TraceAgent(
                    name=row.agent_name,
                    runs=row.runs,
                    failed_runs=row.failed_runs,
                    last_seen=datetime.fromtimestamp(row.last_seen_ms / 1000, tz=timezone.utc),
                    frameworks=row.frameworks,
                )
                for row in rows
            )
        )

    async def get_trace(
        self,
        trace_id: str,
        scope: TraceScope,
        trace_ref: str = "",
        cursor: str | None = None,
        page_size: int | None = None,
    ) -> Trace | None:
        return await self.storage.get_trace(trace_id, scope, trace_ref, cursor, page_size)

    async def get_span(self, trace_id: str, span_id: str, scope: TraceScope, trace_ref: str = "") -> SpanDetail | None:
        return await self.storage.get_span(trace_id, span_id, scope, trace_ref)

    async def get_conversation(
        self,
        trace_id: str,
        scope: TraceScope,
        trace_ref: str = "",
        cursor: str | None = None,
        page_size: int | None = None,
    ) -> TraceConversationPage | None:
        return await self.storage.get_conversation(trace_id, scope, trace_ref, cursor, page_size)

    async def get_span_error(
        self, trace_id: str, span_id: str, scope: TraceScope, trace_ref: str = "", cursor: str | None = None
    ) -> SpanErrorPage | None:
        return await self.storage.get_span_error(trace_id, span_id, scope, trace_ref, cursor)
