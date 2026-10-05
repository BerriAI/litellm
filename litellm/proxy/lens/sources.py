import base64
import json
from collections.abc import Awaitable, Sequence
from typing import Final, Protocol, TypeAlias

from pydantic import TypeAdapter

from litellm.proxy.lens.models import (
    ActivitySelection,
    Evidence,
    Execution,
    ExecutionContent,
    MetadataFilter,
    Sample,
    Scope,
    TracePart,
)
from litellm.rust_bridge.trace.generated.models import (
    ActivityAvailability,
    AgentRow,
    CountRow,
    ExecutionRow,
    LensAccessParams,
    LensContentParams,
    LensEvidenceParams,
    LensSampleParams,
    PartRow,
)


class Storage(Protocol):
    def lens_availability(self, parameters: LensAccessParams) -> Awaitable[Sequence[ActivityAvailability]]: ...
    def lens_agents(self, parameters: LensAccessParams) -> Awaitable[Sequence[AgentRow]]: ...
    def lens_sample(self, parameters: LensSampleParams) -> Awaitable[Sequence[ExecutionRow]]: ...
    def lens_content(self, parameters: LensContentParams) -> Awaitable[Sequence[PartRow]]: ...
    def lens_evidence(self, parameters: LensEvidenceParams) -> Awaitable[Sequence[CountRow]]: ...


ExecutionIdParts: TypeAlias = tuple[str, str, str] | tuple[str, str, str, str]
_EXECUTION_ID: Final[TypeAdapter[ExecutionIdParts]] = TypeAdapter(ExecutionIdParts)


def execution_id(source: str, team_id: str, trace_id: str, trace_ref: str = "") -> str:
    return base64.urlsafe_b64encode(json.dumps((source, team_id, trace_id, trace_ref)).encode()).decode()


def parse_execution(value: str) -> tuple[str, str, str, str]:
    parts: Final = _EXECUTION_ID.validate_json(base64.urlsafe_b64decode(value))
    return (parts[0], parts[1], parts[2], parts[3] if len(parts) == 4 else "")


def access_parameters(scope: Scope) -> LensAccessParams:
    return LensAccessParams(all_teams=1 if scope.all_teams else 0, team=scope.team_id, key_hash=scope.api_key_hash)


def selection_id(value: str) -> str:
    source, team, trace_id, trace_ref = parse_execution(value)
    return "\0".join((source, team, trace_ref or trace_id))


class SourceReader:
    def __init__(self, storage: Storage) -> None:
        self.storage: Final = storage

    async def availability(self, scope: Scope) -> ActivityAvailability:
        rows: Final = await self.storage.lens_availability(access_parameters(scope))
        return rows[0] if rows else ActivityAvailability()

    async def agents(self, scope: Scope) -> tuple[str, ...]:
        rows: Final = await self.storage.lens_agents(access_parameters(scope))
        return tuple(row.agent_name for row in rows)

    async def sample(
        self,
        scope: Scope,
        settings: ActivitySelection,
        start: int,
        end: int,
        offset: int = 0,
        page_size: int = 100,
        preview: bool = False,
        cursor: str = "",
    ) -> Sample:
        params: Final = LensSampleParams(
            all_teams=1 if scope.all_teams else 0,
            team=scope.team_id,
            key_hash=scope.api_key_hash,
            source=settings.source,
            start=start,
            end=end,
            service=settings.service,
            agent_name=settings.agent_name,
            filter_keys=tuple(f.key for f in settings.filters),
            filter_values=tuple(f.value for f in settings.filters),
            limit=page_size,
            offset=offset,
            after=cursor,
            sample_percent=settings.sample_percent,
            sample_cap=settings.sample_size or 0,
            preview=1 if preview else 0,
            selected_team=settings.team_id,
            execution_ids=tuple(selection_id(value) for value in settings.execution_ids),
        )
        rows: Final = await self.storage.lens_sample(params)
        return Sample(
            eligible=rows[0].eligible if rows else 0,
            selected=rows[0].selected if rows else 0,
            next_cursor=rows[-1].selection_key if len(rows) == page_size else None,
            next_offset=(
                offset + len(rows)
                if page_size and rows and offset + len(rows) < (rows[0].eligible if preview else rows[0].selected)
                else None
            ),
            executions=tuple(
                Execution(
                    id=execution_id(row.source, row.team_id, row.trace_id, row.trace_ref),
                    source=row.source,
                    trace_id=row.trace_id,
                    trace_ref=row.trace_ref,
                    team_id=row.team_id,
                    name=row.name,
                    start_time=row.start_time,
                    span_count=row.span_count,
                    root_seen=bool(row.root_seen),
                    service=row.service,
                    metadata=tuple(
                        MetadataFilter(key=k, value=v)
                        for k, v in row.attributes
                        if k != "litellm.api_key_hash" and k and v
                    ),
                )
                for row in rows
            ),
        )

    async def content(self, scope: Scope, execution: Execution, cursor: str = "", offset: int = 0) -> ExecutionContent:
        params: Final = LensContentParams(
            all_teams=1 if scope.all_teams else 0,
            team=scope.team_id,
            key_hash=scope.api_key_hash,
            source=execution.source,
            id=execution.trace_id,
            trace_ref=execution.trace_ref,
            record_team=execution.team_id,
            cursor=cursor,
            offset=offset + 1,
        )
        rows: Final = await self.storage.lens_content(params)
        return ExecutionContent(
            execution=execution,
            parts=tuple(
                TracePart(
                    execution_id=execution.id,
                    span_id=row.span_id,
                    parent_span_id=row.parent_span_id,
                    name=row.name,
                    kind=row.kind,
                    content=row.content,
                    truncated=bool(row.truncated),
                )
                for row in rows
            ),
            next_cursor=rows[-1].span_id if len(rows) == 40 else None,
            partial=not execution.root_seen or any(row.truncated for row in rows),
        )

    async def verify_evidence(self, scope: Scope, execution: Execution, evidence: Evidence) -> bool:
        params: Final = LensEvidenceParams(
            all_teams=1 if scope.all_teams else 0,
            team=scope.team_id,
            key_hash=scope.api_key_hash,
            source=execution.source,
            id=execution.trace_id,
            trace_ref=execution.trace_ref,
            record_team=execution.team_id,
            span=evidence.span_id,
            quote=evidence.quote,
        )
        rows: Final = await self.storage.lens_evidence(params)
        return bool(rows and rows[0].count)
