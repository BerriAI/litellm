import base64
import json
from collections.abc import Awaitable, Mapping
from types import MappingProxyType
from typing import Final, Literal, Protocol

from pydantic import BaseModel, TypeAdapter

from litellm.proxy.engine.models import (
    EngineSettings,
    Evidence,
    Execution,
    ExecutionContent,
    MetadataFilter,
    Sample,
    Scope,
    TracePart,
)


class Storage(Protocol):
    def lens_sample(self, parameters: Mapping[str, object]) -> Awaitable[object]: ...
    def lens_content(self, parameters: Mapping[str, object]) -> Awaitable[object]: ...
    def lens_evidence(self, parameters: Mapping[str, object]) -> Awaitable[object]: ...


class ExecutionRow(BaseModel):
    source: Literal["traces", "requests"]
    trace_id: str
    trace_ref: str = ""
    team_id: str
    name: str
    start_time: str
    span_count: int
    root_seen: int
    eligible: int
    service: str = ""
    attributes: tuple[tuple[str, str], ...] = ()


class PartRow(BaseModel):
    span_id: str
    parent_span_id: str
    name: str
    kind: str
    content: str
    truncated: int


class CountRow(BaseModel):
    count: int


_ROWS: Final = TypeAdapter(tuple[ExecutionRow, ...])
_PARTS: Final = TypeAdapter(tuple[PartRow, ...])
_COUNTS: Final = TypeAdapter(tuple[CountRow, ...])


def execution_id(source: str, team_id: str, trace_id: str, trace_ref: str = "") -> str:
    return base64.urlsafe_b64encode(json.dumps((source, team_id, trace_id, trace_ref)).encode()).decode()


def parse_execution(value: str) -> tuple[str, str, str, str]:
    parts: Final = TypeAdapter(tuple[str, str, str] | tuple[str, str, str, str]).validate_json(
        base64.urlsafe_b64decode(value)
    )
    return (parts[0], parts[1], parts[2], parts[3] if len(parts) == 4 else "")


def parameters(scope: Scope, filters: tuple[MetadataFilter, ...]) -> Mapping[str, object]:
    return MappingProxyType(
        {
            "all_teams": int(scope.all_teams),
            "team": scope.team_id,
            "key_hash": scope.api_key_hash,
            "filter_keys": tuple(f.key for f in filters),
            "filter_values": tuple(f.value for f in filters),
        }
    )


class SourceReader:
    def __init__(self, storage: Storage) -> None:
        self.storage: Final = storage

    async def sample(self, scope: Scope, settings: EngineSettings, start: int, end: int) -> Sample:
        params: Final = MappingProxyType(
            {
                **parameters(scope, settings.filters),
                "source": settings.source,
                "start": start,
                "end": end,
                "service": settings.service,
                "limit": settings.sample_size,
            }
        )
        rows: Final = _ROWS.validate_python(await self.storage.lens_sample(params))
        return Sample(
            eligible=rows[0].eligible if rows else 0,
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
                        if k != "litellm.api_key_hash" and 0 < len(k) <= 200 and 0 < len(v) <= 500
                    ),
                )
                for row in rows
            ),
        )

    async def content(self, scope: Scope, execution: Execution, cursor: str = "", offset: int = 0) -> ExecutionContent:
        params: Final = MappingProxyType(
            {
                **parameters(scope, ()),
                "source": execution.source,
                "id": execution.trace_id,
                "trace_ref": execution.trace_ref,
                "record_team": execution.team_id,
                "cursor": cursor,
                "offset": offset + 1,
            }
        )
        rows: Final = _PARTS.validate_python(await self.storage.lens_content(params))
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
        params: Final = MappingProxyType(
            {
                **parameters(scope, ()),
                "source": execution.source,
                "id": execution.trace_id,
                "trace_ref": execution.trace_ref,
                "record_team": execution.team_id,
                "span": evidence.span_id,
                "quote": evidence.quote,
            }
        )
        rows: Final = _COUNTS.validate_python(await self.storage.lens_evidence(params))
        return bool(rows and rows[0].count)
