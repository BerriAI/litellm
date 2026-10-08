import hashlib
from collections.abc import Mapping
from datetime import datetime, timezone
from itertools import chain
from typing import Final, Protocol

from litellm.proxy.lens.feedback_models import Feedback, FeedbackInput, TraceFeedback, TraceFeedbackSummary
from litellm.proxy.lens.models import Record, Scope, TraceIdentity
from litellm.proxy.lens.sources import access_parameters
from litellm.rust_bridge.trace.generated.models import (
    FeedbackRow,
    FeedbackSummaryRow,
    LensFeedbackParams,
    LensFeedbackSummaryParams,
    LensFeedbackTargetParams,
)
from litellm.rust_bridge.trace.queries import LENS_FEEDBACK, LENS_FEEDBACK_SUMMARY, LENS_FEEDBACK_TARGET
from litellm.rust_bridge.trace.storage import ClickHouseStorage

FEEDBACK_TABLE: Final = "lens_feedback"


def session_trace_id(session_id: str) -> str:
    return hashlib.sha256(f"litellm.claude.session.v1\0{session_id}".encode()).digest()[:16].hex()


class FeedbackWrite(Record):
    trace: TraceIdentity
    author: str
    feedback: FeedbackInput
    at: datetime


class FeedbackStore(Protocol):
    async def for_trace(self, scope: Scope, trace: TraceIdentity) -> TraceFeedback | None: ...
    async def upsert(self, scope: Scope, write: FeedbackWrite) -> Feedback | None: ...
    async def delete(self, scope: Scope, trace: TraceIdentity, author: str, at: datetime) -> bool | None: ...
    async def summaries(self, scope: Scope, traces: tuple[TraceIdentity, ...]) -> tuple[TraceFeedbackSummary, ...]: ...


class _Target(Record):
    trace_id: str
    trace_ref: str
    team_id: str
    key_hash: str


def _iso(at: datetime) -> str:
    return at.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _feedback(row: FeedbackRow) -> Feedback:
    return Feedback(
        trace_id=row.trace_id,
        trace_ref=row.trace_ref,
        score=row.score,
        comment=row.comment,
        author=row.author,
        created_at=datetime.fromisoformat(row.created_at),
        updated_at=datetime.fromisoformat(row.updated_at),
    )


class ClickHouseFeedbackStore:
    def __init__(self, storage: ClickHouseStorage) -> None:
        self.storage: Final = storage

    async def _target(self, scope: Scope, trace: TraceIdentity) -> _Target | None:
        rows: Final = await self.storage.query(
            LENS_FEEDBACK_TARGET,
            LensFeedbackTargetParams(
                **access_parameters(scope).model_dump(), trace_id=trace.trace_id, trace_ref=trace.trace_ref
            ),
        )
        if len(rows) != 1:
            return None
        return _Target(
            trace_id=trace.trace_id, trace_ref=rows[0].trace_ref, team_id=rows[0].team_id, key_hash=rows[0].key_hash
        )

    async def _rows(self, scope: Scope, trace: TraceIdentity) -> tuple[Feedback, ...]:
        rows: Final = await self.storage.query(
            LENS_FEEDBACK,
            LensFeedbackParams(
                **access_parameters(scope).model_dump(), trace_id=trace.trace_id, trace_ref=trace.trace_ref
            ),
        )
        return tuple(_feedback(row) for row in rows)

    async def for_trace(self, scope: Scope, trace: TraceIdentity) -> TraceFeedback | None:
        target: Final = await self._target(scope, trace)
        if target is None:
            return None
        identity: Final = TraceIdentity(trace_id=target.trace_id, trace_ref=target.trace_ref)
        return TraceFeedback(
            trace_id=identity.trace_id, trace_ref=identity.trace_ref, feedback=await self._rows(scope, identity)
        )

    async def _write(self, target: _Target, author: str, values: Mapping[str, object]) -> None:
        await self.storage.insert_rows(
            FEEDBACK_TABLE,
            (
                {
                    "TeamId": target.team_id,
                    "ApiKeyHash": target.key_hash,
                    "TraceId": target.trace_id,
                    "Author": author,
                    **values,
                },
            ),
        )

    async def upsert(self, scope: Scope, write: FeedbackWrite) -> Feedback | None:
        target: Final = await self._target(scope, write.trace)
        if target is None:
            return None
        identity: Final = TraceIdentity(trace_id=target.trace_id, trace_ref=target.trace_ref)
        previous: Final = next((f for f in await self._rows(scope, identity) if f.author == write.author), None)
        created: Final = previous.created_at if previous else write.at
        await self._write(
            target,
            write.author,
            {
                "Score": write.feedback.score,
                "Comment": write.feedback.comment,
                "CreatedAt": _iso(created),
                "UpdatedAt": _iso(write.at),
                "IsDeleted": 0,
            },
        )
        return Feedback(
            trace_id=target.trace_id,
            trace_ref=target.trace_ref,
            score=write.feedback.score,
            comment=write.feedback.comment,
            author=write.author,
            created_at=created,
            updated_at=write.at,
        )

    async def delete(self, scope: Scope, trace: TraceIdentity, author: str, at: datetime) -> bool | None:
        target: Final = await self._target(scope, trace)
        if target is None:
            return None
        identity: Final = TraceIdentity(trace_id=target.trace_id, trace_ref=target.trace_ref)
        previous: Final = next((f for f in await self._rows(scope, identity) if f.author == author), None)
        if previous is None:
            return False
        await self._write(
            target,
            author,
            {
                "Score": previous.score,
                "Comment": "",
                "CreatedAt": _iso(previous.created_at),
                "UpdatedAt": _iso(at),
                "IsDeleted": 1,
            },
        )
        return True

    async def summaries(self, scope: Scope, traces: tuple[TraceIdentity, ...]) -> tuple[TraceFeedbackSummary, ...]:
        rows: Final = await self.storage.query(
            LENS_FEEDBACK_SUMMARY,
            LensFeedbackSummaryParams(
                **access_parameters(scope).model_dump(), trace_ids=sorted({t.trace_id for t in traces})
            ),
        )
        return tuple(chain.from_iterable(_summaries(trace, rows) for trace in traces))


def _summaries(trace: TraceIdentity, rows: tuple[FeedbackSummaryRow, ...]) -> tuple[TraceFeedbackSummary, ...]:
    matched: Final = tuple(
        row for row in rows if row.trace_id == trace.trace_id and trace.trace_ref in ("", row.trace_ref)
    )
    if not matched:
        return (
            TraceFeedbackSummary(
                trace_id=trace.trace_id, trace_ref=trace.trace_ref, count=0, average=None, lowest=None
            ),
        )
    return tuple(
        TraceFeedbackSummary(
            trace_id=row.trace_id, trace_ref=row.trace_ref, count=row.count, average=row.average, lowest=row.lowest
        )
        for row in matched
    )
