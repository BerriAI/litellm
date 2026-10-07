"""Response usefulness feedback. Trace authorization remains the source of truth."""

import hashlib
import json
import secrets
import time
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Annotated, Final, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import Field, TypeAdapter

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.lens.feedback_store import FeedbackStore
from litellm.proxy.lens.models import Record
from litellm.proxy.tracing_endpoints import TraceAccessContext, provide_trace_access, read_failure
from litellm.proxy.tracing_runtime import provide_storage
from litellm.rust_bridge.trace.errors import TraceChanged
from litellm.rust_bridge.trace.storage import ClickHouseStorage

router: Final = APIRouter(prefix="/v1/feedback", tags=["Lens feedback"])

Auth = Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]
Access = Annotated[TraceAccessContext, Depends(provide_trace_access)]


class FeedbackTarget(Record):
    trace_id: str = Field(min_length=1, max_length=128)
    trace_ref: str = Field(default="", max_length=512)
    span_id: str = Field(default="", max_length=128)


class FeedbackCreate(FeedbackTarget):
    key: Literal["usefulness"] = "usefulness"
    value: int = Field(strict=True, ge=0, le=10)
    comment: str = Field(default="", max_length=4000)


class FeedbackUpdate(Record):
    value: int = Field(strict=True, ge=0, le=10)
    comment: str = Field(default="", max_length=4000)


class Feedback(FeedbackCreate):
    id: str
    created_at: datetime
    updated_at: datetime


class FeedbackSummary(Record):
    key: Literal["usefulness"] = "usefulness"
    count: int
    average: float | None
    distribution: dict[str, int]
    mine: Feedback | None = None
    can_rate: bool = True


class FeedbackPage(Record):
    data: tuple[Feedback, ...]
    next_offset: int | None = None


def author(auth: UserAPIKeyAuth, *, write: bool = False) -> str:
    if write and auth.user_role in (
        LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
        LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
    ):
        raise HTTPException(403, "Read-only users cannot submit feedback")
    identity: Final = f"user:{auth.user_id}" if auth.user_id else f"key:{auth.token}" if auth.token else ""
    if not identity:
        raise HTTPException(403, "Feedback requires an authenticated user or key")
    return hashlib.sha256(identity.encode()).hexdigest()


async def target_access(target: FeedbackTarget, context: TraceAccessContext) -> FeedbackTarget:
    tracing, scope = context.reader()
    try:
        trace: Final = await tracing.get_trace(target.trace_id, scope, target.trace_ref, page_size=1)
        if trace is None:
            raise HTTPException(404, "Trace not found; retry after trace ingestion completes")
        reference: Final = trace["summary"].get("trace_ref", "")
        if not reference:
            raise HTTPException(409, "Trace identity is unresolved; refresh the trace and retry")
        if target.span_id and await tracing.get_span(target.trace_id, target.span_id, scope, reference) is None:
            raise HTTPException(404, "Span not found in this trace")
    except (TraceChanged, ValueError, OverflowError, RuntimeError) as error:
        raise read_failure(error) from error
    return FeedbackTarget(trace_id=target.trace_id, trace_ref=reference, span_id=target.span_id)


class StoredFeedback(Record):
    payload: str
    author_id: str
    version: int
    deleted: int = 0


_TARGET_SQL: Final = "trace_id={trace_id:String} AND trace_ref={trace_ref:String} AND span_id={span_id:String}"


def target_params(target: FeedbackTarget) -> dict[str, str]:
    return {"trace_id": target.trace_id, "trace_ref": target.trace_ref, "span_id": target.span_id}


def feedback_id(target: FeedbackTarget, owner: str) -> str:
    return hashlib.sha256(json.dumps((*target_params(target).values(), owner)).encode()).hexdigest()


class FeedbackRepository:
    def __init__(self, store: FeedbackStore) -> None:
        self.store: Final = store

    async def stored(self, identity: str, include_deleted: bool = False) -> StoredFeedback | None:
        rows: Final = await self.store.rows(
            "SELECT payload, author_id, version, deleted FROM lens_feedback FINAL WHERE id={id:String} LIMIT 1",
            {"id": identity},
        )
        record: Final = StoredFeedback.model_validate(rows[0]) if rows else None
        return record if record is not None and (include_deleted or not record.deleted) else None

    async def append(
        self, feedback: Feedback, owner: str, previous: StoredFeedback | None, deleted: bool = False
    ) -> None:
        version: Final = max((time.time_ns() << 64) | secrets.randbits(64), previous.version + 1 if previous else 0)
        await self.store.insert(
            {
                "id": feedback.id,
                **target_params(feedback),
                "author_id": owner,
                "value": feedback.value,
                "payload": feedback.model_dump_json(),
                "version": version,
                "deleted": int(deleted),
            }
        )

    async def save(self, body: FeedbackCreate, target: FeedbackTarget, owner: str) -> Feedback:
        identity: Final = feedback_id(target, owner)
        previous: Final = await self.stored(identity, include_deleted=True)
        now: Final = datetime.now(timezone.utc)
        feedback: Final = Feedback(
            **target_params(target),
            key="usefulness",
            value=body.value,
            comment=body.comment,
            id=identity,
            created_at=Feedback.model_validate_json(previous.payload).created_at if previous else now,
            updated_at=now,
        )
        await self.append(feedback, owner, previous)
        return feedback

    async def get(self, identity: str) -> Feedback:
        stored: Final = await self.stored(identity)
        if stored is None:
            raise HTTPException(404, "Feedback not found")
        return Feedback.model_validate_json(stored.payload)

    async def page(self, target: FeedbackTarget, limit: int, offset: int) -> FeedbackPage:
        rows: Final = await self.store.rows(
            "SELECT payload FROM lens_feedback FINAL WHERE "
            + _TARGET_SQL
            + " AND deleted=0 ORDER BY id LIMIT {limit:UInt32} OFFSET {offset:UInt64}",
            {**target_params(target), "limit": limit + 1, "offset": offset},
        )
        records: Final = tuple(
            Feedback.model_validate_json(TypeAdapter(str).validate_python(row["payload"])) for row in rows
        )
        return FeedbackPage(data=records[:limit], next_offset=offset + limit if len(records) > limit else None)

    async def summary(self, target: FeedbackTarget, owner: str) -> FeedbackSummary:
        rows: Final = await self.store.rows(
            "SELECT value, count() AS count FROM lens_feedback FINAL WHERE "
            + _TARGET_SQL
            + " AND deleted=0 GROUP BY value ORDER BY value",
            target_params(target),
        )
        buckets: Final = {str(row["value"]): TypeAdapter(int).validate_python(row["count"]) for row in rows}
        count: Final = sum(buckets.values())
        mine: Final = await self.stored(feedback_id(target, owner))
        return FeedbackSummary(
            count=count,
            average=sum(int(value) * n for value, n in buckets.items()) / count if count else None,
            distribution={str(value): buckets.get(str(value), 0) for value in range(11)},
            mine=Feedback.model_validate_json(mine.payload) if mine else None,
        )

    async def owned(self, identity: str, owner: str) -> StoredFeedback:
        previous: Final = await self.stored(identity)
        if previous is None:
            raise HTTPException(404, "Feedback not found")
        if previous.author_id != owner:
            raise HTTPException(403, "Only the feedback author can change this rating")
        return previous

    async def update(self, identity: str, owner: str, body: FeedbackUpdate) -> Feedback:
        previous: Final = await self.owned(identity, owner)
        feedback: Final = Feedback.model_validate_json(previous.payload).model_copy(
            update={
                "value": body.value,
                "comment": body.comment,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        await self.append(feedback, owner, previous)
        return feedback

    async def delete(self, identity: str, owner: str) -> None:
        previous: Final = await self.owned(identity, owner)
        await self.append(Feedback.model_validate_json(previous.payload), owner, previous, deleted=True)


async def feedback_repository(
    storage: Annotated[ClickHouseStorage | None, Depends(provide_storage)],
) -> AsyncIterator[FeedbackRepository]:
    if storage is None:
        raise HTTPException(503, "ClickHouse feedback storage is unavailable")
    try:
        yield FeedbackRepository(storage.feedback)
    except RuntimeError as error:
        raise HTTPException(503, "ClickHouse feedback storage is unavailable", headers={"Retry-After": "1"}) from error


Repo = Annotated[FeedbackRepository, Depends(feedback_repository)]


@router.post("", response_model=Feedback)
async def create_feedback(body: FeedbackCreate, auth: Auth, context: Access, repo: Repo) -> Feedback:
    """Create or replace this authenticated author's usefulness rating for one trace/span."""
    owner: Final = author(auth, write=True)
    target: Final = await target_access(body, context)
    return await repo.save(body, target, owner)


@router.get("", response_model=FeedbackPage)
async def list_feedback(
    auth: Auth,
    context: Access,
    repo: Repo,
    trace_id: str = Query(min_length=1, max_length=128),
    trace_ref: str = Query(default="", max_length=512),
    span_id: str = Query(default="", max_length=128),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> FeedbackPage:
    target: Final = await target_access(
        FeedbackTarget(trace_id=trace_id, trace_ref=trace_ref, span_id=span_id), context
    )
    return await repo.page(target, limit, offset)


@router.get("/summary", response_model=FeedbackSummary)
async def summarize_feedback(
    auth: Auth,
    context: Access,
    repo: Repo,
    target: Annotated[FeedbackTarget, Query()],
) -> FeedbackSummary:
    summary: Final = await repo.summary(await target_access(target, context), author(auth))
    return summary.model_copy(
        update={
            "can_rate": auth.user_role
            not in (
                LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
                LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
            )
        }
    )


@router.get("/{feedback_id}", response_model=Feedback)
async def read_feedback(feedback_id: str, auth: Auth, context: Access, repo: Repo) -> Feedback:
    feedback: Final = await repo.get(feedback_id)
    await target_access(feedback, context)
    return feedback


@router.patch("/{feedback_id}", response_model=Feedback)
async def update_feedback(
    feedback_id: str,
    body: FeedbackUpdate,
    auth: Auth,
    context: Access,
    repo: Repo,
) -> Feedback:
    owner: Final = author(auth, write=True)
    feedback: Final = await repo.get(feedback_id)
    await target_access(feedback, context)
    return await repo.update(feedback_id, owner, body)


@router.delete("/{feedback_id}", status_code=204)
async def delete_feedback(feedback_id: str, auth: Auth, context: Access, repo: Repo) -> Response:
    owner: Final = author(auth, write=True)
    feedback: Final = await repo.get(feedback_id)
    await target_access(feedback, context)
    await repo.delete(feedback_id, owner)
    return Response(status_code=204)
