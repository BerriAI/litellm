from datetime import datetime, timezone
from types import MappingProxyType
from typing import Annotated, Final, TypeAlias

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import Field, model_validator

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.lens.endpoints import Auth, user_scope
from litellm.proxy.lens.feedback_repository import (
    ClickHouseFeedbackStore,
    FeedbackStore,
    FeedbackWrite,
    session_trace_id,
)
from litellm.proxy.lens.models import (
    Feedback,
    FeedbackInput,
    Record,
    TraceFeedback,
    TraceFeedbackRequest,
    TraceFeedbackSummary,
    TraceIdentity,
)
from litellm.proxy.tracing_runtime import provide_storage
from litellm.rust_bridge.trace.storage import ClickHouseStorage

router: Final = APIRouter(prefix="/lens/feedback", tags=["Lens"])
TRACE_NOT_FOUND: Final = "Trace not found"


class FeedbackTarget(Record):
    trace_id: str | None = Field(default=None, min_length=1, max_length=128)
    session_id: str | None = Field(default=None, min_length=1, max_length=512)
    trace_ref: str = Field(default="", max_length=512)

    @model_validator(mode="after")
    def one_target(self) -> "FeedbackTarget":
        if (self.trace_id is None) == (self.session_id is None):
            raise ValueError("Pass exactly one of trace_id or session_id")
        return self

    def trace(self) -> TraceIdentity:
        trace_id: Final = self.trace_id if self.trace_id is not None else session_trace_id(self.session_id or "")
        return TraceIdentity(trace_id=trace_id, trace_ref=self.trace_ref)


class FeedbackSubmission(FeedbackTarget, FeedbackInput):
    pass


def feedback_store(
    storage: Annotated[ClickHouseStorage | None, Depends(provide_storage)],
) -> FeedbackStore:
    if storage is None:
        raise HTTPException(
            501, "Lens feedback needs agent tracing. Set `tracing:` in general_settings and CLICKHOUSE_URL."
        )
    return ClickHouseFeedbackStore(storage)


def stored_now() -> datetime:
    now: Final = datetime.now(timezone.utc)
    return now.replace(microsecond=now.microsecond // 1000 * 1000)


def author(auth: UserAPIKeyAuth) -> str:
    identity: Final = auth.user_id or auth.token
    if not identity:
        raise HTTPException(403, "Feedback needs a key tied to a user")
    return identity


Store: TypeAlias = Annotated[FeedbackStore, Depends(feedback_store)]
Now: TypeAlias = Annotated[datetime, Depends(stored_now)]
Target: TypeAlias = Annotated[FeedbackTarget, Query()]


@router.get("", response_model=TraceFeedback)
async def read_feedback(target: Target, auth: Auth, store: Store) -> TraceFeedback:
    feedback: Final = await store.for_trace(user_scope(auth), target.trace())
    if feedback is None:
        raise HTTPException(404, TRACE_NOT_FOUND)
    return feedback.model_copy(update=MappingProxyType({"viewer": author(auth)}))


@router.put("", response_model=Feedback)
async def submit_feedback(body: FeedbackSubmission, auth: Auth, store: Store, now: Now) -> Feedback:
    scope: Final = user_scope(auth, write=True)
    saved: Final = await store.upsert(
        scope,
        FeedbackWrite(
            trace=body.trace(),
            author=author(auth),
            feedback=FeedbackInput(score=body.score, comment=body.comment),
            at=now,
        ),
    )
    if saved is None:
        raise HTTPException(404, TRACE_NOT_FOUND)
    return saved


@router.delete("", status_code=204)
async def delete_feedback(target: Target, auth: Auth, store: Store, now: Now) -> Response:
    scope: Final = user_scope(auth, write=True)
    deleted: Final = await store.delete(scope, target.trace(), author(auth), now)
    if deleted is None:
        raise HTTPException(404, TRACE_NOT_FOUND)
    if not deleted:
        raise HTTPException(404, "You have not left feedback on this trace")
    return Response(status_code=204)


@router.post("/summary", response_model=tuple[TraceFeedbackSummary, ...])
async def feedback_summary(body: TraceFeedbackRequest, auth: Auth, store: Store) -> tuple[TraceFeedbackSummary, ...]:
    return await store.summaries(user_scope(auth), body.traces)
