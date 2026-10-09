import hashlib
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Final

import pytest
from fastapi import HTTPException
from pydantic import BaseModel, ValidationError

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.lens.feedback_endpoints import (
    FeedbackDeletion,
    FeedbackSubmission,
    FeedbackTarget,
    delete_feedback,
    feedback_summary,
    read_feedback,
    submit_feedback,
)
from litellm.proxy.lens.feedback_models import TraceFeedbackRequest
from litellm.proxy.lens.feedback_repository import FEEDBACK_TABLE, ClickHouseFeedbackStore, session_trace_id
from litellm.proxy.lens.models import TraceIdentity
from litellm.rust_bridge.trace.generated.models import (
    FeedbackRow,
    FeedbackSummaryRow,
    FeedbackTargetRow,
    LensFeedbackParams,
    LensFeedbackSummaryParams,
    LensFeedbackTargetParams,
)
from litellm.rust_bridge.trace.queries import ReadQuery
from litellm.rust_bridge.trace.storage import ClickHouseStorage

ADMIN: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin")
OTHER_ADMIN: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="other")
VIEWER: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, user_id="viewer")
INTERNAL: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id="dev")
TEAM_APP: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, team_id="team-a", token="app-key")
SOLO_APP: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, token="key-solo")
T0: Final = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)


def ref(team: str, key: str, trace: str) -> str:
    return hashlib.sha256(f"{team}\0{key}\0{trace}".encode()).hexdigest().upper()


class FakeClickHouse(ClickHouseStorage):
    """Mirrors lens_feedback: ReplacingMergeTree(UpdatedAt, IsDeleted) keyed by team, key, trace, author."""

    def __init__(self, traces: Mapping[str, tuple[tuple[str, str], ...]]) -> None:
        self.traces: Final = traces
        self.rows: Final[list[Mapping[str, object]]] = []  # mutable-ok: stands in for the table

    async def insert_rows(self, table: str, rows: Sequence[Mapping[str, object]]) -> None:
        assert table == FEEDBACK_TABLE
        self.rows.extend(rows)

    def _visible(
        self, team: str, key: str, access: LensFeedbackTargetParams | LensFeedbackParams | LensFeedbackSummaryParams
    ) -> bool:
        if access.all_teams:
            return True
        return team == access.team and (access.key_hash == "" or key == access.key_hash)

    def _latest(self) -> tuple[Mapping[str, object], ...]:
        newest: Final = {}  # mutable-ok: emulates FINAL collapse
        for row in sorted(self.rows, key=lambda r: str(r["UpdatedAt"])):
            newest[(row["TeamId"], row["ApiKeyHash"], row["TraceId"], row["Author"])] = row
        return tuple(r for r in newest.values() if r["IsDeleted"] == 0)

    async def query(self, query: ReadQuery[BaseModel, object], parameters: BaseModel) -> tuple[object, ...]:  # pyright: ignore[reportIncompatibleMethodOverride]  # fake dispatches on the concrete params type
        match parameters:
            case LensFeedbackTargetParams():
                return tuple(
                    FeedbackTargetRow(team_id=team, key_hash=key, trace_ref=ref(team, key, parameters.trace_id))
                    for team, key in self.traces.get(parameters.trace_id, ())
                    if self._visible(team, key, parameters)
                    and parameters.trace_ref in ("", ref(team, key, parameters.trace_id))
                )
            case LensFeedbackParams():
                return tuple(
                    FeedbackRow(
                        trace_id=str(r["TraceId"]),
                        trace_ref=ref(str(r["TeamId"]), str(r["ApiKeyHash"]), str(r["TraceId"])),
                        author=str(r["Author"]),
                        score=int(str(r["Score"])),
                        comment=str(r["Comment"]),
                        created_at=str(r["CreatedAt"]),
                        updated_at=str(r["UpdatedAt"]),
                    )
                    for r in self._latest()
                    if r["TraceId"] == parameters.trace_id
                    and ref(str(r["TeamId"]), str(r["ApiKeyHash"]), str(r["TraceId"])) == parameters.trace_ref
                    and self._visible(str(r["TeamId"]), str(r["ApiKeyHash"]), parameters)
                )
            case LensFeedbackSummaryParams():
                live: Final = tuple(
                    r
                    for r in self._latest()
                    if r["TraceId"] in parameters.trace_ids
                    and self._visible(str(r["TeamId"]), str(r["ApiKeyHash"]), parameters)
                )
                keys: Final = sorted({(str(r["TeamId"]), str(r["ApiKeyHash"]), str(r["TraceId"])) for r in live})
                return tuple(_summary_row(live, team, key, trace) for team, key, trace in keys)
            case _:
                raise AssertionError(f"unexpected query {query.name}")


def _summary_row(live: tuple[Mapping[str, object], ...], team: str, key: str, trace: str) -> FeedbackSummaryRow:
    scores: Final = tuple(
        int(str(r["Score"])) for r in live if (r["TeamId"], r["ApiKeyHash"], r["TraceId"]) == (team, key, trace)
    )
    return FeedbackSummaryRow(
        trace_id=trace,
        trace_ref=ref(team, key, trace),
        count=len(scores),
        average=sum(scores) / len(scores),
        lowest=min(scores),
    )


def store(**traces: tuple[tuple[str, str], ...]) -> ClickHouseFeedbackStore:
    return ClickHouseFeedbackStore(FakeClickHouse(traces or {"t1": (("team-a", "key-a"),)}))


def submission(score: int, comment: str = "", **fields: str) -> FeedbackSubmission:
    target: Final = {} if "trace_id" in fields or "session_id" in fields else {"trace_id": "t1"}
    return FeedbackSubmission.model_validate({"score": score, "comment": comment, **target, **fields})


@pytest.mark.asyncio
async def test_resubmitting_replaces_the_authors_feedback_and_keeps_other_authors() -> None:
    feedback: Final = store()
    first: Final = await submit_feedback(submission(3, "wrong file"), ADMIN, feedback, T0)
    await submit_feedback(submission(9, "great"), OTHER_ADMIN, feedback, T0)
    second: Final = await submit_feedback(
        submission(7, "fixed after retry"), ADMIN, feedback, T0 + timedelta(minutes=5)
    )

    listed: Final = await read_feedback(FeedbackTarget(trace_id="t1"), VIEWER, feedback)

    assert (first.created_at, second.created_at, second.updated_at) == (T0, T0, T0 + timedelta(minutes=5))
    assert {f.author: f.created_at for f in listed.feedback}["admin"] == T0
    assert listed.trace_ref == ref("team-a", "key-a", "t1")
    assert {(f.author, f.score, f.comment) for f in listed.feedback} == {
        ("admin", 7, "fixed after retry"),
        ("other", 9, "great"),
    }


@pytest.mark.asyncio
async def test_session_id_resolves_to_the_trace_lens_derives_at_ingest() -> None:
    session_trace: Final = session_trace_id("session-one")
    feedback: Final = store(**{session_trace: (("team-a", "key-a"),)})

    saved: Final = await submit_feedback(submission(4, session_id="session-one"), ADMIN, feedback, T0)
    listed: Final = await read_feedback(FeedbackTarget(session_id="session-one"), ADMIN, feedback)

    assert saved.trace_id == session_trace
    assert listed.trace_ref == ref("team-a", "key-a", session_trace)
    assert [f.score for f in listed.feedback] == [4]


def test_session_trace_id_matches_the_rust_ingest_hash() -> None:
    # Pinned in litellm-rust/crates/traces/tests/otlp.rs (session_capture_joins_native_logs_...).
    assert session_trace_id("session-one") == "5fddf060372c8501dca4f331b9da882b"


@pytest.mark.asyncio
async def test_unknown_traces_are_not_found_and_write_nothing() -> None:
    feedback: Final = store()

    with pytest.raises(HTTPException) as write:
        await submit_feedback(submission(5, trace_id="missing"), ADMIN, feedback, T0)
    with pytest.raises(HTTPException) as read:
        await read_feedback(FeedbackTarget(trace_id="missing"), ADMIN, feedback)

    assert (write.value.status_code, read.value.status_code) == (404, 404)
    assert isinstance(feedback.storage, FakeClickHouse) and feedback.storage.rows == []


@pytest.mark.parametrize("score", (-1, 11))
def test_scores_outside_zero_to_ten_are_rejected(score: int) -> None:
    with pytest.raises(ValidationError):
        submission(score)


@pytest.mark.parametrize("target", ({}, {"trace_id": "t1", "session_id": "s1"}))
def test_target_needs_exactly_one_of_trace_or_session(target: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        FeedbackTarget.model_validate(target)


@pytest.mark.asyncio
async def test_viewers_can_read_but_not_write_and_non_admins_cannot_read_in_lens() -> None:
    feedback: Final = store()
    await read_feedback(FeedbackTarget(trace_id="t1"), VIEWER, feedback)

    with pytest.raises(HTTPException) as write:
        await submit_feedback(submission(5), VIEWER, feedback, T0)
    with pytest.raises(HTTPException) as read:
        await read_feedback(FeedbackTarget(trace_id="t1"), INTERNAL, feedback)

    assert (write.value.status_code, read.value.status_code) == (403, 403)


@pytest.mark.asyncio
async def test_a_caller_without_a_team_or_key_cannot_write_on_a_teamless_trace() -> None:
    feedback: Final = store(t1=(("", "key-a"),))
    await submit_feedback(submission(9, "mine", user="customer-1"), ADMIN, feedback, T0)

    with pytest.raises(HTTPException) as write:
        await submit_feedback(submission(1, "overwrite", user="customer-1"), INTERNAL, feedback, T0)
    with pytest.raises(HTTPException) as delete:
        await delete_feedback(FeedbackDeletion(trace_id="t1", user="customer-1"), INTERNAL, feedback, T0)

    assert (write.value.status_code, delete.value.status_code) == (403, 403)
    assert [f.score for f in (await read_feedback(FeedbackTarget(trace_id="t1"), ADMIN, feedback)).feedback] == [9]


@pytest.mark.asyncio
async def test_tenant_comes_from_the_trace_and_author_defaults_to_the_caller() -> None:
    feedback: Final = store()
    saved: Final = await submit_feedback(submission(5), OTHER_ADMIN, feedback, T0)

    assert saved.author == "other"
    assert isinstance(feedback.storage, FakeClickHouse)
    assert {(r["TeamId"], r["ApiKeyHash"], r["Author"]) for r in feedback.storage.rows} == {
        ("team-a", "key-a", "other")
    }
    with pytest.raises(ValidationError):
        FeedbackSubmission.model_validate({"trace_id": "t1", "score": 5, "team_id": "someone-else"})


@pytest.mark.asyncio
async def test_delete_hides_only_the_callers_feedback() -> None:
    feedback: Final = store()
    await submit_feedback(submission(2), ADMIN, feedback, T0)
    await submit_feedback(submission(8), OTHER_ADMIN, feedback, T0)

    await delete_feedback(FeedbackDeletion(trace_id="t1"), ADMIN, feedback, T0 + timedelta(minutes=9))
    with pytest.raises(HTTPException) as missing:
        await delete_feedback(FeedbackDeletion(trace_id="t1"), ADMIN, feedback, T0 + timedelta(minutes=9))

    remaining: Final = await read_feedback(FeedbackTarget(trace_id="t1"), ADMIN, feedback)
    assert missing.value.status_code == 404
    assert [f.author for f in remaining.feedback] == ["other"]


@pytest.mark.asyncio
async def test_summary_flags_rated_traces_and_leaves_unrated_ones_empty() -> None:
    feedback: Final = store(t1=(("team-a", "key-a"),), t2=(("team-a", "key-a"),))
    await submit_feedback(submission(2), ADMIN, feedback, T0)
    await submit_feedback(submission(8), OTHER_ADMIN, feedback, T0)
    rated: Final = TraceIdentity(trace_id="t1", trace_ref=ref("team-a", "key-a", "t1"))
    unrated: Final = TraceIdentity(trace_id="t2", trace_ref=ref("team-a", "key-a", "t2"))

    summaries: Final = await feedback_summary(TraceFeedbackRequest(traces=(rated, unrated)), VIEWER, feedback)

    assert {s.trace_id: (s.count, s.average, s.lowest) for s in summaries} == {
        "t1": (2, 5.0, 2),
        "t2": (0, None, None),
    }


@pytest.mark.asyncio
async def test_a_trace_id_shared_by_two_keys_needs_its_trace_ref() -> None:
    feedback: Final = store(t1=(("team-a", "key-a"), ("team-a", "key-b")))

    with pytest.raises(HTTPException) as ambiguous:
        await submit_feedback(submission(5), ADMIN, feedback, T0)
    saved: Final = await submit_feedback(
        submission(5, trace_id="t1", trace_ref=ref("team-a", "key-b", "t1")), ADMIN, feedback, T0
    )

    assert ambiguous.value.status_code == 404
    assert saved.trace_ref == ref("team-a", "key-b", "t1")


@pytest.mark.asyncio
async def test_summary_without_trace_ref_reports_the_rated_trace_with_its_resolved_ref() -> None:
    feedback: Final = store()
    await submit_feedback(submission(6), ADMIN, feedback, T0)

    summaries: Final = await feedback_summary(
        TraceFeedbackRequest(traces=(TraceIdentity(trace_id="t1"),)), VIEWER, feedback
    )

    assert [(s.trace_ref, s.count, s.lowest) for s in summaries] == [(ref("team-a", "key-a", "t1"), 1, 6)]


@pytest.mark.asyncio
async def test_an_app_key_records_its_end_users_feedback_on_its_own_teams_trace() -> None:
    feedback: Final = store()
    await submit_feedback(submission(2, "It ignored my file", user="customer-1"), TEAM_APP, feedback, T0)
    await submit_feedback(submission(9, "Perfect", user="customer-2"), TEAM_APP, feedback, T0)
    await submit_feedback(
        submission(4, "Better after retry", user="customer-1"), TEAM_APP, feedback, T0 + timedelta(minutes=1)
    )

    listed: Final = await read_feedback(FeedbackTarget(trace_id="t1"), ADMIN, feedback)

    assert {(f.author, f.score, f.comment) for f in listed.feedback} == {
        ("customer-1", 4, "Better after retry"),
        ("customer-2", 9, "Perfect"),
    }


@pytest.mark.asyncio
async def test_an_app_key_cannot_write_feedback_on_another_tenants_trace() -> None:
    feedback: Final = store(t1=(("team-b", "key-b"),), solo=(("", "key-solo"),), other=(("", "key-other"),))

    with pytest.raises(HTTPException) as other_team:
        await submit_feedback(submission(5, user="customer-1"), TEAM_APP, feedback, T0)
    with pytest.raises(HTTPException) as other_key:
        await submit_feedback(submission(5, trace_id="other", user="customer-1"), SOLO_APP, feedback, T0)
    saved: Final = await submit_feedback(submission(5, trace_id="solo", user="customer-1"), SOLO_APP, feedback, T0)

    assert (other_team.value.status_code, other_key.value.status_code) == (404, 404)
    assert saved.author == "customer-1"


@pytest.mark.asyncio
async def test_an_app_can_remove_one_end_users_feedback() -> None:
    feedback: Final = store()
    await submit_feedback(submission(2, user="customer-1"), TEAM_APP, feedback, T0)
    await submit_feedback(submission(9, user="customer-2"), TEAM_APP, feedback, T0)

    await delete_feedback(
        FeedbackDeletion(trace_id="t1", user="customer-1"), TEAM_APP, feedback, T0 + timedelta(minutes=1)
    )

    listed: Final = await read_feedback(FeedbackTarget(trace_id="t1"), ADMIN, feedback)
    assert [f.author for f in listed.feedback] == ["customer-2"]
