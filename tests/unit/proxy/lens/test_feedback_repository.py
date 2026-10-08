import json
from datetime import datetime, timezone
from typing import Final

import httpx
import pytest

from litellm.proxy.lens.feedback_models import Feedback, TraceFeedback, TraceFeedbackSummary
from litellm.proxy.lens.feedback_repository import ClickHouseFeedbackStore
from litellm.proxy.lens.models import Scope, TraceIdentity
from litellm.rust_bridge.trace.storage import ClickHouseStorage
from litellm.tracing.remote import RemoteTraceStore

RATED_AT: Final = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
ACCESS: Final = (
    (Scope(team_id="team-a", api_key_hash="hash-a"), "all_teams=0 team=team-a key_hash=hash-a"),
    (Scope(team_id="team-a"), "all_teams=0 team=team-a key_hash="),
    (Scope(api_key_hash="hash-a"), "all_teams=0 team= key_hash=hash-a"),
    (Scope(all_teams=True), "all_teams=1 team= key_hash="),
    (Scope(all_teams=True, team_id="team-a", api_key_hash="hash-a"), "all_teams=1 team=team-a key_hash=hash-a"),
)


def lens_service(request: httpx.Request) -> httpx.Response:
    read: Final = json.loads(request.content)
    asked: Final = read["parameters"]
    access: Final = f"all_teams={asked['all_teams']} team={asked['team']} key_hash={asked['key_hash']}"
    match read["name"]:
        case "feedback_target":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "team_id": "stored-team",
                            "key_hash": "stored-key",
                            "trace_ref": f"{access} trace_id={asked['trace_id']} trace_ref={asked['trace_ref']}",
                        }
                    ]
                },
            )
        case "feedback":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "trace_id": asked["trace_id"],
                            "trace_ref": asked["trace_ref"],
                            "author": "ada",
                            "score": 7,
                            "comment": f"{access} trace_id={asked['trace_id']}",
                            "created_at": "2026-03-01T12:00:00.000+00:00",
                            "updated_at": "2026-03-01T12:00:00.000+00:00",
                        }
                    ]
                },
            )
        case "feedback_summary":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "trace_id": trace_id,
                            "trace_ref": f"{access} trace_ids={','.join(asked['trace_ids'])}",
                            "count": 2,
                            "average": 4.5,
                            "lowest": 3,
                        }
                        for trace_id in asked["trace_ids"]
                    ]
                },
            )
        case _:
            return httpx.Response(400)


def lens_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url="http://lens.test", transport=httpx.MockTransport(lens_service))


@pytest.mark.asyncio
@pytest.mark.parametrize(("scope", "access"), ACCESS)
async def test_trace_feedback_is_resolved_and_read_with_the_callers_access(scope: Scope, access: str) -> None:
    async with lens_client() as client:
        found: Final = await ClickHouseFeedbackStore(ClickHouseStorage(RemoteTraceStore(client))).for_trace(
            scope, TraceIdentity(trace_id="t1", trace_ref="REF")
        )

    resolved: Final = f"{access} trace_id=t1 trace_ref=REF"
    assert found == TraceFeedback(
        trace_id="t1",
        trace_ref=resolved,
        feedback=(
            Feedback(
                trace_id="t1",
                trace_ref=resolved,
                score=7,
                comment=f"{access} trace_id=t1",
                author="ada",
                created_at=RATED_AT,
                updated_at=RATED_AT,
            ),
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("scope", "access"), ACCESS)
async def test_feedback_summaries_are_read_once_per_distinct_trace_with_the_callers_access(
    scope: Scope, access: str
) -> None:
    async with lens_client() as client:
        found: Final = await ClickHouseFeedbackStore(ClickHouseStorage(RemoteTraceStore(client))).summaries(
            scope,
            (TraceIdentity(trace_id="t2"), TraceIdentity(trace_id="t1"), TraceIdentity(trace_id="t2")),
        )

    rated: Final = f"{access} trace_ids=t1,t2"
    assert found == (
        TraceFeedbackSummary(trace_id="t2", trace_ref=rated, count=2, average=4.5, lowest=3),
        TraceFeedbackSummary(trace_id="t1", trace_ref=rated, count=2, average=4.5, lowest=3),
        TraceFeedbackSummary(trace_id="t2", trace_ref=rated, count=2, average=4.5, lowest=3),
    )
