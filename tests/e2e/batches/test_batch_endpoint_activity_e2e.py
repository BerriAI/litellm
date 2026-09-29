"""Team Usage > Endpoint Activity for a completed Vertex managed batch.

A unified (target_model_names) upload plus create against a real Vertex
deployment, driven to completion, lets the CheckBatchCost poller book the batch
cost. The priced row must then appear under the `/batches` endpoint in
GET /team/daily/activity, the response the Admin UI's Endpoint Activity tab
renders, with the same tokens and spend the spend log row carries.

Needs a proxy with the deployment below in its config, a real GCS bucket, and a
short PROXY_BATCH_POLLING_INTERVAL; the poll budget covers Vertex's queue time.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from math import isclose
from typing import Final

import pytest
from batch_client import BatchClient, BatchCreateBody, BatchObject, FileObject
from e2e_config import unique_marker
from e2e_http import FileUploadForm, require_successful_call, unwrap
from lifecycle import ResourceManager
from models import KeyGenerateBody, TeamNewBody
from pydantic import BaseModel
from test_batches_e2e import render_jsonl

pytestmark = pytest.mark.e2e

VERTEX_US_MODEL: Final = os.environ.get("E2E_VERTEX_US_BATCH_MODEL", "gemini-3.5-flash-llmservices-us")
COMPLETION_BUDGET_SECONDS: Final = float(os.environ.get("E2E_BATCH_COMPLETION_BUDGET_SECONDS", "3600"))
POLL_SECONDS: Final = 30.0
BATCHES_ENDPOINT: Final = "/batches"


class ActivityMetrics(BaseModel):
    spend: float = 0.0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    api_requests: int = 0
    successful_requests: int = 0


class ActivityEntry(BaseModel):
    metrics: ActivityMetrics


class ActivityBreakdown(BaseModel):
    endpoints: dict[str, ActivityEntry] = {}


class ActivityRow(BaseModel):
    date: str
    metrics: ActivityMetrics
    breakdown: ActivityBreakdown


class ActivityMetadata(BaseModel):
    total_spend: float = 0.0
    total_tokens: int = 0


class ActivityResponse(BaseModel):
    results: list[ActivityRow] = []
    metadata: ActivityMetadata


class TeamActivityParams(BaseModel):
    team_ids: str
    start_date: str
    end_date: str
    page_size: int = 100


def _team_activity(client: BatchClient, team_id: str) -> ActivityResponse:
    today: Final = datetime.now(timezone.utc).date()
    return unwrap(
        client.proxy.transport.get(
            "/team/daily/activity",
            headers=client.proxy.management_headers(),
            params=TeamActivityParams(
                team_ids=team_id,
                start_date=(today - timedelta(days=1)).isoformat(),
                end_date=(today + timedelta(days=1)).isoformat(),
            ),
            response_type=ActivityResponse,
        )
    )


def _endpoint_totals(activity: ActivityResponse, endpoint: str) -> ActivityMetrics:
    entries: Final = [
        row.breakdown.endpoints[endpoint].metrics for row in activity.results if endpoint in row.breakdown.endpoints
    ]
    return ActivityMetrics(
        spend=sum(m.spend for m in entries),
        total_tokens=sum(m.total_tokens for m in entries),
        prompt_tokens=sum(m.prompt_tokens for m in entries),
        completion_tokens=sum(m.completion_tokens for m in entries),
        api_requests=sum(m.api_requests for m in entries),
        successful_requests=sum(m.successful_requests for m in entries),
    )


def _await_completed(client: BatchClient, batch_id: str, key: str) -> BatchObject:
    deadline: Final = time.monotonic() + COMPLETION_BUDGET_SECONDS
    listed: BatchObject | None = None
    while time.monotonic() < deadline:
        page = unwrap(client.list_batches(key=key, limit=20))
        listed = next((b for b in page.data if b.id == batch_id), None)
        if listed is not None and listed.status in {"completed", "failed", "expired", "cancelled"}:
            return listed
        time.sleep(POLL_SECONDS)
    pytest.fail(f"batch {batch_id} not terminal within {COMPLETION_BUDGET_SECONDS}s: {listed!r}")


@pytest.mark.covers("llm.batches.vertex.endpoint_activity.nonstream.works", exercised_on=["batches", "files"])
def test_completed_vertex_batch_cost_lands_under_batches_endpoint_activity(
    client: BatchClient, resources: ResourceManager
) -> None:
    team_id: Final = client.proxy.create_team(TeamNewBody(team_alias=f"e2e-batch-activity-{unique_marker()}"))
    resources.defer(lambda: client.proxy.delete_team(team_id))
    key: Final = client.proxy.generate_key(KeyGenerateBody(team_id=team_id, user_id="e2e-test-user"))
    resources.defer(lambda: client.proxy.delete_key(key))

    file: Final = unwrap(
        client.upload_file(
            content=render_jsonl(VERTEX_US_MODEL),
            form=FileUploadForm(purpose="batch", target_model_names=VERTEX_US_MODEL),
            key=key,
        )
    )
    assert isinstance(file, FileObject) and file.purpose == "batch"
    created: Final = client.create_batch(body=BatchCreateBody(input_file_id=file.id), key=key)
    require_successful_call(created)
    batch: Final = BatchObject.model_validate_json(created.body)

    finished: Final = _await_completed(client, batch.id, key)
    assert finished.status == "completed", f"batch ended {finished.status!r}"

    priced = client.proxy.poll_logs_for_key(
        key,
        predicate=lambda rows: any(r.call_type == "aretrieve_batch" and (r.spend or 0) > 0 for r in rows),
    )
    cost_rows: Final = [r for r in priced if r.call_type == "aretrieve_batch" and (r.spend or 0) > 0]
    assert cost_rows, (
        f"CheckBatchCost booked no priced aretrieve_batch row for key; rows={[(r.call_type, r.spend) for r in priced]}"
    )
    booked_spend: Final = sum(r.spend or 0 for r in cost_rows)
    booked_tokens: Final = sum(r.total_tokens or 0 for r in cost_rows)
    assert booked_tokens > 0, f"batch cost row carries no tokens: {cost_rows!r}"

    deadline: Final = time.monotonic() + client.proxy.poll_timeout
    activity = _team_activity(client, team_id)
    while time.monotonic() < deadline and activity.metadata.total_spend < booked_spend * 0.999:
        time.sleep(client.proxy.poll_interval)
        activity = _team_activity(client, team_id)
    assert isclose(activity.metadata.total_spend, booked_spend, rel_tol=1e-3), (
        f"team daily activity never rolled up the batch cost: total_spend={activity.metadata.total_spend} booked={booked_spend}"
    )

    batches: Final = _endpoint_totals(activity, BATCHES_ENDPOINT)
    endpoints: Final = {
        name: (entry.metrics.spend, entry.metrics.total_tokens, entry.metrics.api_requests)
        for row in activity.results
        for name, entry in row.breakdown.endpoints.items()
    }
    assert isclose(batches.spend, booked_spend, rel_tol=1e-3) and batches.total_tokens == booked_tokens, (
        f"Endpoint Activity /batches shows spend={batches.spend} total_tokens={batches.total_tokens} "
        f"api_requests={batches.api_requests}, but the batch cost row booked spend={booked_spend} "
        f"total_tokens={booked_tokens}; endpoints breakdown={endpoints}"
    )
