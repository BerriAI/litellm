from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.models.task_insights import (
    UNKNOWN_TASK_ID,
    Freshness,
    MatrixCell,
    TaskDefinition,
    TaskInsightsState,
    TaskVolume,
    WindowTotals,
)
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.deps.repositories import get_task_insights_repository
from litellm.proxy.management_endpoints.task_insights_endpoints import (
    get_capability_probe,
    get_clock,
    get_model_catalog,
    router,
)
from litellm.task_insights.summaries import ModelMetadata
from litellm.types.proxy.task_insights import ModelConfigSummary

_NOW = datetime(2026, 11, 11, 15, 0, tzinfo=timezone.utc)


def _task(task_id: str, status: str = "frozen", source: str = "sample") -> TaskDefinition:
    return TaskDefinition(
        task_id=task_id,
        name=task_id.title(),
        description="d",
        inclusion_criteria="i",
        exclusion_criteria="e",
        required_capabilities=("function_calling",),
        status=status,
        source=source,
        map_x=30,
        map_y=60,
        created_at=_NOW,
    )


class _Store:
    def __init__(self, definitions=(_task("debug"),), *, discovery_accepted=True, frozen=1) -> None:
        self.definitions = definitions
        self.discovery_accepted = discovery_accepted
        self.frozen = frozen
        self.windows = []
        self.samples = []

    async def get_state(self):
        return TaskInsightsState()

    async def list_definitions(self):
        return self.definitions

    async def task_volumes(self, window):
        self.windows.append(window)
        return (
            TaskVolume(
                task_id="debug", request_count=8, delivery_count=9, session_count=3, spend=0.8, cost_observations=8
            ),
            TaskVolume(
                task_id=UNKNOWN_TASK_ID,
                request_count=2,
                delivery_count=2,
                session_count=1,
                spend=None,
                cost_observations=0,
                unattributed_count=2,
            ),
        )

    async def matrix_cells(self, window):
        return (
            MatrixCell(
                model_config_id="opus",
                task_id="debug",
                request_count=8,
                avg_cost_usd=0.1,
                cost_observations=8,
                p95_latency_ms=None,
                latency_observations=0,
            ),
        )

    async def map_points(self, window, limit):
        return ()

    async def request_samples(self, window, task_id, model_config_id, limit):
        self.samples.append((task_id, model_config_id))
        return ()

    async def window_totals(self, window):
        return WindowTotals(
            request_count=10, delivery_count=11, session_count=3, analysis_request_count=1, analysis_spend=0.01
        )

    async def freshness(self):
        return Freshness(last_observed_at=_NOW, last_bucketed_at=None)

    async def edit_draft(self, task_id, edit):
        draft = next((task for task in self.definitions if task.task_id == task_id and task.status == "draft"), None)
        return draft.model_copy(update={"name": edit.name}) if draft and edit.name else draft

    async def freeze_draft(self, frozen_at):
        return self.frozen

    async def request_discovery(self, window, max_requests, requested_at):
        self.windows.append(window)
        return self.discovery_accepted


_WITH_DRAFT = (_task("debug"), _task("draft", status="draft", source="discovered"))


def _client(store: _Store | None, role: LitellmUserRoles = LitellmUserRoles.PROXY_ADMIN) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(api_key="sk", user_role=role)
    app.dependency_overrides[get_task_insights_repository] = lambda: store
    app.dependency_overrides[get_clock] = lambda: lambda: _NOW
    app.dependency_overrides[get_model_catalog] = lambda: (
        ModelConfigSummary(model_config_id="opus", model_name="claude-opus", provider_model="opus", configured=True),
        ModelConfigSummary(model_config_id="idle", model_name="claude-haiku", provider_model="haiku", configured=True),
    )
    app.dependency_overrides[get_capability_probe] = lambda: (
        lambda model: ModelMetadata({"function_calling": model == "opus"}, 200_000, "chat")
    )
    return TestClient(app)


def test_matrix_keeps_missing_evidence_null_with_eval_placeholders_and_idle_columns():
    response = _client(_Store()).get(
        "/task-insights/matrix", params={"start_date": "2026-11-01", "end_date": "2026-11-11"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["measurement_window"] == {"start_date": "2026-11-01", "end_date": "2026-11-11", "team_id": None}
    assert body["entries"] == [
        {
            "model_config_id": "opus",
            "task_id": "debug",
            "request_count": 8,
            "avg_cost_usd": 0.1,
            "cost_observations": 8,
            "p95_latency_ms": None,
            "latency_observations": 0,
            "eval_success_rate": None,
            "eval_sample_count": 0,
        }
    ]
    assert [config["model_config_id"] for config in body["model_configs"]] == ["opus", "idle"]
    assert {(item["model_config_id"], item["status"]) for item in body["compatibility"]} == {
        ("opus", "compatible"),
        ("idle", "incompatible"),
    }
    assert body["unattributed_request_count"] == 2


def test_overview_counts_requests_and_flags_sample_data():
    store = _Store()
    body = _client(store).get("/task-insights/overview", params={"team_id": "team-a"}).json()

    assert (body["totals"]["request_count"], body["totals"]["delivery_count"], body["totals"]["session_count"]) == (
        10,
        11,
        3,
    )
    assert body["status"]["sample_data"] is True
    assert [task["task_id"] for task in body["tasks"]] == ["debug", UNKNOWN_TASK_ID]
    assert body["window"] == {"start_date": "2026-10-13", "end_date": "2026-11-11", "team_id": "team-a"}
    assert (store.windows[0].start.isoformat(), store.windows[0].end.isoformat()) == (
        "2026-10-13T00:00:00+00:00",
        "2026-11-12T00:00:00+00:00",
    )


def test_task_detail_filters_requests_by_deployment_and_404s_unknown_tasks():
    store = _Store()
    client = _client(store)

    assert client.get("/task-insights/tasks/debug", params={"model_config_id": "opus"}).status_code == 200
    assert client.get(f"/task-insights/tasks/{UNKNOWN_TASK_ID}").json()["definition"] is None
    assert client.get("/task-insights/tasks/missing").status_code == 404
    assert store.samples == [("debug", "opus"), (UNKNOWN_TASK_ID, None)]


@pytest.mark.parametrize(
    ("params", "status"),
    [
        ({"start_date": "11/01/2026"}, 400),
        ({"start_date": "2026-11-12", "end_date": "2026-11-11"}, 400),
        ({"start_date": "2024-01-01"}, 400),
    ],
)
def test_invalid_windows_are_rejected(params, status):
    assert _client(_Store()).get("/task-insights/overview", params=params).status_code == status


@pytest.mark.parametrize(
    ("role", "read", "write"),
    [
        (LitellmUserRoles.PROXY_ADMIN, 200, 200),
        (LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, 200, 403),
        (LitellmUserRoles.INTERNAL_USER, 403, 403),
    ],
)
def test_reads_need_an_admin_role_and_writes_need_a_full_admin(role, read, write):
    client = _client(_Store(_WITH_DRAFT), role)

    assert client.get("/task-insights/taxonomy").status_code == read
    assert client.post("/task-insights/taxonomy/freeze", json={"draft_task_count": 1}).status_code == write


def test_authorization_is_checked_before_the_database():
    assert _client(None, LitellmUserRoles.INTERNAL_USER).get("/task-insights/overview").status_code == 403
    assert _client(None).get("/task-insights/overview").status_code == 500


def test_freezing_a_draft_that_changed_since_review_and_double_queued_discovery_conflict():
    client = _client(_Store(_WITH_DRAFT))

    assert client.post("/task-insights/taxonomy/freeze", json={"draft_task_count": 2}).status_code == 409
    assert client.post("/task-insights/taxonomy/freeze", json={"draft_task_count": 0}).status_code == 422
    assert client.post("/task-insights/taxonomy/freeze", json={"draft_task_count": 1}).json() == {
        "frozen_task_count": 1
    }
    assert _client(_Store(discovery_accepted=False)).post("/task-insights/discovery", json={}).status_code == 409


def test_discovery_bounds_its_sample_size():
    client = _client(_Store())

    assert client.post("/task-insights/discovery", json={"max_requests": 501}).status_code == 422
    assert client.post("/task-insights/discovery", json={"max_requests": 200}).status_code == 200


def test_only_draft_definitions_can_be_edited():
    client = _client(_Store(_WITH_DRAFT))

    assert client.patch("/task-insights/tasks/debug", json={"name": "Renamed"}).status_code == 404
    assert client.patch("/task-insights/tasks/draft", json={"name": "Renamed"}).json()["name"] == "Renamed"
    assert client.patch("/task-insights/tasks/draft", json={"name": ""}).status_code == 422
