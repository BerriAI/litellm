import asyncio
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from math import isclose
from types import MappingProxyType
from typing import Final, cast

import httpx
import pytest
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
from litellm.proxy.management_endpoints.roi_calculator_endpoints import (
    _estimator_choices_from_deployments,
    _estimator_models_from_deployments,
    _gateway_transport,
    _next_update,
    get_github_transport,
    get_roi_config_repository,
    register_scheduled_sync,
    router,
    run_scheduled_sync,
)
from litellm.proxy.roi_calculator.estimator import estimator_options
from litellm.proxy.roi_calculator.sample import sample_report
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
from litellm.types.roi_calculator import ROIReport, ROISettings, ROISummaryResponse, ROISyncStatus

_JSON_HEADERS: Final = MappingProxyType({"content-type": "application/json"})


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ("/v1/chat/completions", "/v1/responses", "/v1/messages"))
@pytest.mark.parametrize("string_metadata", (False, True))
async def test_only_internal_estimator_transport_can_mark_persisted_spend(path: str, string_metadata: bool) -> None:
    from litellm.proxy.proxy_server import ProxyConfig

    app: Final = FastAPI()
    tags: Final = ("repo:org/repo", "branch:feature", "litellm-roi-estimator")
    forged: Final = {"tags": tags, "litellm_roi_estimator": True}
    metadata: Final = json.dumps(forged) if string_metadata else forged
    body: Final = {"model": "test-model", "metadata": metadata, "litellm_metadata": metadata}
    now: Final = datetime(2026, 9, 15, tzinfo=timezone.utc)

    @app.post(path)
    async def log_request(request: Request) -> Mapping[str, object]:
        data: Final = await add_litellm_data_to_request(
            data=await request.json(),
            request=request,
            user_api_key_dict=UserAPIKeyAuth(api_key="test-key", metadata={"litellm_roi_estimator": True}),
            proxy_config=ProxyConfig(),
        )
        payload: Final = get_logging_payload(
            kwargs={"model": "test-model", "response_cost": 0.25, "litellm_params": data},
            response_obj={"id": "test-request", "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
            start_time=now,
            end_time=now,
        )
        return {
            "metadata": json.loads(payload["metadata"]),
            "tags": json.loads(payload["request_tags"]),
            "spend": payload["spend"],
        }

    async with (
        httpx.AsyncClient(transport=_gateway_transport(app), base_url="http://test") as internal,
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as external,
    ):
        for client, expected in ((external, False), (internal, True), (external, False)):
            response: Final = await client.post(path, json=body, headers={"x-litellm-roi-estimator": "true"})
            assert response.status_code == 200
            logged: Final = response.json()
            assert logged["metadata"].get("litellm_roi_estimator") is expected
            assert set(logged["tags"]) == set(tags)
            assert logged["spend"] == 0.25


@pytest.mark.asyncio
async def test_repeated_startup_keeps_one_roi_schedule() -> None:
    scheduler: Final = AsyncIOScheduler()
    scheduler.start(paused=True)
    try:
        register_scheduled_sync(scheduler)
        register_scheduled_sync(scheduler)

        jobs: Final = scheduler.get_jobs()
        assert len(jobs) == 1
        assert jobs[0].func is run_scheduled_sync
    finally:
        scheduler.shutdown(wait=False)


def _assert_json_round_trip(value: object) -> None:
    serialized: Final = json.dumps(value)
    decoded: Final[object] = cast(object, json.loads(serialized))
    assert decoded == value


class _Parameter:
    def __init__(self, param_value: object) -> None:
        self.param_value: Final = param_value


class _ConfigRepository:
    def __init__(self) -> None:
        self.values: Mapping[str, object] = MappingProxyType({})

    async def get_param(self, param_name: str) -> _Parameter | None:
        value: Final = self.values.get(param_name)
        return _Parameter(value) if param_name in self.values else None

    async def set_param(self, param_name: str, param_value: object) -> object:
        _assert_json_round_trip(param_value)
        self.values = MappingProxyType({**self.values, param_name: param_value})
        return self.values[param_name]


def _client(
    role: LitellmUserRoles, repository: _ConfigRepository, transport: httpx.AsyncBaseTransport | None = None
) -> TestClient:
    app: Final = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=role)
    app.dependency_overrides[get_roi_config_repository] = lambda: repository
    app.dependency_overrides[get_github_transport] = lambda: transport
    return TestClient(app)


def test_router_group_uses_underlying_model_metadata_for_reasoning_option() -> None:
    import litellm

    supported_model: Final = next(
        model
        for model, metadata in litellm.model_cost.items()
        if metadata.get("supports_none_reasoning_effort") is True
    )
    deployments: Final = (
        {
            "model_name": "roi-estimator",
            "litellm_params": {"model": "custom-deployment"},
            "model_info": {"base_model": supported_model},
        },
    )

    estimator_models: Final = _estimator_models_from_deployments(deployments)

    assert estimator_models == ((supported_model, None),)
    assert estimator_options(estimator_models) == {"reasoning_effort": "none"}


def test_non_admin_cannot_read_roi_settings() -> None:
    client: Final = _client(LitellmUserRoles.INTERNAL_USER, _ConfigRepository())

    response: Final = client.get("/roi-calculator/settings")

    assert response.status_code == 403


def test_view_only_admin_cannot_change_roi_settings() -> None:
    client: Final = _client(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, _ConfigRepository())

    response: Final = client.put(
        "/roi-calculator/settings",
        content='{"repos":["org/repo"]}',
        headers=_JSON_HEADERS,
    )

    assert response.status_code == 403


def test_github_token_is_never_returned_and_url_change_clears_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "roi-calculator-test-salt-key-0123456789")
    repository: Final = _ConfigRepository()
    client: Final = _client(LitellmUserRoles.PROXY_ADMIN, repository)

    saved: Final = client.put(
        "/roi-calculator/settings",
        content=('{"github_token":"private-test-token","repos":["org/repo"],"estimator_model":"test-estimator"}'),
        headers=_JSON_HEADERS,
    )

    assert saved.status_code == 200
    assert saved.json()["has_github_token"] is True
    assert "private-test-token" not in saved.text
    stored_settings: Final = TypeAdapter(ROISettings).validate_python(repository.values["roi_calculator_settings"])
    encrypted_token: Final = stored_settings.github_token.get_secret_value()
    assert encrypted_token != "private-test-token"
    assert "private-test-token" not in encrypted_token

    updated: Final = client.put(
        "/roi-calculator/settings",
        content='{"github_api_url":"https://github.enterprise.test/api/v3"}',
        headers=_JSON_HEADERS,
    )

    assert updated.status_code == 200
    assert updated.json()["has_github_token"] is False


def test_github_api_url_must_use_https() -> None:
    repository: Final = _ConfigRepository()
    client: Final = _client(LitellmUserRoles.PROXY_ADMIN, repository)

    response: Final = client.put(
        "/roi-calculator/settings",
        content='{"github_api_url":"http://github.enterprise.test/api/v3"}',
        headers=_JSON_HEADERS,
    )

    assert response.status_code == 422
    assert not repository.values


@pytest.mark.parametrize(
    "patch", ({"github_api_url": None}, {"gitlab_api_url": None}, {"repos": ["invalid"]}, {"estimator_prompt": " "})
)
def test_invalid_connection_settings_are_rejected_without_saving(patch: Mapping[str, object]) -> None:
    repository: Final = _ConfigRepository()
    client: Final = _client(LitellmUserRoles.PROXY_ADMIN, repository)
    assert client.put("/roi-calculator/settings", json=patch).status_code == 422
    assert not repository.values


@pytest.mark.parametrize("upstream_status", (200, 403))
def test_public_gitlab_repository_browser_and_errors(upstream_status: int) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v4/projects"
        assert request.url.params["search"] == "gateway"
        assert "PRIVATE-TOKEN" not in request.headers
        return httpx.Response(
            upstream_status, json=[{"id": 1, "path_with_namespace": "group/gateway"}], headers={"x-next-page": "2"}
        )

    repository: Final = _ConfigRepository()
    client: Final = _client(LitellmUserRoles.PROXY_ADMIN, repository, httpx.MockTransport(respond))
    assert client.put("/roi-calculator/settings", json={"source_provider": "gitlab"}).status_code == 200
    response: Final = client.get("/roi-calculator/repositories", params={"query": "gateway"})
    if upstream_status == 200:
        assert response.status_code == 200
        assert response.json() == {
            "repositories": [{"name": "group/gateway", "visibility": "private", "archived": False}],
            "page": 1,
            "has_more": True,
        }
    else:
        assert response.status_code == 502
        assert "HTTP 403" in response.json()["detail"]


@pytest.mark.parametrize("role", [LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY])
@pytest.mark.parametrize(
    "method,path,body",
    [
        ("POST", "/roi-calculator/sync", {}),
        ("DELETE", "/roi-calculator/sync", {}),
        ("POST", "/roi-calculator/setup/reset", {}),
        ("POST", "/roi-calculator/connections/test", {}),
        ("PUT", "/roi-calculator/identity-map", {"github_login": "alice", "email": "alice@example.com"}),
    ],
)
def test_all_writes_require_full_admin(role: LitellmUserRoles, method: str, path: str, body: Mapping[str, str]) -> None:
    client: Final = _client(role, _ConfigRepository())
    assert client.request(method, path, json=body).status_code == 403


@pytest.mark.parametrize("login", ("invalid.name", " ", "user/name"))
@pytest.mark.parametrize("email", ("alice@example.com", None))
def test_invalid_identity_login_returns_validation_error(login: str, email: str | None) -> None:
    repository: Final = _ConfigRepository()
    client: Final = _client(LitellmUserRoles.PROXY_ADMIN, repository)
    response: Final = client.put("/roi-calculator/identity-map", json={"github_login": login, "email": email})
    assert response.status_code == 422
    assert not repository.values


def test_schedule_and_estimator_key_persist_without_exposing_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "roi-calculator-test-salt-key-0123456789")
    repository: Final = _ConfigRepository()
    client: Final = _client(LitellmUserRoles.PROXY_ADMIN, repository)
    saved: Final = client.put(
        "/roi-calculator/settings", json={"estimator_key": "sk-test-secret", "update_interval_minutes": 60}
    )
    assert saved.status_code == 200
    assert saved.json()["has_estimator_key"] is True
    assert saved.json()["update_interval_minutes"] == 60
    assert "sk-test-secret" not in saved.text
    assert "sk-test-secret" not in str(repository.values)
    updated: Final = client.put("/roi-calculator/settings", json={"estimator_key": None, "update_interval_minutes": 0})
    assert updated.json()["has_estimator_key"] is False
    assert updated.json()["update_interval_minutes"] == 0


def test_sample_preview_does_not_change_live_settings_or_report() -> None:
    repository: Final = _ConfigRepository()
    client: Final = _client(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, repository)
    response: Final = client.get("/roi-calculator/report", params={"mode": "demo"})
    assert response.status_code == 200
    report: Final = ROISummaryResponse.model_validate(response.json()["report"])
    assert report.mode == "demo"
    assert report.metrics.cost_per_hour is not None and report.metrics.cost_per_hour > 0
    assert all(pull.branch_cost.status == "matched" and (pull.branch_cost.spend or 0) > 0 for pull in report.pulls)
    assert any(not pull.matched for pull in report.pulls)
    assert isclose(report.branch_metrics.spend, sum(pull.branch_cost.spend or 0 for pull in report.pulls))
    assert report.branch_metrics.unlinked_spend > 0
    assert isclose(
        report.branch_metrics.total_tagged_spend, report.branch_metrics.spend + report.branch_metrics.unlinked_spend
    )
    assert not repository.values
    assert client.get("/roi-calculator/report").json()["report"] is None


@pytest.mark.parametrize("interval", [0.1, 1, 4.99])
def test_schedule_rejects_intervals_under_five_minutes(interval: float) -> None:
    client: Final = _client(LitellmUserRoles.PROXY_ADMIN, _ConfigRepository())
    assert client.put("/roi-calculator/settings", json={"update_interval_minutes": interval}).status_code == 422


@pytest.mark.parametrize("anchor", ("2026-09-30T12:00:00", "2026-09-30T12:00:00Z", "2026-09-30T14:00:00+02:00"))
def test_schedule_normalizes_legacy_and_offset_timestamps(anchor: str) -> None:
    settings: Final = ROISettings(repos=("example/repo",), estimator_model="estimator", update_interval_minutes=60)
    status: Final = ROISyncStatus(
        running=False,
        phase="error",
        stage="Interrupted",
        done=0,
        total=0,
        estimated=0,
        reused=0,
        needs_attention=0,
        error=None,
        finished_at=anchor,
    )
    report: Final = sample_report(datetime(2026, 9, 30, tzinfo=timezone.utc))
    assert _next_update(settings, status, report) == datetime(2026, 9, 30, 13, tzinfo=timezone.utc)


def test_manual_match_recalculates_saved_report_and_removal_restores_cohort() -> None:
    repository: Final = _ConfigRepository()
    report: Final[ROIReport] = {**sample_report(datetime(2026, 9, 30, tzinfo=timezone.utc)), "mode": "live"}
    serialized: Final = TypeAdapter(dict[str, object]).validate_json(TypeAdapter(ROIReport).dump_json(report))
    asyncio.run(repository.set_param("roi_calculator_report", serialized))
    client: Final = _client(LitellmUserRoles.PROXY_ADMIN, repository)
    before: Final = client.get("/roi-calculator/report")
    assert before.status_code == 200
    assert before.json()["report"]["metrics"]["output_hours"] == 10.5
    matched: Final = client.put(
        "/roi-calculator/identity-map",
        content='{"github_login":" CASEY ","email":"Alex@Example.com"}',
        headers=_JSON_HEADERS,
    )
    assert matched.status_code == 200
    assert matched.json()["identity_map"]["casey"] == "alex@example.com"
    assert matched.json()["report"]["metrics"]["output_hours"] == 16
    assert matched.json()["report"]["metrics"]["cost_per_hour"] == pytest.approx(31 / 16)
    removed: Final = client.put(
        "/roi-calculator/identity-map", content='{"github_login":"casey","email":null}', headers=_JSON_HEADERS
    )
    assert removed.status_code == 200
    assert not removed.json()["identity_map"]
    assert removed.json()["report"]["metrics"] == before.json()["report"]["metrics"]


def test_switching_sources_clears_report_and_identities_and_keeps_tokens_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "roi-calculator-test-salt-key-0123456789")
    repository: Final = _ConfigRepository()
    client: Final = _client(LitellmUserRoles.PROXY_ADMIN, repository)
    saved: Final = client.put(
        "/roi-calculator/settings",
        json={"source_provider": "gitlab", "gitlab_token": "private-gitlab-test", "repos": ["group/subgroup/project"]},
    )
    assert saved.status_code == 200
    assert saved.json()["has_gitlab_token"] is True
    assert "private-gitlab-test" not in saved.text
    assert "private-gitlab-test" not in str(repository.values)
    assert client.get("/roi-calculator/report").json()["report"] is None
    matched: Final = client.put(
        "/roi-calculator/identity-map", json={"github_login": "dev.name", "email": "dev@example.test"}
    )
    assert matched.status_code == 200
    assert matched.json()["identity_map"] == {"dev.name": "dev@example.test"}
    switched: Final = client.put("/roi-calculator/settings", json={"source_provider": "github"})
    assert switched.status_code == 200
    assert switched.json()["identity_map"] == {}
    assert switched.json()["repos"] == []
    assert client.get("/roi-calculator/report").json()["report"] is None
    changed_host: Final = client.put(
        "/roi-calculator/settings",
        json={"source_provider": "gitlab", "gitlab_api_url": "https://git.example.test/api/v4"},
    )
    assert changed_host.json()["has_gitlab_token"] is False


def test_old_source_report_is_not_returned_when_matching_new_source_identity() -> None:
    repository: Final = _ConfigRepository()
    client: Final = _client(LitellmUserRoles.PROXY_ADMIN, repository)
    assert client.put("/roi-calculator/settings", json={"source_provider": "gitlab"}).status_code == 200
    old_report: Final = sample_report(datetime.now(timezone.utc))
    serialized: Final = TypeAdapter(dict[str, object]).validate_json(TypeAdapter(ROIReport).dump_json(old_report))
    asyncio.run(repository.set_param("roi_calculator_report", serialized))
    assert client.get("/roi-calculator/report").json()["report"] is None
    matched: Final = client.put(
        "/roi-calculator/identity-map", json={"github_login": "dev.name", "email": "dev@example.test"}
    )
    assert matched.status_code == 200
    assert matched.json()["report"] is None
    assert matched.json()["identity_map"] == {"dev.name": "dev@example.test"}


def test_estimator_choices_show_underlying_models_and_exclude_non_chat_routes() -> None:
    deployments: Final = (
        {
            "model_name": "estimator",
            "litellm_params": {"model": "deployment-name"},
            "model_info": {"base_model": "gpt-6-luna", "mode": "chat"},
        },
        {
            "model_name": "estimator",
            "litellm_params": {"model": "second-deployment"},
            "model_info": {"base_model": "gpt-6-luna", "mode": "chat"},
        },
        {
            "model_name": "embeddings",
            "litellm_params": {"model": "custom-embedding"},
            "model_info": {"mode": "embedding"},
        },
        {
            "model_name": "image",
            "litellm_params": {"model": "custom-image"},
            "model_info": {"mode": "image_generation"},
        },
        {"model_name": "*", "litellm_params": {"model": "openai/*"}},
        {"model_name": "missing", "litellm_params": {}},
        {"model_name": "custom-chat", "litellm_params": {"model": "openai/private-model"}},
    )
    choices: Final = _estimator_choices_from_deployments(deployments)
    assert tuple((choice.model_name, choice.provider_models) for choice in choices) == (
        ("custom-chat", ("openai/private-model",)),
        ("estimator", ("gpt-6-luna",)),
    )


def test_estimator_picker_keeps_callable_aliases_and_routing_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server
    from litellm.router import Router

    configured_router: Final = Router(
        model_list=[
            {
                "model_name": "concrete",
                "litellm_params": {"model": "openai/gpt-6-luna", "api_key": "test"},
            },
            {
                "model_name": "team-only",
                "litellm_params": {"model": "openai/gpt-6-luna", "api_key": "test"},
                "model_info": {"team_id": "other-team", "team_public_model_name": "private-estimator"},
            },
        ],
        model_group_alias={"friendly": "concrete"},
        routing_groups=[{"group_name": "balanced", "models": ["concrete"], "routing_strategy": "simple-shuffle"}],
    )
    monkeypatch.setattr(proxy_server, "llm_router", configured_router)
    client: Final = _client(LitellmUserRoles.PROXY_ADMIN, _ConfigRepository())
    for name in ("friendly", "balanced"):
        response: Final = client.put("/roi-calculator/settings", json={"repos": ["org/repo"], "estimator_model": name})
        assert response.status_code == 200, response.text
        settings: Final = response.json()
        assert settings["ready"] is True
        assert set(settings["available_models"]) == {"concrete", "friendly", "balanced"}
        assert {"model_name": name, "provider_models": ["openai/gpt-6-luna"]} in settings["estimator_models"]
