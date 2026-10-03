import asyncio
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.db.model_usage_rollup import build_model_usage_transaction, flush_model_usage_transactions
from litellm.proxy.db.model_usage_task_classifier import ModelUsageTaskClassifier
from litellm.proxy.management_endpoints.model_insights_endpoints import (
    _model_insights_task_classifier_client_builder,
    _model_insights_task_classifier_environment_lookup,
    _model_insights_task_classifier_runtime,
    router,
)
from litellm.llms.oss_decision import OSS_DECISION_MODELS


def _override_auth() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="sk-test", user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN)


def _task_classifier_app(
    role: LitellmUserRoles,
    environment: Mapping[str, str],
    configured: Mapping[str, str] | None = None,
) -> tuple[FastAPI, MagicMock]:
    def lookup_environment_value(key: str) -> str | None:
        return environment.get(key)

    prisma: Final = MagicMock()
    prisma.db.litellm_config.find_unique = AsyncMock(
        return_value=(
            SimpleNamespace(param_name="model_insights_task_classifier", param_value=dict(configured))
            if configured is not None
            else None
        )
    )
    prisma.db.litellm_config.upsert = AsyncMock()
    prisma.db.litellm_config.delete = AsyncMock()
    prisma._model_usage_transactions_lock = asyncio.Lock()
    app: Final = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        api_key="sk-test", user_id="test-user", user_role=role
    )
    app.dependency_overrides[_model_insights_task_classifier_environment_lookup] = (
        lambda: lookup_environment_value
    )
    app.dependency_overrides[_model_insights_task_classifier_client_builder] = lambda: lambda _: MagicMock()
    app.dependency_overrides[_model_insights_task_classifier_runtime] = lambda: ModelUsageTaskClassifier()
    return app, prisma


def _grouped_row(*, prompt_tokens: str = "100", completion_tokens: str = "200", **dimensions: str) -> dict[str, object]:
    return {
        **dimensions,
        "_sum": {
            "spend": 1.25,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "request_count": "3",
            "successful_requests": "3",
            "failed_requests": "0",
        },
    }


def test_model_insights_reads_only_bounded_rollup() -> None:
    model = _grouped_row(model_group="fast-chat", model="openai/gpt-5.4-mini", custom_llm_provider="openai")
    prompt_heavy_model = _grouped_row(
        prompt_tokens="500",
        completion_tokens="10",
        model_group="long-context",
        model="anthropic/claude-sonnet-4-5",
        custom_llm_provider="anthropic",
    )
    daily = _grouped_row(
        date="2026-09-28",
        model_group="fast-chat",
        model="openai/gpt-5.4-mini",
        custom_llm_provider="openai",
    )
    table = MagicMock()
    table.group_by = AsyncMock(side_effect=[[model, prompt_heavy_model], [daily], []])
    prisma = MagicMock()
    prisma.db.litellm_dailymodelusage = table
    prisma.db.query_raw = AsyncMock()
    prisma.db.litellm_spendlogs.find_many = AsyncMock()
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = _override_auth

    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        response = TestClient(app).get("/model-insights?start_date=2026-09-09&end_date=2026-09-28")

    assert response.status_code == 200
    assert response.json()["top_models"][0]["model_group"] == "long-context"
    assert "by_task" not in response.json()
    assert table.group_by.await_count == 3
    prisma.db.query_raw.assert_not_awaited()
    prisma.db.litellm_spendlogs.find_many.assert_not_awaited()


def test_model_insights_rejects_ranges_over_365_days() -> None:
    prisma = MagicMock()
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = _override_auth

    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        response = TestClient(app).get("/model-insights?start_date=2025-09-01&end_date=2026-09-28")

    assert response.status_code == 400


def _call(table: MagicMock, query: str, path: str = "/model-insights") -> object:
    prisma = MagicMock()
    prisma.db.litellm_dailymodelusage = table
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = _override_auth
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        return TestClient(app).get(f"{path}?start_date=2026-09-01&end_date=2026-09-28&{query}")


def test_model_insights_ranks_top_models_by_selected_metric() -> None:
    token_heavy = _grouped_row(
        prompt_tokens="9000", completion_tokens="9000", model_group="big", model="m1", custom_llm_provider="openai"
    )
    request_heavy = _grouped_row(
        prompt_tokens="1", completion_tokens="1", model_group="busy", model="m2", custom_llm_provider="openai"
    )
    request_heavy["_sum"]["request_count"] = "500"
    table = MagicMock()
    table.group_by = AsyncMock(side_effect=[[token_heavy, request_heavy], [], []])

    by_requests = _call(table, "metric=requests").json()
    by_tokens = _call(
        MagicMock(group_by=AsyncMock(side_effect=[[token_heavy, request_heavy], [], []])), "metric=tokens"
    ).json()

    assert by_requests["top_models"][0]["model_group"] == "busy"
    assert by_tokens["top_models"][0]["model_group"] == "big"


def test_model_insights_scopes_daily_to_ranked_deployments() -> None:
    ranked = _grouped_row(model_group="shared", model="m1", custom_llm_provider="openai")
    table = MagicMock()
    table.group_by = AsyncMock(side_effect=[[ranked], [], []])

    _call(table, "metric=tokens")

    daily_where = table.group_by.await_args_list[1].kwargs["where"]
    assert daily_where["OR"] == [{"model_group": "shared", "model": "m1", "custom_llm_provider": "openai"}]
    assert "model_group" not in daily_where


def test_model_insights_daily_totals_cover_every_model_not_just_the_ranked_ones() -> None:
    ranked = _grouped_row(model_group="ranked", model="m1", custom_llm_provider="openai")
    ranked_day = _grouped_row(date="2026-09-28", model_group="ranked", model="m1", custom_llm_provider="openai")
    whole_gateway_day = _grouped_row(prompt_tokens="7000", completion_tokens="3000", date="2026-09-28")
    table = MagicMock()
    table.group_by = AsyncMock(side_effect=[[ranked], [ranked_day], [whole_gateway_day]])

    body = _call(table, "metric=tokens").json()

    totals_call = table.group_by.await_args_list[2].kwargs
    assert totals_call["by"] == ["date"]
    assert "OR" not in totals_call["where"]
    assert body["daily_totals"] == [
        {"date": "2026-09-28", "spend": 1.25, "prompt_tokens": 7000, "completion_tokens": 3000, "requests": 3}
    ]
    assert body["daily"][0]["prompt_tokens"] + body["daily"][0]["completion_tokens"] < 10000


def _task_rows() -> list[dict[str, object]]:
    def row(task: str, group: str, requests: str, spend: float) -> dict[str, object]:
        base = _grouped_row(task_type=task, model_group=group, model=group, custom_llm_provider="openai")
        base["_sum"].update({"request_count": requests, "spend": spend})
        return base

    return [
        row("debugging", "big", "1", 9.0),
        row("debugging", "busy", "50", 1.0),
        row("classification", "busy", "10", 1.0),
    ]


def test_model_insight_tasks_are_summarised_on_the_server() -> None:
    table = MagicMock(group_by=AsyncMock(return_value=_task_rows()))

    body = _call(table, "metric=spend", path="/model-insights/tasks").json()

    assert [(t["task_type"], t["label"], t["category"], t["leader"]) for t in body["tasks"]] == [
        ("debugging", "Debugging", "Code", "big"),
        ("classification", "Classification", "General", "busy"),
    ]
    assert [round(t["share"], 1) for t in body["tasks"]] == [90.9, 9.1]
    assert "OR" not in table.group_by.await_args.kwargs["where"]
    assert "take" not in table.group_by.await_args.kwargs


def test_model_insight_tasks_leader_follows_the_selected_metric() -> None:
    by_spend = _call(MagicMock(group_by=AsyncMock(return_value=_task_rows())), "metric=spend", "/model-insights/tasks")
    by_requests = _call(
        MagicMock(group_by=AsyncMock(return_value=_task_rows())), "metric=requests", "/model-insights/tasks"
    )

    assert by_spend.json()["tasks"][0]["leader"] == "big"
    assert by_requests.json()["tasks"][0]["leader"] == "busy"


def test_model_insight_tasks_unknown_task_shows_as_uncategorized() -> None:
    row = _grouped_row(task_type="uncategorized", model_group="a", model="a", custom_llm_provider="openai")
    body = _call(MagicMock(group_by=AsyncMock(return_value=[row])), "metric=spend", "/model-insights/tasks").json()

    assert [(t["label"], t["category"]) for t in body["tasks"]] == [("Uncategorized", "General")]


def test_model_insight_tasks_require_an_admin() -> None:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        api_key="sk-test", user_id="u", user_role=LitellmUserRoles.INTERNAL_USER
    )
    with patch("litellm.proxy.proxy_server.prisma_client", MagicMock()):
        assert TestClient(app).get("/model-insights/tasks").status_code == 403


def test_model_insights_rejects_unknown_metric() -> None:
    assert _call(MagicMock(group_by=AsyncMock()), "metric=bogus").status_code == 422


def test_task_classifier_get_reports_config_and_readiness_from_injected_environment() -> None:
    app, prisma = _task_classifier_app(
        LitellmUserRoles.PROXY_ADMIN,
        {"LAYA_API_BASE": "https://laya.example"},
        {"provider": "laya", "model": "english"},
    )

    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        response: Final = TestClient(app).get("/model-insights/task-classifier")

    assert response.status_code == 200
    assert response.json() == {
        "configured": {"provider": "laya", "model": "english"},
        "providers": [
            {
                "provider": "jev",
                "label": "Jev (TypeSafe)",
                "models": ["jev-latest"],
                "ready": False,
                "missing_env": ["TYPESAFE_API_KEY"],
            },
            {
                "provider": "laya",
                "label": "Laya",
                "models": list(OSS_DECISION_MODELS["laya"]),
                "ready": True,
                "missing_env": [],
            },
            {
                "provider": "bespoke",
                "label": "Bespoke Nimble",
                "models": list(OSS_DECISION_MODELS["bespoke"]),
                "ready": False,
                "missing_env": ["BESPOKE_API_BASE"],
            },
        ],
    }


def test_task_classifier_put_rejects_an_unready_provider() -> None:
    app: Final = _task_classifier_app(LitellmUserRoles.PROXY_ADMIN, {})[0]

    with patch("litellm.proxy.proxy_server.prisma_client", MagicMock()):
        response: Final = TestClient(app).put(
            "/model-insights/task-classifier",
            json={"provider": "jev", "model": "jev-latest"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Provider 'jev' is not ready; set TYPESAFE_API_KEY"


def test_task_classifier_put_rejects_a_model_not_in_the_provider_catalog() -> None:
    app: Final = _task_classifier_app(
        LitellmUserRoles.PROXY_ADMIN,
        {"LAYA_API_BASE": "https://laya.example"},
    )[0]

    with patch("litellm.proxy.proxy_server.prisma_client", MagicMock()):
        response: Final = TestClient(app).put(
            "/model-insights/task-classifier",
            json={"provider": "laya", "model": "not-a-laya-model"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid model 'not-a-laya-model' for provider 'laya'"


def test_task_classifier_put_persists_and_returns_the_selected_config() -> None:
    app, prisma = _task_classifier_app(
        LitellmUserRoles.PROXY_ADMIN,
        {"LAYA_API_BASE": "https://laya.example"},
    )

    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        response: Final = TestClient(app).put(
            "/model-insights/task-classifier",
            json={"provider": "laya", "model": "english"},
        )

    assert response.status_code == 200
    assert response.json()["configured"] == {"provider": "laya", "model": "english"}
    assert prisma.db.litellm_config.upsert.await_args.kwargs["where"] == {
        "param_name": "model_insights_task_classifier"
    }
    assert prisma.db.litellm_config.upsert.await_args.kwargs["data"]["create"]["param_value"] == (
        '{"provider": "laya", "model": "english"}'
    )


@pytest.mark.parametrize("role", [LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY])
def test_non_admin_cannot_put_task_classifier(role: LitellmUserRoles) -> None:
    app: Final = _task_classifier_app(role, {"LAYA_API_BASE": "https://laya.example"})[0]

    with patch("litellm.proxy.proxy_server.prisma_client", MagicMock()):
        response: Final = TestClient(app).put(
            "/model-insights/task-classifier",
            json={"provider": "laya", "model": "english"},
        )

    assert response.status_code == 403


def test_view_only_admin_can_get_task_classifier() -> None:
    app, prisma = _task_classifier_app(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, {})

    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        response: Final = TestClient(app).get("/model-insights/task-classifier")

    assert response.status_code == 200
    assert response.json()["configured"] is None


def test_admin_can_delete_task_classifier_and_clear_the_local_runtime() -> None:
    app, prisma = _task_classifier_app(
        LitellmUserRoles.PROXY_ADMIN,
        {},
        {"provider": "laya", "model": "english"},
    )

    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        response: Final = TestClient(app).delete("/model-insights/task-classifier")

    assert response.status_code == 200
    assert response.json()["configured"] is None
    prisma.db.litellm_config.delete.assert_awaited_once_with(
        where={"param_name": "model_insights_task_classifier"}
    )


class _InMemoryUsageTable:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, ...], dict[str, float]] = {}

    def upsert(self, where: dict, data: dict) -> None:
        key_fields = where["date_model_group_model_custom_llm_provider_task_type"]
        key = tuple(key_fields.values())
        if key not in self.rows:
            self.rows[key] = {**key_fields, **{k: v for k, v in data["create"].items() if k not in key_fields}}
            return
        for field, change in data["update"].items():
            self.rows[key][field] += change["increment"]

    async def group_by(self, by: list[str], sum: dict, where: dict, **_: object) -> list[dict]:
        grouped: dict[tuple, dict] = {}
        for row in self.rows.values():
            if not where["date"]["gte"] <= row["date"] <= where["date"]["lte"]:
                continue
            if where.get("OR") and not any(all(row[k] == v for k, v in option.items()) for option in where["OR"]):
                continue
            bucket = grouped.setdefault(tuple(row[k] for k in by), {**{k: row[k] for k in by}, "_sum": {}})
            for field in sum:
                bucket["_sum"][field] = bucket["_sum"].get(field, 0) + row[field]
        return list(grouped.values())


class _InMemoryBatcher:
    def __init__(self, table: _InMemoryUsageTable) -> None:
        self.litellm_dailymodelusage = table

    async def __aenter__(self) -> "_InMemoryBatcher":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


@pytest.mark.asyncio
async def test_model_insights_reads_back_what_the_rollup_wrote() -> None:
    table = _InMemoryUsageTable()
    prisma = MagicMock()
    prisma.db.litellm_dailymodelusage = table
    prisma.db.batch_ = MagicMock(return_value=_InMemoryBatcher(table))
    payload = {
        "call_type": "acompletion",
        "spend": 0.5,
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "startTime": datetime(2026, 9, 28, tzinfo=timezone.utc),
        "model": "gpt-5",
        "model_group": "gpt-5",
        "metadata": "{}",
        "request_tags": "[]",
        "custom_llm_provider": "openai",
        "status": "success",
    }

    transactions = tuple(
        build_model_usage_transaction({**payload, "request_tags": "[]"})
        for _ in range(2)
    )
    uncategorized: Final = tuple(transaction for transaction in transactions if transaction is not None)
    classified: Final = (
        replace(
            uncategorized[0],
            key=replace(uncategorized[0].key, task_type="debugging"),
        ),
        uncategorized[1],
    )
    await flush_model_usage_transactions(prisma, classified)

    body = _call(table, "metric=requests").json()

    assert [(m["model_group"], m["requests"], m["prompt_tokens"]) for m in body["top_models"]] == [("gpt-5", 2, 20)]
    tasks = _call(table, "metric=requests", path="/model-insights/tasks").json()["tasks"]
    assert sorted((t["task_type"], t["value"]) for t in tasks) == [("debugging", 1), ("uncategorized", 1)]
    assert [(d["date"], d["requests"]) for d in body["daily"]] == [("2026-09-28", 2)]
