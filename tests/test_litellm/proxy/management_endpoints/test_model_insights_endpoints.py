from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.db.model_usage_rollup import increment_daily_model_usage
from litellm.proxy.management_endpoints.model_insights_endpoints import router


def _override_auth() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="sk-test", user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN)


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
    table.group_by = AsyncMock(side_effect=[[model, prompt_heavy_model], [daily]])
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
    assert table.group_by.await_count == 2
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
    table.group_by = AsyncMock(side_effect=[[token_heavy, request_heavy], []])

    by_requests = _call(table, "metric=requests").json()
    by_tokens = _call(
        MagicMock(group_by=AsyncMock(side_effect=[[token_heavy, request_heavy], []])), "metric=tokens"
    ).json()

    assert by_requests["top_models"][0]["model_group"] == "busy"
    assert by_tokens["top_models"][0]["model_group"] == "big"


def test_model_insights_scopes_daily_to_ranked_deployments() -> None:
    ranked = _grouped_row(model_group="shared", model="m1", custom_llm_provider="openai")
    table = MagicMock()
    table.group_by = AsyncMock(side_effect=[[ranked], []])

    _call(table, "metric=tokens")

    daily_where = table.group_by.await_args_list[1].kwargs["where"]
    assert daily_where["OR"] == [{"model_group": "shared", "model": "m1", "custom_llm_provider": "openai"}]
    assert "model_group" not in daily_where


def _task_rows() -> list[dict[str, object]]:
    def row(task: str, group: str, requests: str, spend: float) -> dict[str, object]:
        base = _grouped_row(task_type=task, model_group=group, model=group, custom_llm_provider="openai")
        base["_sum"].update({"request_count": requests, "spend": spend})  # type: ignore[union-attr]
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


class _InMemoryUsageTable:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, ...], dict[str, float]] = {}

    async def upsert(self, where: dict, data: dict) -> None:
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


@pytest.mark.asyncio
async def test_model_insights_reads_back_what_the_rollup_wrote() -> None:
    table = _InMemoryUsageTable()
    prisma = MagicMock()
    prisma.db.litellm_dailymodelusage = table
    payload = {
        "call_type": "acompletion",
        "spend": 0.5,
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "startTime": datetime(2026, 9, 28, tzinfo=timezone.utc),
        "model": "gpt-5",
        "model_group": "gpt-5",
        "metadata": "{}",
        "request_tags": '["task:debugging"]',
        "custom_llm_provider": "openai",
        "status": "success",
    }

    await increment_daily_model_usage(prisma, payload)
    await increment_daily_model_usage(prisma, {**payload, "request_tags": "[]"})

    body = _call(table, "metric=requests").json()

    assert [(m["model_group"], m["requests"], m["prompt_tokens"]) for m in body["top_models"]] == [("gpt-5", 2, 20)]
    tasks = _call(table, "metric=requests", path="/model-insights/tasks").json()["tasks"]
    assert sorted((t["task_type"], t["value"]) for t in tasks) == [("debugging", 1), ("uncategorized", 1)]
    assert [(d["date"], d["requests"]) for d in body["daily"]] == [("2026-09-28", 2)]
