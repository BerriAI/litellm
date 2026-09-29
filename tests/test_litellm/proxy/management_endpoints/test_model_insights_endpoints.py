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
    task = _grouped_row(
        task_type="chat",
        model_group="fast-chat",
        model="openai/gpt-5.4-mini",
        custom_llm_provider="openai",
    )
    table = MagicMock()
    table.group_by = AsyncMock(side_effect=[[model, prompt_heavy_model], [daily], [task]])
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
    assert response.json()["by_task"][0]["task_type"] == "chat"
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


def _call(table: MagicMock, query: str) -> object:
    prisma = MagicMock()
    prisma.db.litellm_dailymodelusage = table
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = _override_auth
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        return TestClient(app).get(f"/model-insights?start_date=2026-09-01&end_date=2026-09-28&{query}")


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


def test_model_insights_scopes_daily_to_ranked_deployments_but_tasks_to_all_usage() -> None:
    ranked = _grouped_row(model_group="shared", model="m1", custom_llm_provider="openai")
    table = MagicMock()
    table.group_by = AsyncMock(side_effect=[[ranked], [], []])

    _call(table, "metric=tokens")

    daily_where = table.group_by.await_args_list[1].kwargs["where"]
    task_where = table.group_by.await_args_list[2].kwargs["where"]
    assert daily_where["OR"] == [{"model_group": "shared", "model": "m1", "custom_llm_provider": "openai"}]
    assert "model_group" not in daily_where
    assert "OR" not in task_where
    assert task_where["date"] == daily_where["date"]
    assert "take" not in table.group_by.await_args_list[2].kwargs


def test_model_insights_task_breakdown_does_not_change_with_the_chart_metric() -> None:
    def rows() -> list[list[dict[str, object]]]:
        a = _grouped_row(model_group="a", model="m1", custom_llm_provider="openai")
        b = _grouped_row(model_group="b", model="m2", custom_llm_provider="openai")
        task = _grouped_row(task_type="debugging", model_group="b", model="m2", custom_llm_provider="openai")
        return [[a, b], [], [task]]

    by_tokens = _call(MagicMock(group_by=AsyncMock(side_effect=rows())), "metric=tokens").json()
    by_requests = _call(MagicMock(group_by=AsyncMock(side_effect=rows())), "metric=requests").json()

    assert by_tokens["by_task"] == by_requests["by_task"]


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
    assert sorted((t["task_type"], t["requests"]) for t in body["by_task"]) == [("debugging", 1), ("uncategorized", 1)]
    assert [(d["date"], d["requests"]) for d in body["daily"]] == [("2026-09-28", 2)]
