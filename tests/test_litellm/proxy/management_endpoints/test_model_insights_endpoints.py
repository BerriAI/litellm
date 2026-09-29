from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.model_insights_endpoints import router


def _override_auth() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="sk-test", user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN)


def _grouped_row(
    *, prompt_tokens: str = "100", completion_tokens: str = "200", **dimensions: str
) -> dict[str, object]:
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
