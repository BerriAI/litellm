from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.management_endpoints.online_model_experiments import (
    online_model_experiment_metrics,
)


def _auth(role: LitellmUserRoles) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_role=role)


@pytest.mark.asyncio
async def test_metrics_returns_aggregates_grouped_by_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    query_raw = AsyncMock(
        return_value=[
            {
                "variant": "candidate",
                "requests": 3,
                "subjects": 2,
                "cost": 0.12,
                "total_tokens": 240,
                "latency_ms": 410.5,
                "errors": 1,
            }
        ]
    )
    prisma_client = SimpleNamespace(db=SimpleNamespace(query_raw=query_raw))
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma_client)

    result = await online_model_experiment_metrics(
        experiment_id="support-v1",
        start_date=None,
        end_date=None,
        user_api_key_dict=_auth(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY),
    )

    assert result["experiment_id"] == "support-v1"
    assert result["variants"] == [
        {
            "variant": "candidate",
            "requests": 3,
            "subjects": 2,
            "cost": 0.12,
            "total_tokens": 240,
            "latency_ms": 410.5,
            "errors": 1,
        }
    ]
    query_raw.assert_awaited_once()
    query = query_raw.await_args.args[0]
    assert "metadata->'spend_logs_metadata'->>'online_model_experiment_variant'" in query
    assert "AT TIME ZONE 'UTC'" in query
    assert query_raw.await_args.args[1:] == (
        "support-v1",
        result["start_date"],
        result["end_date"],
    )


@pytest.mark.asyncio
async def test_metrics_rejects_non_admin_roles_before_database_access(monkeypatch: pytest.MonkeyPatch) -> None:
    query_raw = AsyncMock()
    prisma_client = SimpleNamespace(db=SimpleNamespace(query_raw=query_raw))
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma_client)

    with pytest.raises(HTTPException) as error:
        await online_model_experiment_metrics(
            experiment_id="support-v1",
            user_api_key_dict=_auth(LitellmUserRoles.INTERNAL_USER),
        )

    assert error.value.status_code == 403
    query_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_metrics_returns_service_unavailable_without_prisma(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)

    with pytest.raises(HTTPException) as error:
        await online_model_experiment_metrics(
            experiment_id="support-v1",
            user_api_key_dict=_auth(LitellmUserRoles.PROXY_ADMIN),
        )

    assert error.value.status_code == 503
