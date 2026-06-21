import sys
from types import SimpleNamespace

import pytest

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.guardrails import usage_endpoints


class _Table:
    def __init__(self, *, rows=None, unique=None):
        self.rows = rows or []
        self.unique = unique

    async def find_many(self, *args, **kwargs):
        return self.rows

    async def find_unique(self, *args, **kwargs):
        return self.unique


def _repo(table):
    return lambda prisma_client: SimpleNamespace(table=table)


def _metric(
    *,
    guardrail_id: str,
    date: str = "2026-06-20",
    requests_evaluated: int = 10,
    passed_count: int = 8,
    blocked_count: int = 2,
    flagged_count: int = 0,
):
    return SimpleNamespace(
        guardrail_id=guardrail_id,
        date=date,
        requests_evaluated=requests_evaluated,
        passed_count=passed_count,
        blocked_count=blocked_count,
        flagged_count=flagged_count,
    )


def _config_guardrail():
    return {
        "guardrail_id": "config-presidio-id",
        "guardrail_name": "presidio-pii-mask-test",
        "litellm_params": {
            "guardrail": "presidio",
            "mode": ["pre_call", "post_call"],
            "default_on": False,
        },
        "guardrail_info": {"description": "Config loaded PII guardrail"},
    }


def _patch_common(monkeypatch, *, metrics, detail_unique=None):
    monkeypatch.setitem(
        sys.modules,
        "litellm.proxy.proxy_server",
        SimpleNamespace(prisma_client=object()),
    )
    monkeypatch.setattr(
        usage_endpoints,
        "GuardrailsRepository",
        _repo(_Table(rows=[], unique=detail_unique)),
    )
    monkeypatch.setattr(
        usage_endpoints,
        "DailyGuardrailMetricsRepository",
        _repo(_Table(rows=metrics)),
    )
    monkeypatch.setattr(
        usage_endpoints,
        "_get_config_loaded_guardrails",
        lambda: [_config_guardrail()],
    )


@pytest.mark.asyncio
async def test_usage_overview_uses_provider_for_config_guardrails(monkeypatch):
    _patch_common(
        monkeypatch,
        metrics=[_metric(guardrail_id="presidio-pii-mask-test")],
    )

    response = await usage_endpoints.guardrails_usage_overview(
        start_date="2026-06-19",
        end_date="2026-06-20",
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    )

    assert len(response.rows) == 1
    row = response.rows[0]
    assert row.id == "config-presidio-id"
    assert row.name == "presidio-pii-mask-test"
    assert row.provider == "presidio"
    assert row.requestsEvaluated == 10
    assert row.failRate == 20.0


@pytest.mark.asyncio
async def test_usage_detail_resolves_config_guardrail_by_name(monkeypatch):
    _patch_common(
        monkeypatch,
        metrics=[_metric(guardrail_id="presidio-pii-mask-test")],
        detail_unique=None,
    )

    response = await usage_endpoints.guardrails_usage_detail(
        guardrail_id="presidio-pii-mask-test",
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
    )

    assert response.guardrail_id == "presidio-pii-mask-test"
    assert response.guardrail_name == "presidio-pii-mask-test"
    assert response.provider == "presidio"
    assert response.description == "Config loaded PII guardrail"
    assert response.requestsEvaluated == 10
    assert response.failRate == 20.0
    assert response.time_series == [
        {"date": "2026-06-20", "passed": 8, "blocked": 2, "score": None}
    ]
