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


def _register_real_config_guardrail(monkeypatch, *, guardrail_id, name, guardrail_info):
    from litellm.proxy.guardrails import guardrail_registry
    from litellm.types.guardrails import Guardrail, LitellmParams

    handler = guardrail_registry.InMemoryGuardrailHandler()
    handler.initialize_guardrail(
        guardrail=Guardrail(
            guardrail_id=guardrail_id,
            guardrail_name=name,
            litellm_params=LitellmParams(
                guardrail="bedrock", mode="pre_call", default_on=False
            ),
            guardrail_info=guardrail_info,
        ),
        source="config",
    )
    monkeypatch.setattr(guardrail_registry, "IN_MEMORY_GUARDRAIL_HANDLER", handler)


@pytest.mark.asyncio
async def test_usage_overview_uses_provider_for_config_guardrails(monkeypatch):
    _patch_common(
        monkeypatch,
        metrics=[_metric(guardrail_id="presidio-pii-mask-test")],
    )

    response = await usage_endpoints.guardrails_usage_overview(
        start_date="2026-06-19",
        end_date="2026-06-20",
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
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


@pytest.mark.asyncio
async def test_usage_detail_raises_404_when_guardrail_absent(monkeypatch):
    from fastapi import HTTPException

    _patch_common(monkeypatch, metrics=[], detail_unique=None)
    monkeypatch.setattr(usage_endpoints, "_get_config_loaded_guardrails", lambda: [])

    with pytest.raises(HTTPException) as exc_info:
        await usage_endpoints.guardrails_usage_detail(
            guardrail_id="missing-guardrail",
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
        )

    assert exc_info.value.status_code == 404


def test_get_config_loaded_guardrails_filters_by_source(monkeypatch):
    from litellm.proxy.guardrails import guardrail_registry

    handler = guardrail_registry.InMemoryGuardrailHandler()
    handler.IN_MEMORY_GUARDRAILS = {
        "cfg-1": {"guardrail_id": "cfg-1", "guardrail_name": "config-one"},
        "db-1": {"guardrail_id": "db-1", "guardrail_name": "db-one"},
        "no-source": {"guardrail_id": "no-source", "guardrail_name": "no-source"},
    }
    handler._sources = {"cfg-1": "config", "db-1": "db"}
    monkeypatch.setattr(guardrail_registry, "IN_MEMORY_GUARDRAIL_HANDLER", handler)

    result = usage_endpoints._get_config_loaded_guardrails()

    returned_ids = {g["guardrail_id"] for g in result}
    assert "cfg-1" in returned_ids
    assert "db-1" not in returned_ids
    assert "no-source" not in returned_ids


def test_merge_config_loaded_guardrails_dedupes_and_appends(monkeypatch):
    db_guardrails = [
        SimpleNamespace(guardrail_id="shared-id", guardrail_name="shared-name")
    ]
    monkeypatch.setattr(
        usage_endpoints,
        "_get_config_loaded_guardrails",
        lambda: [
            {"guardrail_id": "shared-id", "guardrail_name": "shared-name"},
            {"guardrail_id": "cfg-only", "guardrail_name": "cfg-only-name"},
        ],
    )

    merged = usage_endpoints._merge_config_loaded_guardrails(db_guardrails)

    ids = [usage_endpoints._get_guardrail_attrs(g)[0] for g in merged]
    assert ids == ["shared-id", "cfg-only"]


def test_find_config_loaded_guardrail_returns_none_when_absent(monkeypatch):
    monkeypatch.setattr(
        usage_endpoints,
        "_get_config_loaded_guardrails",
        lambda: [{"guardrail_id": "cfg-1", "guardrail_name": "config-one"}],
    )

    assert usage_endpoints._find_config_loaded_guardrail("does-not-exist") is None


def test_get_guardrail_litellm_params_handles_model_dump_and_missing():
    pydantic_like = SimpleNamespace(
        litellm_params=SimpleNamespace(
            model_dump=lambda exclude_none: {"guardrail": "presidio"}
        )
    )
    assert usage_endpoints._get_guardrail_litellm_params(pydantic_like) == {
        "guardrail": "presidio"
    }

    assert usage_endpoints._get_guardrail_litellm_params(SimpleNamespace()) == {}


@pytest.mark.asyncio
async def test_usage_detail_surfaces_guardrail_info_through_real_handler(monkeypatch):
    """End-to-end through the real in-memory handler (no mocking of
    _get_config_loaded_guardrails): description, type and provider must come
    from the config guardrail's persisted guardrail_info / litellm_params."""
    monkeypatch.setitem(
        sys.modules,
        "litellm.proxy.proxy_server",
        SimpleNamespace(prisma_client=object()),
    )
    monkeypatch.setattr(
        usage_endpoints, "GuardrailsRepository", _repo(_Table(rows=[], unique=None))
    )
    monkeypatch.setattr(
        usage_endpoints,
        "DailyGuardrailMetricsRepository",
        _repo(_Table(rows=[_metric(guardrail_id="bedrock-guard")])),
    )
    _register_real_config_guardrail(
        monkeypatch,
        guardrail_id="bedrock-guard-uuid",
        name="bedrock-guard",
        guardrail_info={"description": "blocks PII", "type": "bedrock"},
    )

    response = await usage_endpoints.guardrails_usage_detail(
        guardrail_id="bedrock-guard",
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
    )

    assert response.guardrail_name == "bedrock-guard"
    assert response.description == "blocks PII"
    assert response.type == "bedrock"
    assert response.provider == "bedrock"


@pytest.mark.asyncio
async def test_usage_logs_resolves_config_guardrail_by_uuid(monkeypatch):
    """A config guardrail queried by UUID has no DB row, so the logs endpoint
    must fall back to the in-memory list and add its logical name to the
    SpendLogs index filter; otherwise the logs tab is always empty."""
    captured = {}

    class _IndexTable:
        async def find_many(self, *args, **kwargs):
            captured["where"] = kwargs.get("where")
            return []

        async def count(self, *args, **kwargs):
            return 0

    monkeypatch.setitem(
        sys.modules,
        "litellm.proxy.proxy_server",
        SimpleNamespace(prisma_client=object()),
    )
    monkeypatch.setattr(
        usage_endpoints, "GuardrailsRepository", _repo(_Table(rows=[], unique=None))
    )
    monkeypatch.setattr(
        usage_endpoints,
        "SpendLogGuardrailIndexRepository",
        _repo(_IndexTable()),
    )
    _register_real_config_guardrail(
        monkeypatch,
        guardrail_id="bedrock-guard-uuid",
        name="bedrock-guard",
        guardrail_info={"description": "blocks PII"},
    )

    response = await usage_endpoints.guardrails_usage_logs(
        guardrail_id="bedrock-guard-uuid",
        policy_id=None,
        page=1,
        page_size=50,
        action=None,
        start_date=None,
        end_date=None,
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
    )

    assert response.total == 0
    assert captured["where"]["guardrail_id"] == {
        "in": ["bedrock-guard-uuid", "bedrock-guard"]
    }
