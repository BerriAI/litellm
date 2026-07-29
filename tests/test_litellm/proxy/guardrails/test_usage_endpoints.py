"""
Tests for the /guardrails/usage/* endpoints backing the dashboard Guardrail Monitor.

Regression (LIT-2529): guardrails defined in config.yaml live only in
IN_MEMORY_GUARDRAIL_HANDLER, so the monitor's overview/detail/logs endpoints —
which read the litellm_guardrailstable Prisma table — could not see them:
detail 404'd, overview omitted them (or rendered them as Custom/Guardrail
orphans), and logs missed their logical-name alias.
"""

import os
import sys
from datetime import datetime
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_registry import InMemoryGuardrailHandler
from litellm.proxy.guardrails.usage_endpoints import (
    guardrails_usage_detail,
    guardrails_usage_logs,
    guardrails_usage_overview,
)
from litellm.types.guardrails import Guardrail, LitellmParams

ADMIN = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
# Query() defaults don't resolve to None when the handler is called directly.
START, END = "2026-04-20", "2026-04-27"


def _config_handler(*guardrails: Guardrail) -> InMemoryGuardrailHandler:
    """A real handler seeded with config-sourced YAML guardrails (no callbacks)."""
    handler = InMemoryGuardrailHandler()
    for g in guardrails:
        gid = g["guardrail_id"]
        handler.IN_MEMORY_GUARDRAILS[gid] = g
        handler._sources[gid] = "config"
    return handler


def _yaml_guardrail(
    guardrail_id: str = "yaml-1",
    name: str = "yaml-pii",
    provider: str = "presidio",
    info: Optional[dict] = None,
) -> Guardrail:
    return Guardrail(
        guardrail_id=guardrail_id,
        guardrail_name=name,
        litellm_params=LitellmParams(guardrail=provider, mode="pre_call"),
        guardrail_info=info if info is not None else {"type": "PII", "description": "yaml-defined"},
    )


def _db_row(guardrail_id: str = "db-1", name: str = "db-guard", provider: str = "aim") -> Any:
    """A Prisma-style row: attribute access, litellm_params/guardrail_info as plain dicts."""
    row = MagicMock(spec=["guardrail_id", "guardrail_name", "litellm_params", "guardrail_info"])
    row.guardrail_id = guardrail_id
    row.guardrail_name = name
    row.litellm_params = {"guardrail": provider, "mode": "pre_call"}
    row.guardrail_info = {"type": "ContentSafety", "description": "db-defined"}
    return row


def _metric(guardrail_id: str, date: str = "2026-04-25", requests: int = 10, passed: int = 8, blocked: int = 2) -> Any:
    m = MagicMock()
    m.guardrail_id = guardrail_id
    m.date = date
    m.requests_evaluated = requests
    m.passed_count = passed
    m.blocked_count = blocked
    m.flagged_count = 0
    return m


def _prisma(
    *,
    find_many=None,
    find_unique=None,
    metrics=None,
    index_find_many=None,
) -> MagicMock:
    client = MagicMock()
    db = client.db
    db.litellm_guardrailstable.find_many = AsyncMock(return_value=find_many or [])
    db.litellm_guardrailstable.find_unique = AsyncMock(return_value=find_unique)
    db.litellm_dailyguardrailmetrics.find_many = AsyncMock(return_value=metrics or [])
    db.litellm_spendlogguardrailindex.find_many = AsyncMock(return_value=index_find_many or [])
    db.litellm_spendlogguardrailindex.count = AsyncMock(return_value=0)
    db.litellm_spendlogs.find_many = AsyncMock(return_value=[])
    return client


def _patches(prisma: MagicMock, handler: InMemoryGuardrailHandler):
    return (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch("litellm.proxy.guardrails.guardrail_registry.IN_MEMORY_GUARDRAIL_HANDLER", handler),
    )


# ---- detail -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_detail_returns_yaml_guardrail_when_db_misses():
    prisma = _prisma(find_unique=None)
    handler = _config_handler(_yaml_guardrail())
    p1, p2 = _patches(prisma, handler)
    with p1, p2:
        resp = await guardrails_usage_detail(
            guardrail_id="yaml-1", start_date=START, end_date=END, user_api_key_dict=ADMIN
        )
    assert resp.guardrail_id == "yaml-1"
    assert resp.guardrail_name == "yaml-pii"
    assert resp.provider == "presidio"  # coerced from the LitellmParams pydantic model
    assert resp.type == "PII"  # from guardrail_info
    assert resp.description == "yaml-defined"


@pytest.mark.asyncio
async def test_detail_404_when_neither_db_nor_config():
    prisma = _prisma(find_unique=None)
    handler = _config_handler()  # empty
    p1, p2 = _patches(prisma, handler)
    with p1, p2, pytest.raises(HTTPException) as exc:
        await guardrails_usage_detail(guardrail_id="ghost", start_date=START, end_date=END, user_api_key_dict=ADMIN)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_detail_does_not_surface_db_sourced_in_memory_entry():
    """A stale in-memory entry (source=db, gone from DB) must 404, not resurface."""
    prisma = _prisma(find_unique=None)
    handler = InMemoryGuardrailHandler()
    stale = _yaml_guardrail(guardrail_id="stale-1", name="stale")
    handler.IN_MEMORY_GUARDRAILS["stale-1"] = stale
    handler._sources["stale-1"] = "db"
    p1, p2 = _patches(prisma, handler)
    with p1, p2, pytest.raises(HTTPException) as exc:
        await guardrails_usage_detail(guardrail_id="stale-1", start_date=START, end_date=END, user_api_key_dict=ADMIN)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_detail_db_row_still_resolves():
    prisma = _prisma(find_unique=_db_row(guardrail_id="db-1", provider="aim"))
    handler = _config_handler()
    p1, p2 = _patches(prisma, handler)
    with p1, p2:
        resp = await guardrails_usage_detail(
            guardrail_id="db-1", start_date=START, end_date=END, user_api_key_dict=ADMIN
        )
    assert resp.provider == "aim"
    assert resp.type == "ContentSafety"


# ---- overview ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_overview_includes_yaml_guardrail_with_no_metrics():
    """The core bug: a YAML guardrail with zero metrics must still appear as a row."""
    prisma = _prisma(find_many=[])  # no DB guardrails
    handler = _config_handler(_yaml_guardrail())
    p1, p2 = _patches(prisma, handler)
    with p1, p2:
        resp = await guardrails_usage_overview(start_date=START, end_date=END, user_api_key_dict=ADMIN)
    rows = [r for r in resp.rows if r.id == "yaml-1"]
    assert len(rows) == 1
    assert rows[0].name == "yaml-pii"
    assert rows[0].provider == "presidio"
    assert rows[0].type == "PII"
    assert rows[0].requestsEvaluated == 0


@pytest.mark.asyncio
async def test_overview_yaml_metrics_matched_by_logical_name():
    """Daily metrics are keyed by logical name; the YAML row must pick them up."""
    prisma = _prisma(
        find_many=[],
        metrics=[_metric("yaml-pii", requests=10, blocked=2)],  # keyed by name, not uuid
    )
    handler = _config_handler(_yaml_guardrail(guardrail_id="yaml-uuid", name="yaml-pii"))
    p1, p2 = _patches(prisma, handler)
    with p1, p2:
        resp = await guardrails_usage_overview(start_date=START, end_date=END, user_api_key_dict=ADMIN)
    rows = [r for r in resp.rows if r.id == "yaml-uuid"]
    assert len(rows) == 1
    assert rows[0].requestsEvaluated == 10
    assert rows[0].failRate == 20.0
    # must not also emit an orphan row keyed by the logical name
    assert [r for r in resp.rows if r.id == "yaml-pii"] == []


@pytest.mark.asyncio
async def test_overview_excludes_db_sourced_in_memory_entry():
    """union must not resurrect a stale db-sourced in-memory guardrail."""
    prisma = _prisma(find_many=[])
    handler = InMemoryGuardrailHandler()
    handler.IN_MEMORY_GUARDRAILS["cfg"] = _yaml_guardrail(guardrail_id="cfg", name="cfg-guard")
    handler._sources["cfg"] = "config"
    handler.IN_MEMORY_GUARDRAILS["stale"] = _yaml_guardrail(guardrail_id="stale", name="stale-guard")
    handler._sources["stale"] = "db"
    p1, p2 = _patches(prisma, handler)
    with p1, p2:
        resp = await guardrails_usage_overview(start_date=START, end_date=END, user_api_key_dict=ADMIN)
    ids = {r.id for r in resp.rows}
    assert "cfg" in ids
    assert "stale" not in ids


# ---- logs -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_logs_resolves_config_guardrail_logical_name():
    """The index query must include the YAML guardrail's logical name alias."""
    prisma = _prisma(find_unique=None)
    handler = _config_handler(_yaml_guardrail(guardrail_id="yaml-uuid", name="yaml-pii"))
    p1, p2 = _patches(prisma, handler)
    with p1, p2:
        await guardrails_usage_logs(
            guardrail_id="yaml-uuid",
            policy_id=None,
            page=1,
            page_size=50,
            action=None,
            start_date=START,
            end_date=END,
            user_api_key_dict=ADMIN,
        )
    where = prisma.db.litellm_spendlogguardrailindex.find_many.call_args.kwargs["where"]
    assert where["guardrail_id"] == {"in": ["yaml-uuid", "yaml-pii"]}
