"""
Tests for /guardrails/usage/* endpoints serving the dashboard Guardrail Monitor.

Regression: LIT-2529 — guardrails defined in YAML (config) were invisible to
/guardrails/usage/detail and only partially visible to /guardrails/usage/overview.
"""

import os
import sys
from datetime import datetime
from typing import Optional
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.guardrails.usage_endpoints import (
    guardrails_usage_detail,
    guardrails_usage_logs,
    guardrails_usage_overview,
)
from litellm.types.guardrails import Guardrail, LitellmParams

ADMIN_AUTH = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)

# When calling FastAPI handlers directly (not via TestClient), Query() defaults
# don't resolve to None, so explicit date strings are required.
START_DATE = "2026-04-20"
END_DATE = "2026-04-27"


def _make_yaml_guardrail(
    guardrail_id: str = "yaml-gr-1",
    guardrail_name: str = "my-yaml-pii",
    provider: str = "presidio",
    info: Optional[dict] = None,
) -> Guardrail:
    """Build a Guardrail TypedDict matching what InMemoryGuardrailHandler stores."""
    return Guardrail(
        guardrail_id=guardrail_id,
        guardrail_name=guardrail_name,
        litellm_params=LitellmParams(guardrail=provider, mode="pre_call"),
        guardrail_info=info or {"type": "PII", "description": "YAML-defined"},
    )


def _make_db_guardrail(
    guardrail_id: str = "db-gr-1",
    guardrail_name: str = "db-aim",
    provider: str = "aim",
):
    """Build a mock Prisma row (object with attribute access)."""

    class _Row:
        pass

    row = _Row()
    row.guardrail_id = guardrail_id
    row.guardrail_name = guardrail_name
    row.litellm_params = {"guardrail": provider, "mode": "pre_call"}
    row.guardrail_info = {"type": "ContentSafety", "description": "DB-defined"}
    row.created_at = datetime.now()
    row.updated_at = datetime.now()
    return row


def _make_metric_row(
    guardrail_id: str,
    date: str = "2026-04-25",
    requests: int = 10,
    passed: int = 8,
    blocked: int = 2,
    flagged: int = 0,
):
    class _M:
        pass

    m = _M()
    m.guardrail_id = guardrail_id
    m.date = date
    m.requests_evaluated = requests
    m.passed_count = passed
    m.blocked_count = blocked
    m.flagged_count = flagged
    return m


@pytest.fixture
def mock_prisma(mocker):
    client = mocker.Mock()
    client.db = mocker.Mock()
    client.db.litellm_guardrailstable = mocker.Mock()
    client.db.litellm_guardrailstable.find_many = AsyncMock(return_value=[])
    client.db.litellm_guardrailstable.find_unique = AsyncMock(return_value=None)
    client.db.litellm_dailyguardrailmetrics = mocker.Mock()
    client.db.litellm_dailyguardrailmetrics.find_many = AsyncMock(return_value=[])
    client.db.litellm_spendlogguardrailindex = mocker.Mock()
    client.db.litellm_spendlogguardrailindex.find_many = AsyncMock(return_value=[])
    client.db.litellm_spendlogguardrailindex.count = AsyncMock(return_value=0)
    client.db.litellm_spendlogs = mocker.Mock()
    client.db.litellm_spendlogs.find_many = AsyncMock(return_value=[])
    mocker.patch("litellm.proxy.proxy_server.prisma_client", client)
    return client


@pytest.fixture
def mock_in_memory(mocker):
    handler = mocker.Mock()
    handler.list_in_memory_guardrails.return_value = []
    handler.get_guardrail_by_id.return_value = None
    mocker.patch(
        "litellm.proxy.guardrails.guardrail_registry.IN_MEMORY_GUARDRAIL_HANDLER",
        handler,
    )
    # Also patch the symbol re-imported into usage_endpoints (after the fix lands)
    mocker.patch(
        "litellm.proxy.guardrails.usage_endpoints.IN_MEMORY_GUARDRAIL_HANDLER",
        handler,
        create=True,
    )
    return handler


# ---- /guardrails/usage/detail ----------------------------------------------


@pytest.mark.asyncio
async def test_usage_detail_returns_yaml_guardrail(mock_prisma, mock_in_memory):
    """YAML guardrail (DB miss, in-memory hit) should return 200 with details."""
    yaml_gr = _make_yaml_guardrail()
    mock_in_memory.get_guardrail_by_id.return_value = yaml_gr

    response = await guardrails_usage_detail(
        guardrail_id="yaml-gr-1", user_api_key_dict=ADMIN_AUTH
    )

    assert response.guardrail_id == "yaml-gr-1"
    assert response.guardrail_name == "my-yaml-pii"
    assert response.provider == "presidio"
    assert response.type == "PII"
    assert response.description == "YAML-defined"


@pytest.mark.asyncio
async def test_usage_detail_db_takes_precedence_over_yaml(mock_prisma, mock_in_memory):
    """When both DB and in-memory have the same id, DB wins (parity with get_guardrail_info)."""
    db_row = _make_db_guardrail(guardrail_id="shared-id", provider="bedrock")
    mock_prisma.db.litellm_guardrailstable.find_unique = AsyncMock(return_value=db_row)
    mock_in_memory.get_guardrail_by_id.return_value = _make_yaml_guardrail(
        guardrail_id="shared-id", provider="presidio"
    )

    response = await guardrails_usage_detail(
        guardrail_id="shared-id", user_api_key_dict=ADMIN_AUTH
    )

    assert response.provider == "bedrock"
    # In-memory handler should not be consulted when DB lookup hits
    mock_in_memory.get_guardrail_by_id.assert_not_called()


@pytest.mark.asyncio
async def test_usage_detail_404_when_neither_db_nor_yaml(mock_prisma, mock_in_memory):
    """If both miss, raise 404."""
    with pytest.raises(HTTPException) as exc:
        await guardrails_usage_detail(
            guardrail_id="ghost", user_api_key_dict=ADMIN_AUTH
        )
    assert exc.value.status_code == 404


# ---- /guardrails/usage/overview --------------------------------------------


@pytest.mark.asyncio
async def test_usage_overview_includes_yaml_guardrails_with_no_metrics(
    mock_prisma, mock_in_memory
):
    """YAML guardrails must appear in overview rows even with zero metrics."""
    yaml_gr = _make_yaml_guardrail(
        guardrail_id="yaml-gr-1",
        guardrail_name="my-yaml-pii",
        provider="presidio",
        info={"type": "PII", "description": "YAML-defined"},
    )
    mock_in_memory.list_in_memory_guardrails.return_value = [yaml_gr]

    response = await guardrails_usage_overview(
        start_date=START_DATE, end_date=END_DATE, user_api_key_dict=ADMIN_AUTH
    )

    matching = [r for r in response.rows if r.id == "yaml-gr-1"]
    assert len(matching) == 1, f"Expected one row for yaml-gr-1, got {response.rows}"
    row = matching[0]
    assert row.name == "my-yaml-pii"
    assert row.provider == "presidio"
    assert row.type == "PII"
    assert row.requestsEvaluated == 0


@pytest.mark.asyncio
async def test_usage_overview_yaml_guardrail_metric_lookup_by_logical_name(
    mock_prisma, mock_in_memory
):
    """Metrics are keyed by logical name; row must pick them up via name lookup."""
    yaml_gr = _make_yaml_guardrail(
        guardrail_id="yaml-uuid-xyz", guardrail_name="my-yaml-pii"
    )
    mock_in_memory.list_in_memory_guardrails.return_value = [yaml_gr]
    # metric row keyed by the logical name (how the spend-log writer keys YAML guardrails)
    mock_prisma.db.litellm_dailyguardrailmetrics.find_many = AsyncMock(
        side_effect=[
            [_make_metric_row("my-yaml-pii", requests=10, blocked=2)],  # current
            [],  # previous
        ]
    )

    response = await guardrails_usage_overview(
        start_date=START_DATE, end_date=END_DATE, user_api_key_dict=ADMIN_AUTH
    )

    matching = [r for r in response.rows if r.id == "yaml-uuid-xyz"]
    assert len(matching) == 1
    row = matching[0]
    assert row.requestsEvaluated == 10
    assert row.failRate == 20.0
    assert row.provider == "presidio"
    # Should NOT also produce an orphan "Custom" row keyed by the logical name
    orphan_rows = [r for r in response.rows if r.id == "my-yaml-pii"]
    assert (
        orphan_rows == []
    ), "Logical-name row should be merged into the YAML guardrail row"


@pytest.mark.asyncio
async def test_usage_overview_dedupe_when_guardrail_in_both_db_and_yaml(
    mock_prisma, mock_in_memory
):
    """If the same guardrail_id appears in DB and in-memory, only one row is emitted."""
    db_row = _make_db_guardrail(
        guardrail_id="shared-id", guardrail_name="shared-name", provider="bedrock"
    )
    yaml_gr = _make_yaml_guardrail(
        guardrail_id="shared-id",
        guardrail_name="shared-name",
        provider="presidio",
    )
    mock_prisma.db.litellm_guardrailstable.find_many = AsyncMock(return_value=[db_row])
    mock_in_memory.list_in_memory_guardrails.return_value = [yaml_gr]

    response = await guardrails_usage_overview(
        start_date=START_DATE, end_date=END_DATE, user_api_key_dict=ADMIN_AUTH
    )

    matching = [r for r in response.rows if r.id == "shared-id"]
    assert len(matching) == 1
    # DB row wins
    assert matching[0].provider == "bedrock"


# ---- /guardrails/usage/logs ------------------------------------------------


@pytest.mark.asyncio
async def test_usage_logs_includes_logical_name_for_yaml_guardrail(
    mock_prisma, mock_in_memory
):
    """For a YAML guardrail (DB miss), the logs query must include the logical name alias."""
    yaml_gr = _make_yaml_guardrail(
        guardrail_id="yaml-uuid-xyz", guardrail_name="my-yaml-pii"
    )
    mock_in_memory.get_guardrail_by_id.return_value = yaml_gr

    captured: dict = {}

    async def _capture_find_many(**kwargs):
        captured.update(kwargs)
        return []

    mock_prisma.db.litellm_spendlogguardrailindex.find_many = AsyncMock(
        side_effect=_capture_find_many
    )

    await guardrails_usage_logs(
        guardrail_id="yaml-uuid-xyz",
        policy_id=None,
        page=1,
        page_size=50,
        action=None,
        start_date=None,
        end_date=None,
        user_api_key_dict=ADMIN_AUTH,
    )

    where = captured.get("where", {})
    gid_filter = where.get("guardrail_id")
    # The query should accept either the UUID or the logical name
    assert (
        isinstance(gid_filter, dict) and "in" in gid_filter
    ), f"Expected 'in' filter to include UUID + logical name, got {gid_filter}"
    assert "yaml-uuid-xyz" in gid_filter["in"]
    assert "my-yaml-pii" in gid_filter["in"]


# ---- Integration: real InMemoryGuardrailHandler ----------------------------


@pytest.mark.asyncio
async def test_usage_detail_with_real_in_memory_handler_preserves_guardrail_info(
    mock_prisma, mocker
):
    """
    Regression: `initialize_guardrail` must persist `guardrail_info` into
    IN_MEMORY_GUARDRAILS so that /usage/detail can render `type` and `description`
    for YAML-defined guardrails. Exercises the real handler (not a Mock).
    """
    from litellm.proxy.guardrails.guardrail_registry import (
        IN_MEMORY_GUARDRAIL_HANDLER,
    )

    # Bypass callback initialization; we only care about what gets stored.
    mocker.patch.object(
        IN_MEMORY_GUARDRAIL_HANDLER,
        "initialize_custom_guardrail",
        return_value=None,
    )

    yaml_input = {
        "guardrail_id": "real-handler-yaml",
        "guardrail_name": "real-handler-pii",
        # `module.Class` form routes to initialize_custom_guardrail (mocked above)
        "litellm_params": {
            "guardrail": "my_module.MyCustomGuardrail",
            "mode": "pre_call",
        },
        "guardrail_info": {"type": "PII", "description": "Real-handler-defined"},
    }
    try:
        IN_MEMORY_GUARDRAIL_HANDLER.initialize_guardrail(guardrail=yaml_input)

        response = await guardrails_usage_detail(
            guardrail_id="real-handler-yaml",
            start_date=START_DATE,
            end_date=END_DATE,
            user_api_key_dict=ADMIN_AUTH,
        )

        assert response.guardrail_id == "real-handler-yaml"
        assert response.provider == "my_module.MyCustomGuardrail"
        assert response.type == "PII"
        assert response.description == "Real-handler-defined"
    finally:
        IN_MEMORY_GUARDRAIL_HANDLER.IN_MEMORY_GUARDRAILS.pop("real-handler-yaml", None)
        IN_MEMORY_GUARDRAIL_HANDLER.guardrail_id_to_custom_guardrail.pop(
            "real-handler-yaml", None
        )
