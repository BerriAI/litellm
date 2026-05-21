"""
Unit tests for GET /config/export and POST /config/import endpoints.
"""

import json
import sys
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.management_endpoints.config_export_endpoints import (
    export_config,
    import_config,
)
from litellm.proxy.management_endpoints.config_export_types import (
    ImportRequest,
    ImportResult,
    ImportSectionResult,
    LiteLLMExportEnvelope,
    _redact_litellm_params,
    _validate_dependencies,
    _validate_envelope,
)
from litellm.proxy.management_endpoints.config_import_helpers import (
    _import_section,
    _upsert,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_admin_key() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="test-key", user_role=LitellmUserRoles.PROXY_ADMIN)


def make_viewer_key() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="test-key", user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY
    )


def make_request(base_url: str = "http://testserver") -> MagicMock:
    req = MagicMock()
    req.base_url = base_url
    return req


@contextmanager
def mock_prisma(prisma):
    fake_module = MagicMock()
    fake_module.prisma_client = prisma
    with patch.dict(sys.modules, {"litellm.proxy.proxy_server": fake_module}):
        yield


def make_prisma(
    teams=None,
    models=None,
    mcp_servers=None,
    keys=None,
    credentials=None,
    agents=None,
    guardrails=None,
    tags=None,
    budgets=None,
    orgs=None,
    users=None,
    config_rows=None,
):
    pc = MagicMock()

    def _table(rows):
        t = MagicMock()
        t.find_many = AsyncMock(return_value=rows or [])
        t.find_unique = AsyncMock(return_value=None)
        t.create = AsyncMock()
        t.update = AsyncMock()
        return t

    pc.db.litellm_budgettable = _table(budgets)
    pc.db.litellm_organizationtable = _table(orgs)
    pc.db.litellm_teamtable = _table(teams)
    pc.db.litellm_usertable = _table(users)
    pc.db.litellm_verificationtoken = _table(keys)
    pc.db.litellm_credentialstable = _table(credentials)
    pc.db.litellm_proxymodeltable = _table(models)
    pc.db.litellm_mcpservertable = _table(mcp_servers)
    pc.db.litellm_agentstable = _table(agents)
    pc.db.litellm_guardrailstable = _table(guardrails)
    pc.db.litellm_tagtable = _table(tags)
    pc.db.litellm_config = _table(config_rows)
    return pc


class _NoTxTable:
    """Minimal table stub without a Prisma transaction client (has_tx=False)."""

    def __init__(self):
        self.created = []
        self.updated = []

    async def create(self, data):
        self.created.append(data)

    async def update(self, where, data):
        self.updated.append((where, data))

    async def find_many(self, where=None):
        return []


# ---------------------------------------------------------------------------
# Export — security: litellm_params secrets are redacted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_export_models_redacts_litellm_params_secrets():
    model_row = {
        "model_id": "m1",
        "model_name": "gpt-4",
        "litellm_params": {
            "model": "azure/gpt-4",
            "api_key": "sk-secret",
            "api_base": "https://my.openai.azure.com",
            "aws_secret_access_key": "wJalrX",
        },
    }
    prisma = make_prisma(models=[model_row])

    with mock_prisma(prisma):
        response = await export_config(
            request=make_request(),
            user_api_key_dict=make_admin_key(),
            include="models",
            format="json",
            redact_secrets=True,
            limit=1000,
        )

    body = json.loads(response.body)
    params = body["models"][0]["litellm_params"]
    assert params["api_key"] == "__redacted__"
    assert params["aws_secret_access_key"] == "__redacted__"
    assert params["api_base"] == "https://my.openai.azure.com"  # non-secret kept
    assert params["model"] == "azure/gpt-4"


# ---------------------------------------------------------------------------
# Export — auth: only proxy admins can export
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_export_requires_admin_role():
    with mock_prisma(MagicMock()):
        with pytest.raises(HTTPException) as exc_info:
            await export_config(
                request=make_request(),
                user_api_key_dict=make_viewer_key(),
                include=None,
                format="json",
                redact_secrets=True,
                limit=1000,
            )
    assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# Export — database not connected → HTTP 500
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_export_raises_500_when_db_not_connected():
    fake_module = MagicMock()
    fake_module.prisma_client = None
    with patch.dict(sys.modules, {"litellm.proxy.proxy_server": fake_module}):
        with pytest.raises(HTTPException) as exc_info:
            await export_config(
                request=make_request(),
                user_api_key_dict=make_admin_key(),
                include=None,
                format="json",
                redact_secrets=True,
                limit=1000,
            )
    assert exc_info.value.status_code == 500
    assert "Database not connected" in exc_info.value.detail


# ---------------------------------------------------------------------------
# Import — dry run: reads DB but writes nothing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_import_dry_run_does_not_write():
    prisma = make_prisma()
    envelope = LiteLLMExportEnvelope(
        exported_at="2024-01-01T00:00:00Z",
        source_instance="http://dev",
        include_filters=["teams"],
        teams=[{"team_id": "t1", "team_alias": "alpha"}],
    )
    body = ImportRequest(data=envelope, dry_run=True)

    with mock_prisma(prisma):
        result = await import_config(
            request=make_request(), body=body, user_api_key_dict=make_admin_key()
        )

    assert result.dry_run is True
    assert result.teams.created == 1
    prisma.db.litellm_teamtable.create.assert_not_called()
    prisma.db.litellm_teamtable.update.assert_not_called()


# ---------------------------------------------------------------------------
# Import — full happy path across all sections
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_import_all_sections_smoke(monkeypatch):
    prisma = make_prisma()

    async def fake_import_section(**kwargs):
        sr = kwargs["section_result"]
        sr.created += len(kwargs.get("records", []))
        sr.total_processed += len(kwargs.get("records", []))

    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.config_export_endpoints._import_section",
        fake_import_section,
    )
    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.config_export_endpoints._load_existing",
        AsyncMock(return_value={}),
    )

    envelope = LiteLLMExportEnvelope(
        exported_at="2024-01-01T00:00:00Z",
        source_instance="http://dev",
        include_filters=["*"],
        budgets=[{"budget_id": "b1"}],
        organizations=[{"organization_id": "o1"}],
        teams=[{"team_id": "t1"}],
        users=[{"user_id": "u1"}],
        keys=[{"key_alias": "k1"}],
        credentials=[{"credential_name": "c1", "credential_values": {}}],
        models=[{"model_id": "m1"}],
        mcp_servers=[{"server_id": "s1"}],
        agents=[{"agent_name": "a1"}],
        guardrails=[{"guardrail_name": "g1"}],
        tags=[{"tag_name": "tag1"}],
        general_settings={"max_parallel_requests": 10},
    )
    body = ImportRequest(data=envelope, dry_run=True)

    with mock_prisma(prisma):
        result = await import_config(
            request=make_request(), body=body, user_api_key_dict=make_admin_key()
        )

    assert len(result.sections_attempted) >= 10


# ---------------------------------------------------------------------------
# Import — unhandled exception bubbles as HTTP 500
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_import_handles_exception(monkeypatch):
    prisma = make_prisma()

    async def fail_import_section(**kwargs):
        raise Exception("boom")

    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.config_export_endpoints._import_section",
        fail_import_section,
    )

    envelope = LiteLLMExportEnvelope(
        exported_at="2024-01-01T00:00:00Z",
        source_instance="http://dev",
        include_filters=["teams"],
        teams=[{"team_id": "t1"}],
    )
    body = ImportRequest(data=envelope)

    with mock_prisma(prisma):
        with pytest.raises(HTTPException) as exc:
            await import_config(
                request=make_request(), body=body, user_api_key_dict=make_admin_key()
            )

    assert exc.value.status_code == 500


# ---------------------------------------------------------------------------
# Import — per-section errors do not bleed across sections
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_import_config_per_section_errors_are_isolated(monkeypatch):
    """A record-level error in models does not affect teams result."""

    class _FailCreateTable:
        async def create(self, data):
            raise Exception("models db error")

        async def find_many(self, where=None):
            return []

        async def update(self, where, data):
            pass

    prisma = make_prisma()
    prisma.db.litellm_teamtable.find_many = AsyncMock(return_value=[])
    prisma.db.litellm_proxymodeltable = _FailCreateTable()

    envelope = LiteLLMExportEnvelope(
        exported_at="2024-01-01T00:00:00Z",
        source_instance="http://dev",
        include_filters=["teams", "models"],
        teams=[{"team_id": "t1", "team_alias": "alpha"}],
        models=[{"model_id": "m1", "model_name": "gpt-4"}],
    )
    body = ImportRequest(data=envelope, conflict="skip")

    with mock_prisma(prisma):
        result = await import_config(
            request=make_request(), body=body, user_api_key_dict=make_admin_key()
        )

    assert result.teams.created == 1
    assert result.teams.errors == 0
    assert result.models.errors == 1
    assert result.models.created == 0


# ---------------------------------------------------------------------------
# _upsert — exception is counted as error, not raised
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_exception_counted_as_error():
    table = MagicMock()
    table.create = AsyncMock(side_effect=Exception("constraint violation"))

    sr = ImportSectionResult()
    await _upsert(
        table=table,
        rec={"team_id": "t1", "team_alias": "alpha"},
        id_field="team_id",
        conflict="skip",
        dry_run=False,
        section_result=sr,
        existing_map={},
    )

    assert sr.errors == 1
    assert any("constraint violation" in w for w in sr.warnings)


# ---------------------------------------------------------------------------
# _import_section — transaction rollback restores counts (no double-counting)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_import_section_transaction_rollback_does_not_double_count():
    """
    When the async-with block raises on exit (commit fails), the snapshot/restore
    path must zero out per-record increments before recording the failure.
    Without the fix: created=2, errors=2, total=2 (counts don't add up).
    After the fix: created=0, errors=2, total=2.
    """

    class _TxContextManager:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            raise Exception("commit failed")

    class _TxClient:
        def tx(self):
            return _TxContextManager()

    class _TxTable(_NoTxTable):
        def __init__(self, tx_context):
            super().__init__()
            self._client = tx_context

        def __getattr__(self, name):
            return self

    table = _TxTable(_TxClient())
    sr = ImportSectionResult()
    await _import_section(
        table=table,
        table_name="litellm_teamtable",
        records=[{"team_id": "t1"}, {"team_id": "t2"}],
        id_field="team_id",
        conflict="skip",
        dry_run=False,
        section_result=sr,
        existing_map={},
    )

    assert sr.errors == 2
    assert sr.created == 0
    assert sr.updated == 0
    assert sr.skipped == 0
    assert sr.total_processed == 2
    assert sr.errors + sr.skipped + sr.created + sr.updated == sr.total_processed
    assert any("rolled back" in w for w in sr.warnings)


# ---------------------------------------------------------------------------
# _import_section — transaction failure emits a log entry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_import_section_logs_error_on_transaction_failure():
    class _FailOnExitTx:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            raise Exception("commit failed")

    class _TxClientFail:
        def tx(self):
            return _FailOnExitTx()

    class _TxTableFail(_NoTxTable):
        def __init__(self):
            super().__init__()
            self._client = _TxClientFail()

        def __getattr__(self, name):
            return self

    with patch(
        "litellm.proxy.management_endpoints.config_import_helpers.verbose_proxy_logger"
    ) as mock_logger:
        sr = ImportSectionResult()
        await _import_section(
            table=_TxTableFail(),
            table_name="litellm_teamtable",
            records=[{"team_id": "t1"}],
            id_field="team_id",
            conflict="skip",
            dry_run=False,
            section_result=sr,
            existing_map={},
        )

    mock_logger.error.assert_called_once()
    assert sr.errors == 1
