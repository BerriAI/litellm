"""Unit tests for GET /config/export and POST /config/import endpoints."""

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
    _resolve_sections,
    _validate_dependencies,
    _validate_envelope,
)
from litellm.proxy.management_endpoints.config_import_helpers import (
    _encrypt_dict_field,
    _import_section,
    _upsert,
)


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
    teams=None, models=None, mcp_servers=None, keys=None, credentials=None,
    agents=None, guardrails=None, tags=None, budgets=None, orgs=None,
    users=None, config_rows=None,
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


class _FailCommitTxContext:
    """Async context manager whose __aexit__ always raises (simulates commit failure)."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        raise Exception("commit failed")


class _FailCommitTxClient:
    def tx(self):
        return _FailCommitTxContext()


class _TxTable(_NoTxTable):
    """_NoTxTable with a fail-on-commit transaction client attached."""

    def __init__(self, client=None):
        super().__init__()
        self._client = client or _FailCommitTxClient()

    def __getattr__(self, name):
        return self


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
    assert params["api_base"] == "https://my.openai.azure.com"
    assert params["model"] == "azure/gpt-4"


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


@pytest.mark.asyncio
async def test_import_config_per_section_errors_are_isolated(monkeypatch):
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


@pytest.mark.asyncio
async def test_import_section_transaction_rollback_does_not_double_count():
    sr = ImportSectionResult()
    await _import_section(
        table=_TxTable(),
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


@pytest.mark.asyncio
async def test_import_section_logs_error_on_transaction_failure():
    with patch(
        "litellm.proxy.management_endpoints.config_import_helpers.verbose_proxy_logger"
    ) as mock_logger:
        sr = ImportSectionResult()
        await _import_section(
            table=_TxTable(),
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


@pytest.mark.asyncio
async def test_import_does_not_write_redacted_placeholders_to_db(monkeypatch):
    captured: list = []

    async def fake_import_section(**kwargs):
        captured.extend(kwargs["records"])
        sr = kwargs["section_result"]
        sr.created += len(kwargs["records"])
        sr.total_processed += len(kwargs["records"])

    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.config_export_endpoints._import_section",
        fake_import_section,
    )
    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.config_export_endpoints._load_existing",
        AsyncMock(return_value={}),
    )

    # Bypass real encryption (requires a master key) — this test checks stripping
    # of redacted sentinels, not encryption correctness.
    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.config_import_helpers.encrypt_value_helper",
        lambda v: v,
    )

    envelope = LiteLLMExportEnvelope(
        exported_at="2024-01-01T00:00:00Z",
        source_instance="http://dev",
        include_filters=["models", "mcp_servers"],
        models=[{
            "model_id": "m1",
            "litellm_params": {
                "model": "azure/gpt-4",
                "api_key": "__redacted__",
                "aws_secret_access_key": "__redacted__",
                "api_base": "https://my.openai.azure.com",
            },
        }],
        mcp_servers=[{
            "server_id": "s1",
            "server_name": "my-mcp",
            "url": "https://mcp.example.com",
            "credentials": {"__redacted__": True},
            "static_headers": {"__redacted__": True},
            "env": {"__redacted__": True},
        }],
    )
    body = ImportRequest(data=envelope, dry_run=False, conflict="replace")

    with mock_prisma(make_prisma()):
        result = await import_config(
            request=make_request(), body=body, user_api_key_dict=make_admin_key()
        )

    # models — string-sentinel sub-fields stripped from litellm_params
    model_rec = next(r for r in captured if r.get("model_id") == "m1")
    params = model_rec["litellm_params"]
    assert "api_key" not in params
    assert "aws_secret_access_key" not in params
    assert params["api_base"] == "https://my.openai.azure.com"

    # mcp_servers — dict-form sentinel fields stripped, non-secret kept
    mcp_rec = next(r for r in captured if r.get("server_id") == "s1")
    assert "credentials" not in mcp_rec
    assert "static_headers" not in mcp_rec
    assert "env" not in mcp_rec
    assert mcp_rec["url"] == "https://mcp.example.com"

    # warnings emitted for each stripped MCP field
    warns = result.mcp_servers.warnings
    assert any("credentials" in w for w in warns)
    assert any("static_headers" in w for w in warns)
    assert any("env" in w for w in warns)


@pytest.mark.asyncio
async def test_upsert_replace_merge_and_edge_cases():
    t = MagicMock()
    t.update = AsyncMock()
    sr = ImportSectionResult()
    await _upsert(t, {"team_id": "t1", "x": 1}, "team_id", "replace", False, sr, {"t1": {}})
    assert sr.updated == 1 and t.update.called
    sr2 = ImportSectionResult()
    await _upsert(t, {"team_id": "t2", "x": 2}, "team_id", "merge", False, sr2, {"t2": {"team_id": "t2", "x": 0}})
    assert sr2.updated == 1
    sr3 = ImportSectionResult()
    await _upsert(t, {"team_id": "t3"}, "team_id", "skip", True, sr3, {"t3": {}})
    assert sr3.skipped == 1
    sr4 = ImportSectionResult()
    await _upsert(t, {}, "team_id", "skip", False, sr4, {})
    assert sr4.skipped == 1 and sr4.warnings


def test_resolve_sections_rejects_unknown_section():
    with pytest.raises(HTTPException) as exc:
        _resolve_sections("not_a_real_section")
    assert exc.value.status_code == 400


def test_encrypt_dict_field(monkeypatch):
    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.config_import_helpers.encrypt_value_helper",
        lambda v: f"enc:{v}",
    )
    # encrypts values in the named field
    rec = {"credential_name": "c1", "credential_values": {"api_key": "sk-secret", "region": "us-east-1"}}
    result = _encrypt_dict_field(rec, "credential_values")
    assert result["credential_values"] == {"api_key": "enc:sk-secret", "region": "enc:us-east-1"}
    assert result["credential_name"] == "c1"  # other fields untouched

    # returns record unchanged when field is absent
    assert _encrypt_dict_field({"x": 1}, "credential_values") == {"x": 1}

    # returns record unchanged when field is not a dict
    assert _encrypt_dict_field({"credential_values": None}, "credential_values") == {"credential_values": None}

    # returns record unchanged when dict is empty
    assert _encrypt_dict_field({"credential_values": {}}, "credential_values") == {"credential_values": {}}


def test_validate_envelope_rejects_duplicate_ids():
    with pytest.raises(HTTPException) as exc:
        _validate_envelope(LiteLLMExportEnvelope(
            exported_at="2024-01-01T00:00:00Z", source_instance="http://dev",
            include_filters=[], teams=[{"team_id": "t1"}, {"team_id": "t1"}],
        ))
    assert exc.value.status_code == 400
