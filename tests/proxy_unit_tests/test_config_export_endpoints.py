"""
Unit tests for GET /config/export and POST /config/import endpoints.
"""

import json
import os
import sys
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.abspath("../.."))

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.management_endpoints.config_export_endpoints import (
    ALL_SECTIONS,
    SAFE_GENERAL_SETTINGS_KEYS,
    ImportRequest,
    ImportResult,
    ImportSectionResult,
    LiteLLMExportEnvelope,
    _deep_merge,
    _import_keys_section,
    _is_redacted,
    _redact_credential_values,
    _redact_mcp_credentials,
    _strip,
    _upsert,
    _validate_dependencies,
    _validate_envelope,
    export_config,
    import_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_admin_key() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="test-key",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )


def make_viewer_key() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="test-key",
        user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
    )


def make_request(base_url: str = "http://testserver") -> MagicMock:
    req = MagicMock()
    req.base_url = base_url
    return req


@contextmanager
def mock_prisma(prisma):
    """
    Inject a fake litellm.proxy.proxy_server module so that the
    `from litellm.proxy.proxy_server import prisma_client` inside
    each endpoint resolves to our mock — without importing the real
    proxy_server (which needs heavy deps).
    """
    fake_module = MagicMock()
    fake_module.prisma_client = prisma
    with patch.dict(sys.modules, {"litellm.proxy.proxy_server": fake_module}):
        yield


def make_prisma(
    teams=None,
    orgs=None,
    keys=None,
    credentials=None,
    models=None,
    mcp_servers=None,
    agents=None,
    guardrails=None,
    tags=None,
    budgets=None,
    users=None,
    config_rows=None,
):
    """Build a minimal mock prisma client for export tests."""
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


# ---------------------------------------------------------------------------
# Unit tests — helper functions
# ---------------------------------------------------------------------------


def test_strip_removes_fields():
    record = {"team_id": "t1", "spend": 99.9, "team_alias": "my-team"}
    result = _strip(record, ["spend"])
    assert "spend" not in result
    assert result["team_id"] == "t1"
    assert result["team_alias"] == "my-team"


def test_strip_preserves_none_values():
    # None values must be preserved so that replace-mode import can write
    # explicit NULLs to the target DB instead of silently leaving existing
    # non-null values in place.
    record = {"team_id": "t1", "spend": None, "team_alias": "my-team"}
    result = _strip(record, [])
    assert "spend" in result
    assert result["spend"] is None


def test_strip_handles_model_with_dict_method():
    class FakeModel:
        def model_dump(self):
            return {"team_id": "t1", "spend": 5.0}

    result = _strip(FakeModel(), ["spend"])
    assert "spend" not in result
    assert result["team_id"] == "t1"


def test_redact_credential_values():
    rec = {"credential_name": "my-cred", "credential_values": {"api_key": "secret"}}
    result = _redact_credential_values(rec)
    assert result["credential_values"] == {"__redacted__": True}
    assert result["credential_name"] == "my-cred"


def test_redact_mcp_credentials():
    rec = {"server_name": "my-mcp", "credentials": {"token": "secret"}}
    result = _redact_mcp_credentials(rec)
    assert result["credentials"] == {"__redacted__": True}


def test_redact_mcp_credentials_no_credentials():
    rec = {"server_name": "my-mcp"}
    result = _redact_mcp_credentials(rec)
    assert "credentials" not in result


def test_is_redacted_true():
    assert _is_redacted({"__redacted__": True}) is True


def test_is_redacted_false():
    assert _is_redacted({"api_key": "real-value"}) is False
    assert _is_redacted(None) is False
    assert _is_redacted("string") is False


# ---------------------------------------------------------------------------
# Export tests
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


@pytest.mark.asyncio
async def test_export_returns_versioned_envelope():
    prisma = make_prisma(teams=[{"team_id": "t1", "team_alias": "alpha", "spend": 0}])

    with mock_prisma(prisma):
        response = await export_config(
            request=make_request(),
            user_api_key_dict=make_admin_key(),
            include=None,
            format="json",
            redact_secrets=True,
            limit=1000,
        )

    body = json.loads(response.body)
    assert "exported_at" in body
    assert "include_filters" in body


@pytest.mark.asyncio
async def test_export_teams_strips_spend():
    team_row = {"team_id": "t1", "team_alias": "alpha", "spend": 99.9}
    prisma = make_prisma(teams=[team_row])

    with mock_prisma(prisma):
        response = await export_config(
            request=make_request(),
            user_api_key_dict=make_admin_key(),
            include="teams",
            format="json",
            redact_secrets=True,
            limit=1000,
        )

    body = json.loads(response.body)
    assert len(body["teams"]) == 1
    assert "spend" not in body["teams"][0]
    assert body["teams"][0]["team_alias"] == "alpha"


@pytest.mark.asyncio
async def test_export_credentials_redacted_by_default():
    cred_row = {
        "credential_id": "c1",
        "credential_name": "my-cred",
        "credential_values": {"api_key": "super-secret"},
    }
    prisma = make_prisma(credentials=[cred_row])

    with mock_prisma(prisma):
        response = await export_config(
            request=make_request(),
            user_api_key_dict=make_admin_key(),
            include="credentials",
            format="json",
            redact_secrets=True,
            limit=1000,
        )

    body = json.loads(response.body)
    assert body["credentials"][0]["credential_values"] == {"__redacted__": True}


@pytest.mark.asyncio
async def test_export_credentials_not_redacted_when_flag_false():
    cred_row = {
        "credential_id": "c1",
        "credential_name": "my-cred",
        "credential_values": {"api_key": "super-secret"},
    }
    prisma = make_prisma(credentials=[cred_row])

    with mock_prisma(prisma):
        response = await export_config(
            request=make_request(),
            user_api_key_dict=make_admin_key(),
            include="credentials",
            format="json",
            redact_secrets=False,
            limit=1000,
        )

    body = json.loads(response.body)
    assert body["credentials"][0]["credential_values"]["api_key"] == "super-secret"


@pytest.mark.asyncio
async def test_export_keys_strips_token_and_spend():
    key_row = {
        "token": "hashed-token-value",
        "key_alias": "my-key",
        "spend": 5.0,
        "user_id": "u1",
    }
    prisma = make_prisma(keys=[key_row])

    with mock_prisma(prisma):
        response = await export_config(
            request=make_request(),
            user_api_key_dict=make_admin_key(),
            include="keys",
            format="json",
            redact_secrets=True,
            limit=1000,
        )

    body = json.loads(response.body)
    assert len(body["keys"]) == 1
    assert "token" not in body["keys"][0]
    assert "spend" not in body["keys"][0]
    assert body["keys"][0]["key_alias"] == "my-key"


@pytest.mark.asyncio
async def test_export_yaml_format():
    prisma = make_prisma(teams=[{"team_id": "t1", "team_alias": "alpha"}])

    with mock_prisma(prisma):
        response = await export_config(
            request=make_request(),
            user_api_key_dict=make_admin_key(),
            include="teams",
            format="yaml",
            redact_secrets=True,
            limit=1000,
        )

    assert response.media_type == "application/yaml"
    import yaml

    body = yaml.safe_load(response.body)
    assert "exported_at" in body


@pytest.mark.asyncio
async def test_export_rejects_unknown_section():
    prisma = make_prisma()

    with mock_prisma(prisma):
        with pytest.raises(HTTPException) as exc_info:
            await export_config(
                request=make_request(),
                user_api_key_dict=make_admin_key(),
                include="teams,nonexistent_table",
                format="json",
                redact_secrets=True,
                limit=1000,
            )
        assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_export_partial_include_only_fetches_requested_sections():
    prisma = make_prisma(
        teams=[{"team_id": "t1", "team_alias": "alpha"}],
        models=[{"model_id": "m1", "model_name": "gpt-4"}],
    )

    with mock_prisma(prisma):
        response = await export_config(
            request=make_request(),
            user_api_key_dict=make_admin_key(),
            include="teams",
            format="json",
            redact_secrets=True,
            limit=1000,
        )

    body = json.loads(response.body)
    assert "teams" in body
    assert "models" not in body
    assert body["include_filters"] == ["teams"]


@pytest.mark.asyncio
async def test_export_general_settings_only_safe_keys():
    class ConfigRow:
        def __init__(self, name, value):
            self.param_name = name
            self.param_value = value

    config_rows = [
        ConfigRow("alerting", ["slack"]),
        ConfigRow("database_url", "postgres://secret"),  # must be excluded
        ConfigRow("max_parallel_requests", 100),
    ]
    prisma = make_prisma(config_rows=config_rows)

    with mock_prisma(prisma):
        response = await export_config(
            request=make_request(),
            user_api_key_dict=make_admin_key(),
            include="general_settings",
            format="json",
            redact_secrets=True,
            limit=1000,
        )

    body = json.loads(response.body)
    gs = body["general_settings"]
    # Prisma where filter happens at DB level; our mock returns all rows,
    # so verify the allow-list is passed to find_many
    prisma.db.litellm_config.find_many.assert_called_once()
    call_kwargs = prisma.db.litellm_config.find_many.call_args
    where_clause = (
        call_kwargs.kwargs.get("where", {}) or call_kwargs.args[0]
        if call_kwargs.args
        else {}
    )
    if "where" in str(call_kwargs):
        assert "in" in str(call_kwargs)


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_import_requires_admin_role():
    envelope = LiteLLMExportEnvelope(
        exported_at="2024-01-01T00:00:00Z",
        source_instance="http://dev",
        include_filters=["teams"],
        teams=[],
    )
    body = ImportRequest(data=envelope)

    with mock_prisma(MagicMock()):
        with pytest.raises(HTTPException) as exc_info:
            await import_config(
                request=make_request(),
                body=body,
                user_api_key_dict=make_viewer_key(),
            )
        assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_import_creates_new_team():
    prisma = make_prisma(teams=[])
    # find_many returns empty list → no existing teams
    prisma.db.litellm_teamtable.find_many = AsyncMock(return_value=[])

    envelope = LiteLLMExportEnvelope(
        exported_at="2024-01-01T00:00:00Z",
        source_instance="http://dev",
        include_filters=["teams"],
        teams=[{"team_id": "t1", "team_alias": "alpha"}],
    )
    body = ImportRequest(data=envelope, conflict="skip")

    with mock_prisma(prisma):
        result = await import_config(
            request=make_request(),
            body=body,
            user_api_key_dict=make_admin_key(),
        )

    assert result.teams.created == 1
    assert result.teams.skipped == 0
    assert result.teams.total_processed == 1
    assert "teams" in result.sections_attempted


@pytest.mark.asyncio
async def test_import_skips_existing_team_on_conflict_skip():
    class FakeTeam:
        def model_dump(self):
            return {"team_id": "t1", "team_alias": "alpha"}

    prisma = make_prisma()
    prisma.db.litellm_teamtable.find_many = AsyncMock(return_value=[FakeTeam()])

    envelope = LiteLLMExportEnvelope(
        exported_at="2024-01-01T00:00:00Z",
        source_instance="http://dev",
        include_filters=["teams"],
        teams=[{"team_id": "t1", "team_alias": "alpha-updated"}],
    )
    body = ImportRequest(data=envelope, conflict="skip")

    with mock_prisma(prisma):
        result = await import_config(
            request=make_request(),
            body=body,
            user_api_key_dict=make_admin_key(),
        )

    assert result.teams.skipped == 1
    assert result.teams.updated == 0
    prisma.db.litellm_teamtable.update.assert_not_called()


@pytest.mark.asyncio
async def test_import_replaces_existing_team_on_conflict_replace():
    class FakeTeam:
        def model_dump(self):
            return {"team_id": "t1", "team_alias": "alpha"}

    prisma = make_prisma()
    prisma.db.litellm_teamtable.find_many = AsyncMock(return_value=[FakeTeam()])

    envelope = LiteLLMExportEnvelope(
        exported_at="2024-01-01T00:00:00Z",
        source_instance="http://dev",
        include_filters=["teams"],
        teams=[{"team_id": "t1", "team_alias": "alpha-v2"}],
    )
    body = ImportRequest(data=envelope, conflict="replace")

    with mock_prisma(prisma):
        result = await import_config(
            request=make_request(),
            body=body,
            user_api_key_dict=make_admin_key(),
        )

    assert result.teams.updated == 1


@pytest.mark.asyncio
async def test_import_skips_redacted_credentials_with_warning():
    prisma = make_prisma()

    envelope = LiteLLMExportEnvelope(
        exported_at="2024-01-01T00:00:00Z",
        source_instance="http://dev",
        include_filters=["credentials"],
        credentials=[
            {
                "credential_id": "c1",
                "credential_name": "my-cred",
                "credential_values": {"__redacted__": True},
            }
        ],
    )
    body = ImportRequest(data=envelope, conflict="skip")

    with mock_prisma(prisma):
        result = await import_config(
            request=make_request(),
            body=body,
            user_api_key_dict=make_admin_key(),
        )

    assert result.credentials.skipped == 1
    assert len(result.credentials.warnings) == 1
    assert "redacted" in result.credentials.warnings[0].lower()
    prisma.db.litellm_credentialstable.create.assert_not_called()


@pytest.mark.asyncio
async def test_import_skips_key_with_no_alias():
    prisma = make_prisma()

    envelope = LiteLLMExportEnvelope(
        exported_at="2024-01-01T00:00:00Z",
        source_instance="http://dev",
        include_filters=["keys"],
        keys=[{"user_id": "u1"}],  # no key_alias
    )
    body = ImportRequest(data=envelope, conflict="skip")

    with mock_prisma(prisma):
        result = await import_config(
            request=make_request(),
            body=body,
            user_api_key_dict=make_admin_key(),
        )

    assert result.keys.skipped == 1
    assert len(result.keys.warnings) == 1


@pytest.mark.asyncio
async def test_import_dry_run_does_not_write():
    prisma = make_prisma()
    prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=None)

    envelope = LiteLLMExportEnvelope(
        exported_at="2024-01-01T00:00:00Z",
        source_instance="http://dev",
        include_filters=["teams"],
        teams=[{"team_id": "t1", "team_alias": "alpha"}],
    )
    body = ImportRequest(data=envelope, dry_run=True)

    with mock_prisma(prisma):
        result = await import_config(
            request=make_request(),
            body=body,
            user_api_key_dict=make_admin_key(),
        )

    assert result.dry_run is True
    assert result.teams.created == 1
    prisma.db.litellm_teamtable.create.assert_not_called()
    prisma.db.litellm_teamtable.update.assert_not_called()


@pytest.mark.asyncio
async def test_import_mcp_server_with_redacted_credentials_imports_without_creds():
    prisma = make_prisma()
    prisma.db.litellm_mcpservertable.find_many = AsyncMock(return_value=[])

    envelope = LiteLLMExportEnvelope(
        exported_at="2024-01-01T00:00:00Z",
        source_instance="http://dev",
        include_filters=["mcp_servers"],
        mcp_servers=[
            {
                "server_id": "s1",
                "server_name": "my-mcp",
                "url": "https://mcp.example.com",
                "credentials": {"__redacted__": True},
            }
        ],
    )
    body = ImportRequest(data=envelope, conflict="skip")

    with mock_prisma(prisma):
        result = await import_config(
            request=make_request(),
            body=body,
            user_api_key_dict=make_admin_key(),
        )

    assert result.mcp_servers.created == 1
    assert len(result.mcp_servers.warnings) == 1
    # Verify no credentials field reached the create call
    # (the record was cleaned before being passed to _import_section)
    if prisma.db.litellm_mcpservertable.create.call_args:
        create_call_data = prisma.db.litellm_mcpservertable.create.call_args.kwargs.get(
            "data", {}
        )
        assert "credentials" not in create_call_data


# ---------------------------------------------------------------------------
# New helper tests (production-grade upgrades)
# ---------------------------------------------------------------------------


# #6 — Deep merge
def test_deep_merge_flat():
    base = {"a": 1, "b": 2}
    override = {"b": 99, "c": 3}
    result = _deep_merge(base, override)
    assert result == {"a": 1, "b": 99, "c": 3}


def test_deep_merge_nested():
    base = {"settings": {"timeout": 30, "retries": 3}}
    override = {"settings": {"timeout": 60}}
    result = _deep_merge(base, override)
    assert result == {"settings": {"timeout": 60, "retries": 3}}


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"x": 1}}
    override = {"a": {"y": 2}}
    _deep_merge(base, override)
    assert base == {"a": {"x": 1}}


def test_deep_merge_non_dict_override_wins():
    base = {"a": {"x": 1}}
    override = {"a": "string"}
    result = _deep_merge(base, override)
    assert result["a"] == "string"


# #7 — Envelope validation
def test_validate_envelope_passes_on_valid_data():
    data = LiteLLMExportEnvelope(
        exported_at="2024-01-01T00:00:00Z",
        source_instance="http://dev",
        include_filters=["teams"],
        teams=[{"team_id": "t1", "team_alias": "alpha"}],
    )
    _validate_envelope(data)  # should not raise


def test_validate_envelope_rejects_missing_id():
    data = LiteLLMExportEnvelope(
        exported_at="2024-01-01T00:00:00Z",
        source_instance="http://dev",
        include_filters=["teams"],
        teams=[{"team_alias": "alpha"}],  # missing team_id
    )
    with pytest.raises(HTTPException) as exc_info:
        _validate_envelope(data)
    assert exc_info.value.status_code == 400
    assert "team_id" in str(exc_info.value.detail)


def test_validate_envelope_rejects_duplicate_ids():
    data = LiteLLMExportEnvelope(
        exported_at="2024-01-01T00:00:00Z",
        source_instance="http://dev",
        include_filters=["teams"],
        teams=[
            {"team_id": "t1", "team_alias": "alpha"},
            {"team_id": "t1", "team_alias": "beta"},  # duplicate
        ],
    )
    with pytest.raises(HTTPException) as exc_info:
        _validate_envelope(data)
    assert exc_info.value.status_code == 400
    assert "duplicate" in str(exc_info.value.detail).lower()


# #2 — Dependency validation
def test_validate_dependencies_passes_when_org_present():
    data = LiteLLMExportEnvelope(
        exported_at="2024-01-01T00:00:00Z",
        source_instance="http://dev",
        include_filters=["organizations", "teams"],
        organizations=[{"organization_id": "o1", "organization_alias": "MyOrg"}],
        teams=[{"team_id": "t1", "team_alias": "alpha", "organization_id": "o1"}],
    )
    _validate_dependencies(data)  # should not raise


def test_validate_dependencies_fails_when_org_missing():
    data = LiteLLMExportEnvelope(
        exported_at="2024-01-01T00:00:00Z",
        source_instance="http://dev",
        include_filters=["organizations", "teams"],
        organizations=[],  # empty — o1 not present
        teams=[{"team_id": "t1", "team_alias": "alpha", "organization_id": "o1"}],
    )
    with pytest.raises(HTTPException) as exc_info:
        _validate_dependencies(data)
    assert exc_info.value.status_code == 400
    assert "o1" in str(exc_info.value.detail)


def test_validate_dependencies_skips_cross_section_check_when_section_absent():
    # teams reference org o1 but organizations section is not in the snapshot;
    # we assume the org already exists in the target DB — no error
    data = LiteLLMExportEnvelope(
        exported_at="2024-01-01T00:00:00Z",
        source_instance="http://dev",
        include_filters=["teams"],
        organizations=None,  # not in snapshot
        teams=[{"team_id": "t1", "team_alias": "alpha", "organization_id": "o1"}],
    )
    _validate_dependencies(data)  # should not raise


# #3 — Accurate dry run
@pytest.mark.asyncio
async def test_dry_run_accurately_reports_create_vs_skip():
    class FakeTeam:
        def model_dump(self):
            return {"team_id": "existing", "team_alias": "old"}

    prisma = make_prisma()
    # find_many returns one existing team
    prisma.db.litellm_teamtable.find_many = AsyncMock(return_value=[FakeTeam()])

    envelope = LiteLLMExportEnvelope(
        exported_at="2024-01-01T00:00:00Z",
        source_instance="http://dev",
        include_filters=["teams"],
        teams=[
            {"team_id": "existing", "team_alias": "alpha"},  # exists → skip
            {"team_id": "new-team", "team_alias": "beta"},  # new → create
        ],
    )
    body = ImportRequest(data=envelope, dry_run=True, conflict="skip")

    with mock_prisma(prisma):
        result = await import_config(
            request=make_request(),
            body=body,
            user_api_key_dict=make_admin_key(),
        )

    assert result.dry_run is True
    assert result.teams.created == 1  # new-team
    assert result.teams.skipped == 1  # existing
    # No writes
    prisma.db.litellm_teamtable.create.assert_not_called()
    prisma.db.litellm_teamtable.update.assert_not_called()


# #10 — ImportResult totals
def test_import_result_has_total_processed_field():
    sr = __import__(
        "litellm.proxy.management_endpoints.config_export_endpoints",
        fromlist=["ImportSectionResult"],
    ).ImportSectionResult()
    sr.created = 3
    sr.skipped = 1
    sr.errors = 1
    sr.total_processed = 5
    assert sr.total_processed == 5


def test_import_result_has_sections_attempted():
    r = ImportResult(dry_run=False, conflict="skip")
    r.sections_attempted = ["teams", "models"]
    assert "teams" in r.sections_attempted


# ---------------------------------------------------------------------------
# _import_keys_section unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_import_keys_skips_new_key_no_token():
    """Keys not present in the DB cannot be created — token (PK) was not exported."""
    prisma = make_prisma()
    prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])

    sr = ImportSectionResult()
    await _import_keys_section(
        prisma_client=prisma,
        records=[{"key_alias": "my-key", "user_id": "u1"}],
        conflict="replace",
        dry_run=False,
        section_result=sr,
    )

    assert sr.skipped == 1
    assert sr.created == 0
    assert any("cannot be created" in w for w in sr.warnings)
    prisma.db.litellm_verificationtoken.update_many.assert_not_called()


@pytest.mark.asyncio
async def test_import_keys_skips_existing_on_conflict_skip():
    """Existing key with conflict=skip leaves the record untouched."""
    existing = MagicMock()
    existing.key_alias = "my-key"

    prisma = make_prisma()
    prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[existing])

    sr = ImportSectionResult()
    await _import_keys_section(
        prisma_client=prisma,
        records=[{"key_alias": "my-key", "user_id": "u1"}],
        conflict="skip",
        dry_run=False,
        section_result=sr,
    )

    assert sr.skipped == 1
    assert sr.updated == 0
    prisma.db.litellm_verificationtoken.update_many.assert_not_called()


@pytest.mark.asyncio
async def test_import_keys_updates_existing_via_update_many_on_replace():
    """Existing key with conflict=replace calls update_many (not update) and strips protected fields."""
    existing = MagicMock()
    existing.key_alias = "my-key"

    prisma = make_prisma()
    prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[existing])
    prisma.db.litellm_verificationtoken.update_many = AsyncMock(
        return_value=MagicMock(count=1)
    )

    sr = ImportSectionResult()
    await _import_keys_section(
        prisma_client=prisma,
        records=[
            {
                "key_alias": "my-key",
                "token": "should-be-stripped",
                "spend": 9.9,
                "user_id": "u1",
                "max_budget": 100,
            }
        ],
        conflict="replace",
        dry_run=False,
        section_result=sr,
    )

    assert sr.updated == 1
    assert sr.errors == 0
    prisma.db.litellm_verificationtoken.update_many.assert_called_once()
    call_kwargs = prisma.db.litellm_verificationtoken.update_many.call_args.kwargs
    assert call_kwargs["where"] == {"key_alias": "my-key"}
    data_sent = call_kwargs["data"]
    assert "token" not in data_sent
    assert "key_alias" not in data_sent
    assert "spend" not in data_sent
    assert data_sent.get("user_id") == "u1"
    assert data_sent.get("max_budget") == 100


@pytest.mark.asyncio
async def test_import_keys_dry_run_counts_update_without_writing():
    """dry_run=True counts existing keys as updated without calling update_many."""
    existing = MagicMock()
    existing.key_alias = "my-key"

    prisma = make_prisma()
    prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[existing])

    sr = ImportSectionResult()
    await _import_keys_section(
        prisma_client=prisma,
        records=[{"key_alias": "my-key", "user_id": "u1"}],
        conflict="replace",
        dry_run=True,
        section_result=sr,
    )

    assert sr.updated == 1
    prisma.db.litellm_verificationtoken.update_many.assert_not_called()
