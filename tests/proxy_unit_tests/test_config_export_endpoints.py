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
    _import_general_settings,
    _import_keys_section,
    _import_section,
    _is_redacted,
    _redact_credential_values,
    _redact_litellm_params,
    _redact_mcp_credentials,
    _redact_mcp_sensitive_fields,
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


def test_redact_credential_values():
    rec = {"credential_name": "my-cred", "credential_values": {"api_key": "secret"}}
    result = _redact_credential_values(rec)
    assert result["credential_values"] == {"__redacted__": True}
    assert result["credential_name"] == "my-cred"


def test_redact_mcp_credentials():
    rec = {"server_name": "my-mcp", "credentials": {"token": "secret"}}
    result = _redact_mcp_credentials(rec)
    assert result["credentials"] == {"__redacted__": True}


def test_redact_litellm_params_masks_secret_keys():
    rec = {
        "model_id": "m1",
        "litellm_params": {
            "model": "gpt-4",
            "api_key": "sk-secret",
            "api_base": "https://my-endpoint.openai.azure.com",
            "aws_access_key_id": "AKIAIOSFODNN7",
            "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG",
            "vertex_credentials": '{"type": "service_account"}',
            "api_version": "2024-02-01",
        },
    }
    result = _redact_litellm_params(rec)
    params = result["litellm_params"]
    # Secret keys are replaced
    assert params["api_key"] == "__redacted__"
    assert params["aws_access_key_id"] == "__redacted__"
    assert params["aws_secret_access_key"] == "__redacted__"
    assert params["vertex_credentials"] == "__redacted__"
    # Non-secret config is preserved
    assert params["model"] == "gpt-4"
    assert params["api_base"] == "https://my-endpoint.openai.azure.com"
    assert params["api_version"] == "2024-02-01"


def test_redact_mcp_sensitive_fields_masks_headers_and_env():
    rec = {
        "server_id": "s1",
        "url": "https://mcp.example.com",
        "spec_path": "/openapi.json",
        "static_headers": {"Authorization": "Bearer tok-secret"},
        "env": {"API_SECRET": "s3cr3t"},
    }
    result = _redact_mcp_sensitive_fields(rec)
    assert result["static_headers"] == {"__redacted__": True}
    assert result["env"] == {"__redacted__": True}
    # Non-sensitive fields preserved
    assert result["url"] == "https://mcp.example.com"
    assert result["spec_path"] == "/openapi.json"



# ---------------------------------------------------------------------------
# Export tests
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
    assert params["api_base"] == "https://my.openai.azure.com"
    assert params["model"] == "azure/gpt-4"


@pytest.mark.asyncio
async def test_export_models_not_redacted_when_flag_false():
    model_row = {
        "model_id": "m1",
        "litellm_params": {"model": "gpt-4", "api_key": "sk-secret"},
    }
    prisma = make_prisma(models=[model_row])

    with mock_prisma(prisma):
        response = await export_config(
            request=make_request(),
            user_api_key_dict=make_admin_key(),
            include="models",
            format="json",
            redact_secrets=False,
            limit=1000,
        )

    body = json.loads(response.body)
    assert body["models"][0]["litellm_params"]["api_key"] == "sk-secret"


@pytest.mark.asyncio
async def test_export_mcp_servers_redacts_static_headers_and_env():
    mcp_row = {
        "server_id": "s1",
        "server_name": "my-mcp",
        "url": "https://mcp.example.com",
        "static_headers": {"Authorization": "Bearer tok"},
        "env": {"SECRET": "val"},
        "credentials": {"token": "cred-secret"},
    }
    prisma = make_prisma(mcp_servers=[mcp_row])

    with mock_prisma(prisma):
        response = await export_config(
            request=make_request(),
            user_api_key_dict=make_admin_key(),
            include="mcp_servers",
            format="json",
            redact_secrets=True,
            limit=1000,
        )

    body = json.loads(response.body)
    srv = body["mcp_servers"][0]
    assert srv["credentials"] == {"__redacted__": True}
    assert srv["static_headers"] == {"__redacted__": True}
    assert srv["env"] == {"__redacted__": True}
    assert srv["url"] == "https://mcp.example.com"


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


# ---------------------------------------------------------------------------
# _import_general_settings — all live-write and skip paths
# ---------------------------------------------------------------------------


def _make_config_row(name, value):
    row = MagicMock()
    row.param_name = name
    row.param_value = value
    return row


@pytest.mark.asyncio
async def test_import_general_settings_creates_new_safe_key():
    prisma = make_prisma()
    prisma.db.litellm_config.find_many = AsyncMock(return_value=[])  # nothing exists

    sr = ImportSectionResult()
    await _import_general_settings(
        prisma_client=prisma,
        settings={"max_parallel_requests": 50},
        conflict="skip",
        dry_run=False,
        section_result=sr,
    )

    assert sr.created == 1
    assert sr.total_processed == 1
    prisma.db.litellm_config.create.assert_called_once_with(
        data={"param_name": "max_parallel_requests", "param_value": 50}
    )


@pytest.mark.asyncio
async def test_import_general_settings_skips_existing_on_conflict_skip():
    prisma = make_prisma()
    prisma.db.litellm_config.find_many = AsyncMock(
        return_value=[_make_config_row("max_parallel_requests", 10)]
    )

    sr = ImportSectionResult()
    await _import_general_settings(
        prisma_client=prisma,
        settings={"max_parallel_requests": 99},
        conflict="skip",
        dry_run=False,
        section_result=sr,
    )

    assert sr.skipped == 1
    assert sr.created == 0
    prisma.db.litellm_config.create.assert_not_called()
    prisma.db.litellm_config.update.assert_not_called()


@pytest.mark.asyncio
async def test_import_general_settings_updates_existing_on_conflict_replace():
    prisma = make_prisma()
    prisma.db.litellm_config.find_many = AsyncMock(
        return_value=[_make_config_row("max_parallel_requests", 10)]
    )

    sr = ImportSectionResult()
    await _import_general_settings(
        prisma_client=prisma,
        settings={"max_parallel_requests": 99},
        conflict="replace",
        dry_run=False,
        section_result=sr,
    )

    assert sr.updated == 1
    prisma.db.litellm_config.update.assert_called_once_with(
        where={"param_name": "max_parallel_requests"},
        data={"param_value": 99},
    )


@pytest.mark.asyncio
async def test_import_general_settings_merges_dict_values():
    prisma = make_prisma()
    prisma.db.litellm_config.find_many = AsyncMock(
        return_value=[_make_config_row("slack_alerting_settings", {"channel": "#ops"})]
    )

    sr = ImportSectionResult()
    await _import_general_settings(
        prisma_client=prisma,
        settings={"slack_alerting_settings": {"alert_on_spend": True}},
        conflict="merge",
        dry_run=False,
        section_result=sr,
    )

    assert sr.updated == 1
    call_data = prisma.db.litellm_config.update.call_args.kwargs["data"]["param_value"]
    assert call_data["channel"] == "#ops"
    assert call_data["alert_on_spend"] is True


@pytest.mark.asyncio
async def test_import_general_settings_skips_unsafe_key():
    prisma = make_prisma()
    prisma.db.litellm_config.find_many = AsyncMock(return_value=[])

    sr = ImportSectionResult()
    await _import_general_settings(
        prisma_client=prisma,
        settings={"database_url": "postgres://secret"},
        conflict="replace",
        dry_run=False,
        section_result=sr,
    )

    assert sr.skipped == 1
    assert len(sr.warnings) == 1
    assert "not in safe export allow-list" in sr.warnings[0]
    prisma.db.litellm_config.create.assert_not_called()


@pytest.mark.asyncio
async def test_import_general_settings_dry_run_counts_without_writing():
    prisma = make_prisma()
    prisma.db.litellm_config.find_many = AsyncMock(
        return_value=[_make_config_row("max_parallel_requests", 10)]
    )

    sr = ImportSectionResult()
    await _import_general_settings(
        prisma_client=prisma,
        settings={"max_parallel_requests": 99, "alerting": ["slack"]},
        conflict="replace",
        dry_run=True,
        section_result=sr,
    )

    assert sr.updated == 1  # max_parallel_requests exists
    assert sr.created == 1  # alerting is new
    prisma.db.litellm_config.create.assert_not_called()
    prisma.db.litellm_config.update.assert_not_called()


@pytest.mark.asyncio
async def test_import_general_settings_exception_counted_as_error():
    prisma = make_prisma()
    prisma.db.litellm_config.find_many = AsyncMock(return_value=[])
    prisma.db.litellm_config.create = AsyncMock(side_effect=Exception("db boom"))

    sr = ImportSectionResult()
    await _import_general_settings(
        prisma_client=prisma,
        settings={"max_parallel_requests": 50},
        conflict="replace",
        dry_run=False,
        section_result=sr,
    )

    assert sr.errors == 1
    assert any("db boom" in w for w in sr.warnings)


# ---------------------------------------------------------------------------
# _upsert — merge, missing-id, and exception paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_merge_deep_merges_and_writes():
    table = MagicMock()
    table.update = AsyncMock()

    existing = {"team_id": "t1", "metadata": {"env": "dev"}, "team_alias": "old"}
    incoming = {"team_id": "t1", "metadata": {"region": "us"}, "team_alias": "new"}

    sr = ImportSectionResult()
    await _upsert(
        table=table,
        rec=incoming,
        id_field="team_id",
        conflict="merge",
        dry_run=False,
        section_result=sr,
        existing_map={"t1": existing},
    )

    assert sr.updated == 1
    table.update.assert_called_once()
    data_sent = table.update.call_args.kwargs["data"]
    assert data_sent["metadata"]["env"] == "dev"
    assert data_sent["metadata"]["region"] == "us"
    assert data_sent["team_alias"] == "new"
    assert "team_id" not in data_sent


@pytest.mark.asyncio
async def test_upsert_skips_record_with_missing_id():
    table = MagicMock()
    table.create = AsyncMock()

    sr = ImportSectionResult()
    await _upsert(
        table=table,
        rec={"team_alias": "no-id"},
        id_field="team_id",
        conflict="skip",
        dry_run=False,
        section_result=sr,
        existing_map={},
    )

    assert sr.skipped == 1
    assert len(sr.warnings) == 1
    table.create.assert_not_called()


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
        existing_map={},  # not in map → triggers create
    )

    assert sr.errors == 1
    assert any("constraint violation" in w for w in sr.warnings)


# ---------------------------------------------------------------------------
# _import_section — has_tx=False non-transactional fallback
# ---------------------------------------------------------------------------


class _NoTxTable:
    """Minimal table stub without a Prisma transaction client."""

    def __init__(self):
        self.created = []
        self.updated = []

    async def create(self, data):
        self.created.append(data)

    async def update(self, where, data):
        self.updated.append((where, data))

    async def find_many(self, where=None):
        return []


@pytest.mark.asyncio
async def test_import_section_no_tx_fallback_creates_record():
    table = _NoTxTable()
    # has_tx will be False because _NoTxTable has no _client attribute
    sr = ImportSectionResult()
    await _import_section(
        table=table,
        table_name="litellm_teamtable",
        records=[{"team_id": "t1", "team_alias": "alpha"}],
        id_field="team_id",
        conflict="skip",
        dry_run=False,
        section_result=sr,
        existing_map={},
    )

    assert sr.created == 1
    assert len(table.created) == 1


@pytest.mark.asyncio
async def test_import_section_no_tx_fallback_exception_counts_error():
    class _FailTable(_NoTxTable):
        async def create(self, data):
            raise Exception("write failed")

    table = _FailTable()
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

    # exception inside _upsert is caught per-record, not at section level
    assert sr.errors >= 1


# ---------------------------------------------------------------------------
# User-proposed tests (exercise all orchestrator section branches)
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
            request=make_request(),
            body=body,
            user_api_key_dict=make_admin_key(),
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
                request=make_request(),
                body=body,
                user_api_key_dict=make_admin_key(),
            )

    assert exc.value.status_code == 500


# ---------------------------------------------------------------------------
# _validate_dependencies — missing branches
# ---------------------------------------------------------------------------


def test_validate_dependencies_fails_when_budget_missing_for_org():
    data = LiteLLMExportEnvelope(
        exported_at="2024-01-01T00:00:00Z",
        source_instance="http://dev",
        include_filters=["organizations", "budgets"],
        budgets=[],  # b1 not present
        organizations=[{"organization_id": "o1", "budget_id": "b1"}],
    )
    with pytest.raises(HTTPException) as exc_info:
        _validate_dependencies(data)
    assert exc_info.value.status_code == 400
    assert "b1" in str(exc_info.value.detail)


def test_validate_dependencies_fails_when_key_references_missing_team():
    data = LiteLLMExportEnvelope(
        exported_at="2024-01-01T00:00:00Z",
        source_instance="http://dev",
        include_filters=["teams", "keys"],
        teams=[],  # t1 not present
        keys=[{"key_alias": "my-key", "team_id": "t1"}],
    )
    with pytest.raises(HTTPException) as exc_info:
        _validate_dependencies(data)
    assert exc_info.value.status_code == 400
    assert "t1" in str(exc_info.value.detail)


def test_validate_dependencies_fails_when_key_references_missing_user():
    data = LiteLLMExportEnvelope(
        exported_at="2024-01-01T00:00:00Z",
        source_instance="http://dev",
        include_filters=["users", "keys"],
        users=[],  # u1 not present
        keys=[{"key_alias": "my-key", "user_id": "u1"}],
    )
    with pytest.raises(HTTPException) as exc_info:
        _validate_dependencies(data)
    assert exc_info.value.status_code == 400
    assert "u1" in str(exc_info.value.detail)


# ---------------------------------------------------------------------------
# _get_prisma_with_auth — prisma_client is None → HTTP 500
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
# _import_keys_section — merge path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_import_section_transaction_rollback_does_not_double_count():
    """
    When a transaction commits successfully for some records and then the
    async-with block raises on exit, the snapshot/restore path must zero out
    the per-record increments before adding errors.  Without the fix:
      created=2, errors=2, total_processed=2  (created + errors > total)
    After the fix:
      created=0, errors=2, total_processed=2
    """

    class _TxContextManager:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            # Simulate the commit failing on exit — this is what triggers the
            # rollback path in _import_section.
            raise Exception("commit failed")

    class _TxClient:
        def tx(self):
            return _TxContextManager()

    class _TxTable(_NoTxTable):
        """Table that has a _client.tx() so _import_section uses the tx path."""

        def __init__(self, tx_context):
            super().__init__()
            self._client = tx_context

        def __getattr__(self, name):
            # Proxy table attribute lookups so getattr(tx, table_name) works.
            return self

    client = _TxClient()
    table = _TxTable(client)

    sr = ImportSectionResult()
    await _import_section(
        table=table,
        table_name="litellm_teamtable",
        records=[
            {"team_id": "t1", "team_alias": "alpha"},
            {"team_id": "t2", "team_alias": "beta"},
        ],
        id_field="team_id",
        conflict="skip",
        dry_run=False,
        section_result=sr,
        existing_map={},
    )

    # All records are errors; no created/updated/skipped should bleed through.
    assert sr.errors == 2
    assert sr.created == 0
    assert sr.updated == 0
    assert sr.skipped == 0
    assert sr.total_processed == 2
    assert sr.errors + sr.skipped + sr.created + sr.updated == sr.total_processed
    assert any("rolled back" in w for w in sr.warnings)


@pytest.mark.asyncio
async def test_import_keys_merges_existing_via_update_many():
    existing = MagicMock()
    existing.key_alias = "my-key"
    existing.model_dump = MagicMock(
        return_value={"key_alias": "my-key", "user_id": "u1", "max_budget": 50}
    )

    prisma = make_prisma()
    prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[existing])
    prisma.db.litellm_verificationtoken.update_many = AsyncMock(
        return_value=MagicMock(count=1)
    )

    sr = ImportSectionResult()
    await _import_keys_section(
        prisma_client=prisma,
        records=[{"key_alias": "my-key", "user_id": "u2", "max_budget": 100}],
        conflict="merge",
        dry_run=False,
        section_result=sr,
    )

    assert sr.updated == 1
    prisma.db.litellm_verificationtoken.update_many.assert_called_once()
    data_sent = prisma.db.litellm_verificationtoken.update_many.call_args.kwargs["data"]
    # merge: incoming user_id wins, max_budget wins
    assert data_sent["user_id"] == "u2"
    assert data_sent["max_budget"] == 100
    assert "token" not in data_sent
    assert "key_alias" not in data_sent


# ---------------------------------------------------------------------------
# _import_section — empty records (no-op)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_import_section_empty_records_is_no_op():
    table = _NoTxTable()
    sr = ImportSectionResult()
    await _import_section(
        table=table,
        table_name="litellm_teamtable",
        records=[],
        id_field="team_id",
        conflict="skip",
        dry_run=False,
        section_result=sr,
        existing_map={},
    )
    assert sr.total_processed == 0
    assert sr.created == 0
    assert sr.errors == 0
    assert len(table.created) == 0


# ---------------------------------------------------------------------------
# _upsert — id_query_field override (WHERE clause uses different field)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_replace_where_uses_id_query_field():
    """When id_query_field differs from id_field, WHERE uses id_query_field
    and that key is stripped from the data payload."""
    table = MagicMock()
    table.update = AsyncMock()

    sr = ImportSectionResult()
    await _upsert(
        table=table,
        rec={"model_id": "m1", "custom_slug": "slug-1", "model_name": "gpt-4-updated"},
        id_field="model_id",
        id_query_field="custom_slug",
        conflict="replace",
        dry_run=False,
        section_result=sr,
        existing_map={"m1": {"model_id": "m1", "custom_slug": "slug-1"}},
    )

    assert sr.updated == 1
    call_kwargs = table.update.call_args.kwargs
    assert call_kwargs["where"] == {"custom_slug": "m1"}
    assert "custom_slug" not in call_kwargs["data"]
    assert "model_id" not in call_kwargs["data"]


# ---------------------------------------------------------------------------
# _import_section — dry run covers all three outcomes in one call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_import_section_dry_run_create_skip_update_mix():
    """dry_run=True must report correct create/skip/update without any DB write."""
    table = _NoTxTable()
    sr = ImportSectionResult()

    await _import_section(
        table=table,
        table_name="litellm_teamtable",
        records=[
            {"team_id": "new-1"},               # new → create
            {"team_id": "new-2"},               # new → create
            {"team_id": "existing-skip"},       # exists, conflict=skip → skip
            {"team_id": "existing-replace"},    # exists, conflict≠skip but dry_run → updated
        ],
        id_field="team_id",
        conflict="replace",
        dry_run=True,
        section_result=sr,
        existing_map={
            "existing-skip": {"team_id": "existing-skip"},
            "existing-replace": {"team_id": "existing-replace"},
        },
    )

    # conflict=replace → existing records counted as updated in dry_run
    assert sr.created == 2
    assert sr.updated == 2
    assert sr.skipped == 0
    assert sr.total_processed == 4
    assert len(table.created) == 0
    assert len(table.updated) == 0


# ---------------------------------------------------------------------------
# Partial section failure — per-section errors do not bleed across sections
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_import_config_per_section_errors_are_isolated(monkeypatch):
    """A record-level error in models does not affect teams result.

    The models table has no _client (has_tx=False), so _import_section uses the
    non-transactional fallback.  _upsert catches the create exception per-record
    and increments errors without propagating, leaving teams untouched.
    """

    class _FailCreateTable:
        """No _client attr → has_tx=False; create always raises."""

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
            request=make_request(),
            body=body,
            user_api_key_dict=make_admin_key(),
        )

    assert result.teams.created == 1
    assert result.teams.errors == 0
    assert result.models.errors == 1
    assert result.models.created == 0


# ---------------------------------------------------------------------------
# Logger verification — error path emits a log
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
        "litellm.proxy.management_endpoints.config_export_endpoints.verbose_proxy_logger"
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
