"""
Tests for partial-update semantics of PUT /v1/mcp/server.

A partial update must only write the fields the caller explicitly provided.
Omitting a field must NOT reset it to its Pydantic schema default (e.g.
``transport=sse``, ``mcp_access_groups=[]``, ``allow_all_keys=False``), which
would silently overwrite the existing DB row.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma import Json

from litellm.proxy._experimental.mcp_server.db import (
    create_mcp_server,
    update_mcp_server,
)
from litellm.proxy._types import NewMCPServerRequest, UpdateMCPServerRequest


def _credentials_cleared(value) -> bool:
    """The clear sentinel after the edge translation: prisma Json(None) (SQL null) or a bare None."""
    return value is None or (isinstance(value, Json) and getattr(value, "data", "x") is None)


def _mock_prisma():
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_mcpservertable = AsyncMock()
    mock_prisma.db.litellm_mcpservertable.update = AsyncMock(return_value=MagicMock())
    mock_prisma.db.litellm_mcpservertable.create = AsyncMock(return_value=MagicMock())
    return mock_prisma


async def _run_update(data: UpdateMCPServerRequest, fields_set=None) -> dict:
    mock_prisma = _mock_prisma()
    await update_mcp_server(mock_prisma, data, "test-user", fields_set=fields_set)
    return mock_prisma.db.litellm_mcpservertable.update.call_args[1]["data"]


@pytest.mark.asyncio
async def test_partial_update_omits_unset_defaultful_fields():
    """
    A PUT touching only allowed_tools must not write transport,
    mcp_access_groups, allow_all_keys, available_on_public_internet,
    delegate_auth_to_upstream, is_byok, args, env or byok_description.
    """
    data = UpdateMCPServerRequest(
        server_id="my-test-server",
        allowed_tools=["foo"],
    )

    data_dict = await _run_update(data)

    # The intended change is present.
    assert data_dict["allowed_tools"] == ["foo"]

    # Fields the caller did not provide must not be in the write payload, so the
    # existing DB value is preserved.
    for trapped_field in (
        "transport",
        "mcp_access_groups",
        "allow_all_keys",
        "available_on_public_internet",
        "delegate_auth_to_upstream",
        "is_byok",
        "args",
        "env",
        "byok_description",
    ):
        assert trapped_field not in data_dict, (
            f"{trapped_field} should not be written on a partial update that "
            f"omitted it (would reset the row to a schema default)"
        )


@pytest.mark.asyncio
async def test_partial_update_null_tool_name_maps_clear_to_empty_json():
    """Explicit null on Json map fields must clear overrides (UI legacy)."""
    data = UpdateMCPServerRequest(
        server_id="my-test-server",
        tool_name_to_display_name=None,
        tool_name_to_description=None,
    )

    data_dict = await _run_update(data)

    assert data_dict["tool_name_to_display_name"] == "{}"
    assert data_dict["tool_name_to_description"] == "{}"


@pytest.mark.asyncio
async def test_partial_update_null_allowed_tools_clears_whitelist():
    """Explicit null must clear the whitelist (UI legacy); Prisma requires []."""
    data = UpdateMCPServerRequest(
        server_id="my-test-server",
        allowed_tools=None,
    )

    data_dict = await _run_update(data)

    assert data_dict["allowed_tools"] == []


@pytest.mark.asyncio
async def test_partial_update_preserves_http_transport():
    """The reported prod incident: a PUT without transport must not flip http->sse."""
    data = UpdateMCPServerRequest(
        server_id="atlassian_url",
        allowed_tools=[],
    )

    data_dict = await _run_update(data)

    assert "transport" not in data_dict
    assert data_dict["allowed_tools"] == []


@pytest.mark.asyncio
async def test_partial_update_writes_explicitly_provided_fields():
    """Explicitly provided fields are written, including falsy/default-equal values."""
    data = UpdateMCPServerRequest(
        server_id="my-test-server",
        url="https://example.com/mcp",
        transport="http",
        allow_all_keys=False,
        mcp_access_groups=["mcp-dev-sandbox"],
        available_on_public_internet=True,
    )

    data_dict = await _run_update(data)

    assert data_dict["transport"] == "http"
    # Explicitly provided False must still be written.
    assert data_dict["allow_all_keys"] is False
    assert data_dict["mcp_access_groups"] == ["mcp-dev-sandbox"]
    assert data_dict["available_on_public_internet"] is True


@pytest.mark.asyncio
async def test_partial_update_can_explicitly_reset_allow_all_keys():
    """Caller can still reset a field to its default by sending it explicitly."""
    enabled = await _run_update(UpdateMCPServerRequest(server_id="s", allow_all_keys=True))
    assert enabled["allow_all_keys"] is True

    disabled = await _run_update(UpdateMCPServerRequest(server_id="s", allow_all_keys=False))
    assert disabled["allow_all_keys"] is False


@pytest.mark.asyncio
async def test_partial_update_does_not_clear_alias_when_unset():
    """alias is force-normalized on the payload; an unset/None alias must not be written."""
    data = UpdateMCPServerRequest(
        server_id="my-test-server",
        allowed_tools=["foo"],
    )
    fields_set = set(data.fields_set())
    # Simulate validate_and_normalize_mcp_server_payload assigning alias=None.
    data.alias = None

    data_dict = await _run_update(data, fields_set=fields_set)

    assert "alias" not in data_dict


@pytest.mark.asyncio
async def test_partial_update_can_explicitly_clear_alias():
    """Caller can clear an existing alias by explicitly sending alias=None."""
    data = UpdateMCPServerRequest(
        server_id="my-test-server",
        alias=None,
    )
    fields_set = set(data.fields_set())
    # Simulate validate_and_normalize_mcp_server_payload preserving alias=None.
    data.alias = None

    data_dict = await _run_update(data, fields_set=fields_set)

    assert "alias" in data_dict
    assert data_dict["alias"] is None


async def _run_update_with_existing(data: UpdateMCPServerRequest, existing_auth_type: str) -> dict:
    mock_prisma = _mock_prisma()
    existing = MagicMock()
    existing.auth_type = existing_auth_type
    existing.credentials = None
    mock_prisma.db.litellm_mcpservertable.find_unique = AsyncMock(return_value=existing)
    await update_mcp_server(mock_prisma, data, "test-user")
    return mock_prisma.db.litellm_mcpservertable.update.call_args[1]["data"]


@pytest.mark.asyncio
async def test_auth_type_switch_clears_stale_flow_scoped_fields():
    """
    Switching oauth2 -> oauth2_token_exchange must clear the previous flow's
    endpoint config: a stale token_url would otherwise be picked up as the
    token-exchange endpoint and suppress RFC 9728/8414 discovery.
    """
    data = UpdateMCPServerRequest(server_id="my-test-server", auth_type="oauth2_token_exchange")

    data_dict = await _run_update_with_existing(data, existing_auth_type="oauth2")

    for stale_field in (
        "issuer",
        "authorization_url",
        "token_url",
        "registration_url",
        "oauth2_flow",
        "dcr_bridge",
        "token_exchange_endpoint",
        "audience",
        "subject_token_type",
        "token_exchange_profile",
    ):
        assert data_dict[stale_field] is None, f"{stale_field} must be cleared on auth_type switch"
    assert _credentials_cleared(data_dict["credentials"])


@pytest.mark.asyncio
async def test_url_change_clears_stale_discovered_oauth_fields():
    """Re-pointing the server url at a potentially different upstream must clear the discovered or
    trust-on-first-use OAuth issuer and endpoints, so the new upstream re-discovers instead of
    anchoring on the previous upstream's issuer (RFC 8414 §3.3 against a stale anchor)."""
    mock_prisma = _mock_prisma()
    existing = MagicMock()
    existing.auth_type = "oauth2"
    existing.url = "https://old.example.com/mcp"
    existing.credentials = None
    mock_prisma.db.litellm_mcpservertable.find_unique = AsyncMock(return_value=existing)

    data = UpdateMCPServerRequest(server_id="my-test-server", url="https://new.example.com/mcp")
    await update_mcp_server(mock_prisma, data, "test-user")
    data_dict = mock_prisma.db.litellm_mcpservertable.update.call_args[1]["data"]

    assert data_dict["url"] == "https://new.example.com/mcp"
    for stale_field in ("issuer", "authorization_url", "token_url", "registration_url"):
        assert data_dict[stale_field] is None, f"{stale_field} must be cleared on url change"


@pytest.mark.asyncio
async def test_url_change_clears_stale_oauth_fields_even_when_resubmitted_unchanged():
    """The edit form re-sends every field, so a URL change arrives WITH the previous upstream's issuer
    and endpoints in the payload. Those resubmitted-unchanged values are stale and must still clear
    (otherwise they survive the url change and win in the resolution merge). A genuinely new value the
    caller changed in the same submit is kept."""
    mock_prisma = _mock_prisma()
    existing = MagicMock()
    existing.auth_type = "oauth2"
    existing.url = "https://old.example.com/mcp"
    existing.credentials = None
    existing.issuer = "https://old-idp.example.com"
    existing.token_url = "https://old-idp.example.com/token"
    existing.authorization_url = "https://old-idp.example.com/authorize"
    mock_prisma.db.litellm_mcpservertable.find_unique = AsyncMock(return_value=existing)

    data = UpdateMCPServerRequest(
        server_id="my-test-server",
        url="https://new.example.com/mcp",
        issuer="https://old-idp.example.com",  # resubmitted unchanged -> stale, must clear
        token_url="https://old-idp.example.com/token",  # resubmitted unchanged -> stale, must clear
        authorization_url="https://new-idp.example.com/authorize",  # genuinely changed -> kept
    )
    await update_mcp_server(mock_prisma, data, "test-user")
    data_dict = mock_prisma.db.litellm_mcpservertable.update.call_args[1]["data"]

    assert data_dict["issuer"] is None
    assert data_dict["token_url"] is None
    assert data_dict["authorization_url"] == "https://new-idp.example.com/authorize"


@pytest.mark.asyncio
async def test_clearing_pinned_issuer_clears_stale_oauth_endpoints():
    """Clearing a previously pinned issuer must not revive the endpoints resolved under it. Under an
    issuer anchor the endpoints come solely from the issuer document and are not persisted, but a row
    that was resource-rooted before the pin can still hold stale authorization_url/token_url; clearing
    the anchor without clearing those would let them win the resolution merge and be posted to without
    fresh discovery (RFC 8414 §3.3 provenance)."""
    mock_prisma = _mock_prisma()
    existing = MagicMock()
    existing.auth_type = "oauth2"
    existing.url = "https://same.example.com/mcp"
    existing.credentials = None
    existing.issuer = "https://pinned-idp.example.com"
    existing.token_url = "https://pinned-idp.example.com/token"
    existing.authorization_url = "https://pinned-idp.example.com/authorize"
    mock_prisma.db.litellm_mcpservertable.find_unique = AsyncMock(return_value=existing)

    data = UpdateMCPServerRequest(
        server_id="my-test-server",
        issuer="",  # admin clears the anchor; url and auth_type unchanged
        token_url="https://pinned-idp.example.com/token",
        authorization_url="https://pinned-idp.example.com/authorize",
    )
    await update_mcp_server(mock_prisma, data, "test-user")
    data_dict = mock_prisma.db.litellm_mcpservertable.update.call_args[1]["data"]

    assert data_dict["token_url"] is None
    assert data_dict["authorization_url"] is None


@pytest.mark.asyncio
async def test_repointing_pinned_issuer_clears_stale_endpoints_keeps_new_issuer():
    """Re-pointing the issuer to a different authorization server invalidates the old issuer's
    endpoints while keeping the new issuer the admin submitted."""
    mock_prisma = _mock_prisma()
    existing = MagicMock()
    existing.auth_type = "oauth2"
    existing.url = "https://same.example.com/mcp"
    existing.credentials = None
    existing.issuer = "https://old-idp.example.com"
    existing.token_url = "https://old-idp.example.com/token"
    existing.authorization_url = "https://old-idp.example.com/authorize"
    mock_prisma.db.litellm_mcpservertable.find_unique = AsyncMock(return_value=existing)

    data = UpdateMCPServerRequest(
        server_id="my-test-server",
        issuer="https://new-idp.example.com",
        token_url="https://old-idp.example.com/token",  # resubmitted stale -> must clear
        authorization_url="https://old-idp.example.com/authorize",  # resubmitted stale -> must clear
    )
    await update_mcp_server(mock_prisma, data, "test-user")
    data_dict = mock_prisma.db.litellm_mcpservertable.update.call_args[1]["data"]

    assert data_dict["issuer"] == "https://new-idp.example.com"
    assert data_dict["token_url"] is None
    assert data_dict["authorization_url"] is None


@pytest.mark.asyncio
async def test_establishing_issuer_first_time_preserves_discovered_fields():
    """Establishing an issuer for the first time (None -> X), which is exactly what the trust-on-first-use
    discovery write-back does, must NOT clear the endpoints or oauth2_flow it discovered in the same
    write. Only an issuer that was already pinned and is now changed or cleared invalidates its
    endpoints, so the discovery persist cannot wipe the fields it just resolved."""
    mock_prisma = _mock_prisma()
    existing = MagicMock()
    existing.auth_type = "oauth2"
    existing.url = "https://same.example.com/mcp"
    existing.credentials = None
    existing.issuer = None
    mock_prisma.db.litellm_mcpservertable.find_unique = AsyncMock(return_value=existing)

    data = UpdateMCPServerRequest(
        server_id="my-test-server",
        issuer="https://discovered-idp.example.com",
        authorization_url="https://discovered-idp.example.com/authorize",
        token_url="https://discovered-idp.example.com/token",
        oauth2_flow="authorization_code",
    )
    await update_mcp_server(mock_prisma, data, "mcp_oauth_discovery")
    data_dict = mock_prisma.db.litellm_mcpservertable.update.call_args[1]["data"]

    assert data_dict["issuer"] == "https://discovered-idp.example.com"
    assert data_dict["authorization_url"] == "https://discovered-idp.example.com/authorize"
    assert data_dict["token_url"] == "https://discovered-idp.example.com/token"
    assert data_dict.get("oauth2_flow") == "authorization_code"


@pytest.mark.asyncio
async def test_unchanged_url_does_not_clear_discovered_oauth_fields():
    """A partial update that resends the same url (or omits it) must not clear the discovered OAuth
    fields, so a routine save does not force needless re-discovery."""
    mock_prisma = _mock_prisma()
    existing = MagicMock()
    existing.auth_type = "oauth2"
    existing.url = "https://same.example.com/mcp"
    existing.credentials = None
    mock_prisma.db.litellm_mcpservertable.find_unique = AsyncMock(return_value=existing)

    data = UpdateMCPServerRequest(server_id="my-test-server", url="https://same.example.com/mcp")
    await update_mcp_server(mock_prisma, data, "test-user")
    data_dict = mock_prisma.db.litellm_mcpservertable.update.call_args[1]["data"]

    for preserved_field in ("issuer", "authorization_url", "token_url", "registration_url"):
        assert preserved_field not in data_dict, f"{preserved_field} must not be cleared when url is unchanged"


@pytest.mark.asyncio
async def test_auth_type_switch_keeps_explicitly_provided_flow_fields():
    """Fields explicitly provided alongside the auth_type switch must survive it."""
    data = UpdateMCPServerRequest(
        server_id="my-test-server",
        auth_type="oauth2_token_exchange",
        token_exchange_endpoint="https://idp.example.com/oauth2/token",
    )

    data_dict = await _run_update_with_existing(data, existing_auth_type="oauth2")

    assert data_dict["token_exchange_endpoint"] == "https://idp.example.com/oauth2/token"
    assert data_dict["token_url"] is None


@pytest.mark.asyncio
async def test_auth_type_switch_to_client_forwarded_keeps_explicit_dcr_bridge():
    data = UpdateMCPServerRequest(
        server_id="my-test-server",
        auth_type="true_passthrough",
        dcr_bridge=True,
    )

    data_dict = await _run_update_with_existing(data, existing_auth_type="oauth2")

    assert data_dict["dcr_bridge"] is True
    assert data_dict["oauth2_flow"] is None


@pytest.mark.asyncio
async def test_auth_type_switch_back_to_oauth2_clears_token_exchange_fields():
    """The reverse switch must not leave token-exchange settings behind to
    silently reactivate if the server is later switched back."""
    data = UpdateMCPServerRequest(server_id="my-test-server", auth_type="oauth2")

    data_dict = await _run_update_with_existing(data, existing_auth_type="oauth2_token_exchange")

    assert data_dict["token_exchange_endpoint"] is None
    assert data_dict["audience"] is None
    assert data_dict["subject_token_type"] is None
    assert data_dict["token_exchange_profile"] is None


@pytest.mark.asyncio
async def test_unchanged_auth_type_does_not_clear_flow_fields():
    """An update that keeps the auth_type must not touch flow-scoped fields, so a
    legacy OBO server using token_url as its exchange endpoint keeps working."""
    data = UpdateMCPServerRequest(
        server_id="my-test-server",
        auth_type="oauth2_token_exchange",
        allowed_tools=["foo"],
    )

    data_dict = await _run_update_with_existing(data, existing_auth_type="oauth2_token_exchange")

    for flow_field in (
        "authorization_url",
        "token_url",
        "registration_url",
        "oauth2_flow",
        "dcr_bridge",
        "token_exchange_endpoint",
        "audience",
        "subject_token_type",
        "token_exchange_profile",
    ):
        assert flow_field not in data_dict


@pytest.mark.asyncio
async def test_create_still_writes_defaults():
    """
    Regression guard: create (POST) must keep writing defaults so DB columns
    without a default get populated. exclude_unset is update-only.
    """
    mock_prisma = _mock_prisma()
    data = NewMCPServerRequest(
        server_id="new-server",
        url="https://example.com/mcp",
        transport="http",
    )

    await create_mcp_server(mock_prisma, data, "test-user")

    data_dict = mock_prisma.db.litellm_mcpservertable.create.call_args[1]["data"]

    assert data_dict["transport"] == "http"
    # is_byok is force-written on create.
    assert data_dict["is_byok"] is False
    # alias key is always present on create (even if None).
    assert "alias" in data_dict
    # audit fields set by create_mcp_server.
    assert data_dict["created_by"] == "test-user"
    assert data_dict["updated_by"] == "test-user"


# ── token-exchange blob → column normalization ────────────────────────────────
#
# token_exchange_endpoint / audience / subject_token_type have dedicated columns;
# their MCPCredentials copies are a legacy shape. Writes must lift blob values
# into the columns and strip them from the stored blob so the read-time
# ``column or blob`` fallback can never resurrect a stale blob value after the
# column is cleared.


@pytest.fixture(autouse=True)
def _salt_key(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-1234")


def _existing_row(auth_type: str, credentials: dict | None = None):
    existing = MagicMock()
    existing.auth_type = auth_type
    existing.credentials = json.dumps(credentials) if credentials is not None else None
    existing.token_exchange_endpoint = None
    existing.audience = None
    existing.subject_token_type = None
    existing.token_exchange_profile = None
    return existing


@pytest.mark.asyncio
async def test_create_lifts_blob_token_exchange_settings_into_columns():
    """The legacy REST shape (TE settings inside ``credentials``) must land in
    the dedicated columns, and the stored blob must not keep a copy."""
    mock_prisma = _mock_prisma()
    data = NewMCPServerRequest(
        server_id="te-server",
        url="https://example.com/mcp",
        transport="http",
        auth_type="oauth2_token_exchange",
        credentials={
            "client_id": "cid",
            "client_secret": "sec",
            "token_exchange_endpoint": "https://idp.example.com/oauth2/token",
            "audience": "api://upstream",
            "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
            "token_exchange_profile": "entra_obo",
        },
    )

    await create_mcp_server(mock_prisma, data, "test-user")
    data_dict = mock_prisma.db.litellm_mcpservertable.create.call_args[1]["data"]

    assert data_dict["token_exchange_endpoint"] == "https://idp.example.com/oauth2/token"
    assert data_dict["audience"] == "api://upstream"
    assert data_dict["subject_token_type"] == "urn:ietf:params:oauth:token-type:jwt"
    assert data_dict["token_exchange_profile"] == "entra_obo"
    stored_blob = json.loads(data_dict["credentials"])
    for te_field in ("token_exchange_endpoint", "audience", "subject_token_type", "token_exchange_profile"):
        assert te_field not in stored_blob
    assert "client_id" in stored_blob


@pytest.mark.asyncio
async def test_create_explicit_column_wins_over_blob_copy():
    mock_prisma = _mock_prisma()
    data = NewMCPServerRequest(
        server_id="te-server",
        url="https://example.com/mcp",
        transport="http",
        auth_type="oauth2_token_exchange",
        token_exchange_endpoint="https://top-level.example.com/token",
        credentials={"client_id": "cid", "token_exchange_endpoint": "https://blob.example.com/token"},
    )

    await create_mcp_server(mock_prisma, data, "test-user")
    data_dict = mock_prisma.db.litellm_mcpservertable.create.call_args[1]["data"]

    assert data_dict["token_exchange_endpoint"] == "https://top-level.example.com/token"
    assert "token_exchange_endpoint" not in json.loads(data_dict["credentials"])


@pytest.mark.asyncio
async def test_credentials_merge_migrates_legacy_blob_te_settings():
    """A same-auth credentials update on a legacy row (TE settings in the blob,
    columns null) must move the settings to the columns and drop them from the
    merged blob."""
    mock_prisma = _mock_prisma()
    existing = _existing_row(
        "oauth2_token_exchange",
        credentials={
            "client_id": "enc-old-cid",
            "token_exchange_endpoint": "https://legacy-idp.example.com/token",
            "audience": "api://legacy",
            "token_exchange_profile": "entra_obo",
        },
    )
    mock_prisma.db.litellm_mcpservertable.find_unique = AsyncMock(return_value=existing)

    data = UpdateMCPServerRequest(
        server_id="te-server",
        auth_type="oauth2_token_exchange",
        credentials={"client_id": "new-cid"},
    )
    await update_mcp_server(mock_prisma, data, "test-user")
    data_dict = mock_prisma.db.litellm_mcpservertable.update.call_args[1]["data"]

    assert data_dict["token_exchange_endpoint"] == "https://legacy-idp.example.com/token"
    assert data_dict["audience"] == "api://legacy"
    assert data_dict["token_exchange_profile"] == "entra_obo"
    merged_blob = json.loads(data_dict["credentials"])
    for te_field in ("token_exchange_endpoint", "audience", "subject_token_type", "token_exchange_profile"):
        assert te_field not in merged_blob


@pytest.mark.asyncio
async def test_cleared_column_is_not_resurrected_by_legacy_blob_value():
    """The Greptile scenario: explicitly clearing the column (to re-enable
    RFC 9728/8414 discovery) while the legacy blob still holds an endpoint must
    NOT resurrect the blob value — the explicit null wins and the blob copy is
    stripped."""
    mock_prisma = _mock_prisma()
    existing = _existing_row(
        "oauth2_token_exchange",
        credentials={"client_id": "enc-old-cid", "token_exchange_endpoint": "https://dead-idp.example.com/token"},
    )
    mock_prisma.db.litellm_mcpservertable.find_unique = AsyncMock(return_value=existing)

    data = UpdateMCPServerRequest(
        server_id="te-server",
        auth_type="oauth2_token_exchange",
        token_exchange_endpoint=None,
        credentials={"client_id": "new-cid"},
    )
    await update_mcp_server(mock_prisma, data, "test-user")
    data_dict = mock_prisma.db.litellm_mcpservertable.update.call_args[1]["data"]

    assert data_dict["token_exchange_endpoint"] is None
    assert "token_exchange_endpoint" not in json.loads(data_dict["credentials"])


@pytest.mark.asyncio
async def test_merge_strips_blob_te_copy_when_column_already_set():
    """When the row already has a column value, the blob copy is shadowed at
    read time anyway — the merge must strip it rather than carry it forward."""
    mock_prisma = _mock_prisma()
    existing = _existing_row(
        "oauth2_token_exchange",
        credentials={"client_id": "enc-old-cid", "token_exchange_endpoint": "https://blob-copy.example.com/token"},
    )
    existing.token_exchange_endpoint = "https://column.example.com/token"
    mock_prisma.db.litellm_mcpservertable.find_unique = AsyncMock(return_value=existing)

    data = UpdateMCPServerRequest(
        server_id="te-server",
        auth_type="oauth2_token_exchange",
        credentials={"client_id": "new-cid"},
    )
    await update_mcp_server(mock_prisma, data, "test-user")
    data_dict = mock_prisma.db.litellm_mcpservertable.update.call_args[1]["data"]

    # Column untouched by this update (not in payload), blob copy gone.
    assert "token_exchange_endpoint" not in data_dict
    assert "token_exchange_endpoint" not in json.loads(data_dict["credentials"])


@pytest.mark.asyncio
async def test_auth_type_switch_clears_flow_fields_with_external_fields_set():
    """The management endpoint passes ``fields_set`` explicitly (PUT
    /v1/mcp/server). The auth-switch clearing must fire on that path too — it is
    gated on ``data.auth_type``/the existing row, not on how fields_set arrives."""
    mock_prisma = _mock_prisma()
    existing = _existing_row("oauth2")
    mock_prisma.db.litellm_mcpservertable.find_unique = AsyncMock(return_value=existing)

    data = UpdateMCPServerRequest(server_id="te-server", auth_type="oauth2_token_exchange")
    await update_mcp_server(mock_prisma, data, "test-user", fields_set=set(data.fields_set()))
    data_dict = mock_prisma.db.litellm_mcpservertable.update.call_args[1]["data"]

    for stale_field in (
        "authorization_url",
        "token_url",
        "registration_url",
        "oauth2_flow",
        "token_exchange_endpoint",
        "audience",
        "subject_token_type",
        "token_exchange_profile",
    ):
        assert data_dict[stale_field] is None, f"{stale_field} must be cleared via the fields_set path"


@pytest.mark.asyncio
async def test_explicit_clear_without_credentials_purges_legacy_blob_copy():
    """Clearing a column in an update that does not touch credentials must strip
    the legacy blob copy too — otherwise the next credentials update's
    migrate-on-write would repopulate the column the admin just cleared."""
    mock_prisma = _mock_prisma()
    existing = _existing_row(
        "oauth2_token_exchange",
        credentials={"client_id": "enc-old-cid", "token_exchange_endpoint": "https://dead-idp.example.com/token"},
    )
    mock_prisma.db.litellm_mcpservertable.find_unique = AsyncMock(return_value=existing)

    data = UpdateMCPServerRequest(server_id="te-server", token_exchange_endpoint=None)
    await update_mcp_server(mock_prisma, data, "test-user")
    data_dict = mock_prisma.db.litellm_mcpservertable.update.call_args[1]["data"]

    assert data_dict["token_exchange_endpoint"] is None
    stored_blob = json.loads(data_dict["credentials"])
    assert "token_exchange_endpoint" not in stored_blob
    # Unrelated blob keys (encrypted secrets) survive untouched.
    assert stored_blob["client_id"] == "enc-old-cid"


@pytest.mark.asyncio
async def test_explicit_te_write_without_credentials_migrates_other_legacy_fields():
    """A no-credentials update that writes one token-exchange column migrates the
    whole row: untouched null columns are lifted from the blob, and every blob
    copy is stripped."""
    mock_prisma = _mock_prisma()
    existing = _existing_row(
        "oauth2_token_exchange",
        credentials={
            "client_id": "enc-old-cid",
            "token_exchange_endpoint": "https://legacy-idp.example.com/token",
            "audience": "api://legacy",
        },
    )
    mock_prisma.db.litellm_mcpservertable.find_unique = AsyncMock(return_value=existing)

    data = UpdateMCPServerRequest(server_id="te-server", audience="api://new")
    await update_mcp_server(mock_prisma, data, "test-user")
    data_dict = mock_prisma.db.litellm_mcpservertable.update.call_args[1]["data"]

    assert data_dict["audience"] == "api://new"
    assert data_dict["token_exchange_endpoint"] == "https://legacy-idp.example.com/token"
    stored_blob = json.loads(data_dict["credentials"])
    for te_field in ("token_exchange_endpoint", "audience", "subject_token_type"):
        assert te_field not in stored_blob


@pytest.mark.asyncio
async def test_te_update_without_blob_te_keys_leaves_credentials_untouched():
    """A no-credentials column write on a row whose blob has no legacy copies
    must not rewrite the credentials blob at all."""
    mock_prisma = _mock_prisma()
    existing = _existing_row("oauth2_token_exchange", credentials={"client_id": "enc-old-cid"})
    mock_prisma.db.litellm_mcpservertable.find_unique = AsyncMock(return_value=existing)

    data = UpdateMCPServerRequest(server_id="te-server", token_exchange_endpoint="https://new.example.com/token")
    await update_mcp_server(mock_prisma, data, "test-user")
    data_dict = mock_prisma.db.litellm_mcpservertable.update.call_args[1]["data"]

    assert data_dict["token_exchange_endpoint"] == "https://new.example.com/token"
    assert "credentials" not in data_dict


# ── client-forwarded credential class: true_passthrough <-> oauth_delegate share one
# stored-app shape, so a switch between them must MERGE (keep the declared app), not REPLACE ──


@pytest.mark.asyncio
async def test_cf_pair_switch_without_credentials_keeps_stored_app_and_endpoints():
    """true_passthrough -> oauth_delegate with no credentials in the update must not clear the
    stored client or null the endpoint columns: both modes use the same declared app and relay."""
    mock_prisma = _mock_prisma()
    existing = _existing_row("true_passthrough", credentials={"client_id": "enc-A", "client_secret": "enc-B"})
    existing.authorization_url = "https://provider.example/authorize"
    existing.token_url = "https://provider.example/token"
    existing.registration_url = "https://provider.example/register"
    mock_prisma.db.litellm_mcpservertable.find_unique = AsyncMock(return_value=existing)

    data = UpdateMCPServerRequest(server_id="cf-server", auth_type="oauth_delegate")
    await update_mcp_server(mock_prisma, data, "test-user")
    data_dict = mock_prisma.db.litellm_mcpservertable.update.call_args[1]["data"]

    assert "credentials" not in data_dict
    for scoped_field in ("authorization_url", "token_url", "registration_url", "oauth2_flow"):
        assert scoped_field not in data_dict, f"{scoped_field} must not be nulled within the CF class"


@pytest.mark.asyncio
async def test_cf_pair_switch_with_partial_credentials_merges_not_replaces():
    """oauth_delegate update carrying only client_id onto a true_passthrough row must MERGE, so the
    stored client_secret survives instead of being dropped by a REPLACE."""
    mock_prisma = _mock_prisma()
    existing = _existing_row("true_passthrough", credentials={"client_id": "enc-A", "client_secret": "enc-B"})
    mock_prisma.db.litellm_mcpservertable.find_unique = AsyncMock(return_value=existing)

    data = UpdateMCPServerRequest(server_id="cf-server", auth_type="oauth_delegate", credentials={"client_id": "B"})
    await update_mcp_server(mock_prisma, data, "test-user")
    data_dict = mock_prisma.db.litellm_mcpservertable.update.call_args[1]["data"]

    merged = json.loads(data_dict["credentials"])
    assert merged["client_secret"] == "enc-B"
    assert merged["client_id"] != "enc-A"


@pytest.mark.asyncio
async def test_null_existing_auth_type_to_cf_counts_as_changed_and_clears_blob():
    """A legacy row with NULL auth_type switched to a client-forwarded mode is a cross-class change,
    so the stale blob must be cleared (the two change predicates must agree on this)."""
    mock_prisma = _mock_prisma()
    existing = _existing_row(None, credentials={"client_id": "enc-old"})
    mock_prisma.db.litellm_mcpservertable.find_unique = AsyncMock(return_value=existing)

    data = UpdateMCPServerRequest(server_id="cf-server", auth_type="true_passthrough")
    await update_mcp_server(mock_prisma, data, "test-user")
    data_dict = mock_prisma.db.litellm_mcpservertable.update.call_args[1]["data"]

    # The clear must reach prisma as Json(None) (SQL null), never a raw None, which prisma rejects.
    assert isinstance(data_dict["credentials"], Json)
    assert getattr(data_dict["credentials"], "data", "x") is None


@pytest.mark.asyncio
async def test_client_rotation_strips_legacy_minted_token_keys():
    """Rotating the client on a same-class row must drop stale minted token material the update did
    not set, so an old access_token/refresh_token never rides forward under the new client."""
    mock_prisma = _mock_prisma()
    existing = _existing_row(
        "oauth2", credentials={"client_id": "A", "access_token": "T", "refresh_token": "R", "expires_in": 3600}
    )
    mock_prisma.db.litellm_mcpservertable.find_unique = AsyncMock(return_value=existing)

    data = UpdateMCPServerRequest(
        server_id="oauth2-server", auth_type="oauth2", credentials={"client_id": "B", "client_secret": "S"}
    )
    await update_mcp_server(mock_prisma, data, "test-user")
    data_dict = mock_prisma.db.litellm_mcpservertable.update.call_args[1]["data"]

    merged = json.loads(data_dict["credentials"])
    assert "access_token" not in merged
    assert "refresh_token" not in merged
    assert "expires_in" not in merged
    assert "client_secret" in merged


@pytest.mark.asyncio
async def test_cf_to_non_cf_switch_clears_dcr_bridge():
    """A cross-class switch OUT of a client-forwarded mode (true_passthrough -> api_key) must clear
    dcr_bridge: the switch is cross-class so the flow-scoped sweep runs and nulls it, leaving no stale
    dcr_bridge=True on a row that no longer supports it."""
    data = UpdateMCPServerRequest(server_id="s", auth_type="api_key")
    data_dict = await _run_update_with_existing(data, existing_auth_type="true_passthrough")
    assert data_dict["dcr_bridge"] is None


@pytest.mark.asyncio
async def test_cf_pair_switch_does_not_clear_dcr_bridge():
    """A within-class switch (true_passthrough <-> oauth_delegate) is not a credential-class change, so
    the flow-scoped sweep does not run and dcr_bridge is left intact (both modes use the DCR bridge)."""
    data = UpdateMCPServerRequest(server_id="s", auth_type="oauth_delegate")
    data_dict = await _run_update_with_existing(data, existing_auth_type="true_passthrough")
    assert "dcr_bridge" not in data_dict
