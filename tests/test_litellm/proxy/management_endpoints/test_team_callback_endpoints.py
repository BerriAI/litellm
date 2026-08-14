"""
Regression tests for team callback endpoint access control and audit logging.

The team callback endpoints mutate or expose callback credentials. They must
enforce target-team management access and, when audit logging is enabled, emit
redacted audit rows for callback mutations.
"""

import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi import HTTPException, Request

import litellm
from litellm.proxy._types import (
    AddTeamCallback,
    LitellmTableNames,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.management_endpoints.team_callback_endpoints import (
    add_team_callbacks,
    disable_team_logging,
    get_team_callbacks,
)


def _team_row(
    *,
    team_id: str = "team-victim",
    metadata: dict | None = None,
    admin_user_id: str = "victim_admin",
    organization_id: str = "org-victim",
) -> MagicMock:
    row = MagicMock()
    row.team_id = team_id
    row.metadata = metadata or {}
    row.model_dump.return_value = {
        "team_id": team_id,
        "team_alias": "victim-team",
        "members_with_roles": [
            {"role": "admin", "user_id": admin_user_id},
        ],
        "organization_id": organization_id,
        "metadata": row.metadata,
    }
    return row


def _patch_prisma(existing_team: MagicMock):
    mock_prisma = MagicMock()
    mock_prisma.get_data = AsyncMock(return_value=existing_team)

    updated_row = MagicMock()
    updated_row.team_id = existing_team.team_id
    mock_prisma.db.litellm_teamtable.update = AsyncMock(return_value=updated_row)
    return mock_prisma


def _admin_auth() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="hashed",
        user_id="admin-user",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )


@pytest.fixture(autouse=True)
def stub_team_cache_refresh():
    """Keep the cached-team refresh out of the way of the mocked prisma rows.

    The endpoints under test now refresh the auth cache after their DB write.
    That helper validates a real Prisma row into LiteLLM_TeamTableCachedObj,
    which the MagicMock rows these tests use cannot satisfy. The refresh being
    called at all is asserted explicitly in
    test_disable_team_logging_refreshes_cached_team.
    """
    with patch(
        "litellm.proxy.management_endpoints.team_callback_endpoints._refresh_cached_team",
        new_callable=AsyncMock,
    ) as refresh:
        yield refresh


@pytest.fixture
def unauthorized_caller():
    return UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="random_authenticated_user",
        api_key="sk-random",
    )


@pytest.fixture
def patched_prisma():
    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_client,
        patch(
            "litellm.proxy.management_endpoints.team_endpoints._is_user_org_admin_for_team",
            new_callable=AsyncMock,
            return_value=False,
        ),
    ):
        mock_client.get_data = AsyncMock(return_value=_team_row())
        mock_client.db.litellm_teamtable.update = AsyncMock()
        yield mock_client


@pytest.mark.asyncio
async def test_add_team_callbacks_rejects_unauthorized_caller(
    patched_prisma, unauthorized_caller
):
    data = AddTeamCallback(
        callback_name="langfuse",
        callback_type="success",
        callback_vars={
            "langfuse_public_key": "pk-attacker",
            "langfuse_secret_key": "sk-attacker",
        },
    )
    with pytest.raises(HTTPException) as exc:
        await add_team_callbacks(
            data=data,
            http_request=Mock(spec=Request),
            team_id="team-victim",
            user_api_key_dict=unauthorized_caller,
        )
    assert exc.value.status_code == 403
    patched_prisma.db.litellm_teamtable.update.assert_not_called()


@pytest.mark.asyncio
async def test_disable_team_logging_rejects_unauthorized_caller(
    patched_prisma, unauthorized_caller
):
    with pytest.raises(HTTPException) as exc:
        await disable_team_logging(
            http_request=Mock(spec=Request),
            team_id="team-victim",
            user_api_key_dict=unauthorized_caller,
        )
    assert exc.value.status_code == 403
    patched_prisma.db.litellm_teamtable.update.assert_not_called()


@pytest.mark.asyncio
async def test_get_team_callbacks_rejects_unauthorized_caller(
    patched_prisma, unauthorized_caller
):
    with pytest.raises(HTTPException) as exc:
        await get_team_callbacks(
            http_request=Mock(spec=Request),
            team_id="team-victim",
            user_api_key_dict=unauthorized_caller,
        )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_proxy_admin_can_add_team_callbacks(patched_prisma):
    data = AddTeamCallback(
        callback_name="langfuse",
        callback_type="success",
        callback_vars={
            "langfuse_public_key": "pk-admin",
            "langfuse_secret_key": "sk-admin",
        },
    )
    await add_team_callbacks(
        data=data,
        http_request=Mock(spec=Request),
        team_id="team-victim",
        user_api_key_dict=_admin_auth(),
    )
    patched_prisma.db.litellm_teamtable.update.assert_awaited_once()


@pytest.mark.asyncio
async def test_team_admin_of_target_team_can_add_callbacks(patched_prisma):
    patched_prisma.get_data = AsyncMock(
        return_value=_team_row(admin_user_id="team_admin_user")
    )

    team_admin = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="team_admin_user",
        api_key="sk-team-admin",
    )
    data = AddTeamCallback(
        callback_name="langfuse",
        callback_type="success",
        callback_vars={
            "langfuse_public_key": "pk-team",
            "langfuse_secret_key": "sk-team",
        },
    )
    await add_team_callbacks(
        data=data,
        http_request=Mock(spec=Request),
        team_id="team-victim",
        user_api_key_dict=team_admin,
    )
    patched_prisma.db.litellm_teamtable.update.assert_awaited_once()


@pytest.mark.asyncio
async def test_disable_team_logging_emits_audit_log_when_enabled(monkeypatch):
    monkeypatch.setattr(litellm, "store_audit_logs", True)
    mock_prisma = _patch_prisma(
        _team_row(
            team_id="team-1",
            metadata={
                "callback_settings": {
                    "success_callback": ["langfuse"],
                    "failure_callback": [],
                }
            },
        )
    )

    audit_calls = []

    async def capture(request_data):
        audit_calls.append(request_data)

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.management_helpers.audit_logs.create_audit_log_for_update",
            new=capture,
        ),
    ):
        await disable_team_logging(
            http_request=MagicMock(spec=Request),
            team_id="team-1",
            user_api_key_dict=_admin_auth(),
            litellm_changed_by=None,
        )
        import asyncio

        for _ in range(3):
            await asyncio.sleep(0)

    assert len(audit_calls) == 1
    log = audit_calls[0]
    assert log.table_name == LitellmTableNames.TEAM_TABLE_NAME
    assert log.object_id == "team-1"
    assert log.action == "updated"
    assert log.changed_by == "admin-user"

    before = json.loads(log.before_value)
    after = json.loads(log.updated_values)
    assert before["metadata"]["callback_settings"]["success_callback"] == ["langfuse"]
    assert after["metadata"]["callback_settings"]["success_callback"] == []
    assert after["metadata"]["callback_settings"]["failure_callback"] == []
    # The audit row has to show the slot the callbacks actually live in, so a
    # disable of a logging-configured team does not record an empty diff.
    assert after["metadata"]["logging"] == []


@pytest.mark.asyncio
async def test_disable_team_logging_no_audit_when_disabled(monkeypatch):
    monkeypatch.setattr(litellm, "store_audit_logs", False)
    mock_prisma = _patch_prisma(
        _team_row(
            team_id="team-1",
            metadata={
                "callback_settings": {
                    "success_callback": ["langfuse"],
                    "failure_callback": [],
                }
            },
        )
    )

    audit_calls = []

    async def capture(request_data):
        audit_calls.append(request_data)

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch(
            "litellm.proxy.management_helpers.audit_logs.create_audit_log_for_update",
            new=capture,
        ),
    ):
        await disable_team_logging(
            http_request=MagicMock(spec=Request),
            team_id="team-1",
            user_api_key_dict=_admin_auth(),
            litellm_changed_by=None,
        )

    assert audit_calls == []


@pytest.mark.asyncio
async def test_add_team_callbacks_emits_audit_log_when_enabled(monkeypatch):
    monkeypatch.setattr(litellm, "store_audit_logs", True)
    mock_prisma = _patch_prisma(_team_row(team_id="team-1", metadata={"logging": []}))

    audit_calls = []

    async def capture(request_data):
        audit_calls.append(request_data)

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.management_helpers.audit_logs.create_audit_log_for_update",
            new=capture,
        ),
    ):
        await add_team_callbacks(
            data=AddTeamCallback(
                callback_name="langfuse",
                callback_type="success",
                callback_vars={
                    "langfuse_public_key": "pk",
                    "langfuse_secret_key": "sk",
                },
            ),
            http_request=MagicMock(spec=Request),
            team_id="team-1",
            user_api_key_dict=_admin_auth(),
            litellm_changed_by="ops-on-call",
        )
        import asyncio

        for _ in range(3):
            await asyncio.sleep(0)

    assert len(audit_calls) == 1
    log = audit_calls[0]
    assert log.table_name == LitellmTableNames.TEAM_TABLE_NAME
    assert log.object_id == "team-1"
    assert log.action == "updated"
    assert log.changed_by == "ops-on-call"

    before = json.loads(log.before_value)
    after = json.loads(log.updated_values)
    assert before["metadata"]["logging"] == []
    assert len(after["metadata"]["logging"]) == 1
    assert after["metadata"]["logging"][0]["callback_name"] == "langfuse"

    callback_vars = after["metadata"]["logging"][0]["callback_vars"]
    assert callback_vars["langfuse_public_key"] != "pk"
    assert callback_vars["langfuse_secret_key"] != "sk"
    assert "langfuse_public_key" in callback_vars
    assert "langfuse_secret_key" in callback_vars
    assert "sk" not in log.updated_values.replace("sk-", "")
    assert "pk" not in (log.updated_values.replace("pk-", "").replace("public_key", ""))


@pytest.mark.asyncio
async def test_disable_team_logging_redacts_existing_callback_secrets(monkeypatch):
    monkeypatch.setattr(litellm, "store_audit_logs", True)
    mock_prisma = _patch_prisma(
        _team_row(
            team_id="team-1",
            metadata={
                "callback_settings": {
                    "success_callback": ["langfuse"],
                    "failure_callback": [],
                    "callback_vars": {
                        "langfuse_public_key": "pk-real",
                        "langfuse_secret_key": "sk-real-secret",
                    },
                }
            },
        )
    )

    audit_calls = []

    async def capture(request_data):
        audit_calls.append(request_data)

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.management_helpers.audit_logs.create_audit_log_for_update",
            new=capture,
        ),
    ):
        await disable_team_logging(
            http_request=MagicMock(spec=Request),
            team_id="team-1",
            user_api_key_dict=_admin_auth(),
            litellm_changed_by=None,
        )
        import asyncio

        for _ in range(3):
            await asyncio.sleep(0)

    assert len(audit_calls) == 1
    log = audit_calls[0]
    assert "sk-real-secret" not in log.before_value
    assert "sk-real-secret" not in log.updated_values
    assert "pk-real" not in log.before_value
    assert "pk-real" not in log.updated_values


@pytest.mark.asyncio
async def test_add_team_callbacks_no_audit_when_disabled(monkeypatch):
    monkeypatch.setattr(litellm, "store_audit_logs", False)
    mock_prisma = _patch_prisma(_team_row(team_id="team-1", metadata={"logging": []}))

    audit_calls = []

    async def capture(request_data):
        audit_calls.append(request_data)

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch(
            "litellm.proxy.management_helpers.audit_logs.create_audit_log_for_update",
            new=capture,
        ),
    ):
        await add_team_callbacks(
            data=AddTeamCallback(
                callback_name="langfuse",
                callback_type="success",
                callback_vars={
                    "langfuse_public_key": "pk",
                    "langfuse_secret_key": "sk",
                },
            ),
            http_request=MagicMock(spec=Request),
            team_id="team-1",
            user_api_key_dict=_admin_auth(),
            litellm_changed_by=None,
        )

    assert audit_calls == []


@pytest.mark.asyncio
async def test_add_team_callbacks_writes_encrypted_callback_vars(monkeypatch):
    """add_team_callbacks must encrypt callback_vars values before the DB write."""
    from litellm.proxy.common_utils.callback_utils import decrypt_callback_vars

    monkeypatch.setenv("LITELLM_SALT_KEY", "test-salt-32-bytes-aaaaaaaaaaaaaa")
    mock_prisma = _patch_prisma(_team_row(team_id="team-1", metadata={"logging": []}))

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch("litellm.proxy.proxy_server.master_key", None),
    ):
        await add_team_callbacks(
            data=AddTeamCallback(
                callback_name="langfuse",
                callback_type="success",
                callback_vars={
                    "langfuse_public_key": "pk-lf-real-public",
                    "langfuse_secret_key": "sk-lf-real-secret",
                },
            ),
            http_request=MagicMock(spec=Request),
            team_id="team-1",
            user_api_key_dict=_admin_auth(),
            litellm_changed_by=None,
        )

    written = json.loads(
        mock_prisma.db.litellm_teamtable.update.await_args.kwargs["data"]["metadata"]
    )
    cv = written["logging"][0]["callback_vars"]
    assert cv["langfuse_secret_key"] != "sk-lf-real-secret"
    assert cv["langfuse_public_key"] != "pk-lf-real-public"
    recovered = decrypt_callback_vars(written)["logging"][0]["callback_vars"]
    assert recovered["langfuse_secret_key"] == "sk-lf-real-secret"
    assert recovered["langfuse_public_key"] == "pk-lf-real-public"


@pytest.mark.asyncio
async def test_get_team_callbacks_returns_callbacks_registered_via_post(monkeypatch):
    """POST then GET must round-trip.

    add_team_callbacks writes metadata["logging"]; a GET that reads only
    metadata["callback_settings"] reports an empty list for every team
    configured through the API. The row handed to the GET here is the exact
    payload the POST wrote, so the two code paths cannot drift apart again.
    """
    monkeypatch.setenv("LITELLM_SALT_KEY", "test-salt-32-bytes-aaaaaaaaaaaaaa")
    row = _team_row(team_id="team-1", metadata={})
    mock_prisma = _patch_prisma(row)

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch("litellm.proxy.proxy_server.master_key", None),
    ):
        await add_team_callbacks(
            data=AddTeamCallback(
                callback_name="langsmith",
                callback_type="success",
                callback_vars={
                    "langsmith_api_key": "lsv2-real-secret",
                    "langsmith_project": "tenant-project",
                },
            ),
            http_request=MagicMock(spec=Request),
            team_id="team-1",
            user_api_key_dict=_admin_auth(),
            litellm_changed_by=None,
        )

        # Feed the GET exactly what the POST persisted.
        row.metadata = json.loads(mock_prisma.db.litellm_teamtable.update.await_args.kwargs["data"]["metadata"])
        row.model_dump.return_value["metadata"] = row.metadata

        response = await get_team_callbacks(
            http_request=MagicMock(spec=Request),
            team_id="team-1",
            user_api_key_dict=_admin_auth(),
        )

    assert response["data"]["success_callbacks"] == ["langsmith"]
    assert response["data"]["failure_callbacks"] == []
    # Non-secret vars come back usable, the credential is masked, and the
    # ciphertext that is stored on the row never reaches the response.
    assert response["data"]["callback_vars"]["langsmith_project"] == "tenant-project"
    assert response["data"]["callback_vars"]["langsmith_api_key"] == "***REDACTED***"
    assert "lsv2-real-secret" not in json.dumps(response)
    assert "litellm_enc::" not in json.dumps(response)


@pytest.mark.asyncio
async def test_get_team_callbacks_prefers_logging_over_deprecated_callback_settings():
    """A team carrying both shapes must report only the one that actually fires.

    Request-time resolution in _get_dynamic_logging_metadata stops at the
    first populated slot: metadata["logging"] wins and callback_settings is
    never consulted. Reporting the union here would tell an operator that
    gcs_bucket is active on a team whose requests never send to it.
    """
    metadata = {
        "callback_settings": {
            "success_callback": ["gcs_bucket"],
            "failure_callback": [],
            "callback_vars": {"gcs_bucket_name": "legacy-bucket"},
        },
        "logging": [
            {
                "callback_name": "langsmith",
                "callback_type": "success_and_failure",
                "callback_vars": {"langsmith_project": "tenant-project"},
            },
            {"callback_name": "missing-required-fields"},
        ],
    }
    mock_prisma = _patch_prisma(_team_row(team_id="team-1", metadata=metadata))

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
        response = await get_team_callbacks(
            http_request=MagicMock(spec=Request),
            team_id="team-1",
            user_api_key_dict=_admin_auth(),
        )

    assert response["data"]["success_callbacks"] == ["langsmith"]
    assert response["data"]["failure_callbacks"] == ["langsmith"]
    # The deprecated slot contributes nothing, and the malformed logging entry
    # is skipped rather than failing the whole read.
    assert response["data"]["callback_vars"] == {"langsmith_project": "tenant-project"}


@pytest.mark.asyncio
async def test_get_team_callbacks_reports_nothing_when_logging_slot_is_empty():
    """An empty logging slot must not fall through to callback_settings.

    Request-time resolution stops at the first non-None slot, so a team whose
    logging list is empty fires no callbacks at all even when the deprecated
    shape is still populated next to it. Reporting gcs_bucket here would tell
    an operator a destination is live when nothing is being sent to it.
    """
    metadata = {
        "logging": [],
        "callback_settings": {
            "success_callback": ["gcs_bucket"],
            "failure_callback": [],
            "callback_vars": {"gcs_bucket_name": "legacy-bucket"},
        },
    }
    mock_prisma = _patch_prisma(_team_row(team_id="team-1", metadata=metadata))

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
        response = await get_team_callbacks(
            http_request=MagicMock(spec=Request),
            team_id="team-1",
            user_api_key_dict=_admin_auth(),
        )

    assert response["data"]["success_callbacks"] == []
    assert response["data"]["failure_callbacks"] == []
    assert response["data"]["callback_vars"] == {}


@pytest.mark.asyncio
async def test_get_team_callbacks_decrypts_vars_stored_under_non_sensitive_keys(monkeypatch):
    """Stored ciphertext must be decrypted, not handed back raw.

    Which keys count as sensitive is a moving classification, so a value can be
    encrypted at rest under a key that later stops being masked on read. Without
    the decrypt step that value comes back as an unusable litellm_enc:: blob.
    """
    from litellm.proxy.common_utils.callback_utils import _CALLBACK_VAR_ENCRYPTED_PREFIX, is_sensitive_callback_key
    from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper

    monkeypatch.setenv("LITELLM_SALT_KEY", "test-salt-32-bytes-aaaaaaaaaaaaaa")
    assert not is_sensitive_callback_key("langsmith_project"), "test needs a key that is not masked on read"

    metadata = {
        "logging": [
            {
                "callback_name": "langsmith",
                "callback_type": "success",
                "callback_vars": {
                    "langsmith_project": _CALLBACK_VAR_ENCRYPTED_PREFIX + encrypt_value_helper("tenant-project"),
                },
            }
        ]
    }
    mock_prisma = _patch_prisma(_team_row(team_id="team-1", metadata=metadata))

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.master_key", None),
    ):
        response = await get_team_callbacks(
            http_request=MagicMock(spec=Request),
            team_id="team-1",
            user_api_key_dict=_admin_auth(),
        )

    assert response["data"]["callback_vars"]["langsmith_project"] == "tenant-project"


@pytest.mark.asyncio
async def test_get_team_callbacks_masks_values_that_fail_to_decrypt(monkeypatch):
    """A value that cannot be decrypted must never leave as ciphertext.

    After a salt-key rotation an existing value no longer decrypts, and the
    shared helper passes it through untouched. Under a key that is not
    classified as sensitive it would otherwise reach the caller as an opaque
    blob that is indistinguishable from a real value.
    """
    from litellm.proxy.common_utils.callback_utils import _CALLBACK_VAR_ENCRYPTED_PREFIX
    from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper

    monkeypatch.setenv("LITELLM_SALT_KEY", "test-salt-32-bytes-aaaaaaaaaaaaaa")
    stale = _CALLBACK_VAR_ENCRYPTED_PREFIX + encrypt_value_helper("tenant-project")
    monkeypatch.setenv("LITELLM_SALT_KEY", "test-salt-32-bytes-bbbbbbbbbbbbbb")

    metadata = {
        "logging": [
            {
                "callback_name": "langsmith",
                "callback_type": "success",
                "callback_vars": {"langsmith_project": stale},
            }
        ]
    }
    mock_prisma = _patch_prisma(_team_row(team_id="team-1", metadata=metadata))

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.master_key", None),
    ):
        response = await get_team_callbacks(
            http_request=MagicMock(spec=Request),
            team_id="team-1",
            user_api_key_dict=_admin_auth(),
        )

    assert response["data"]["success_callbacks"] == ["langsmith"]
    assert response["data"]["callback_vars"]["langsmith_project"] == "***REDACTED***"
    assert _CALLBACK_VAR_ENCRYPTED_PREFIX not in json.dumps(response)


@pytest.mark.asyncio
async def test_get_team_callbacks_falls_back_to_deprecated_callback_settings():
    """Teams that never used the API keep working: with no logging slot, the
    deprecated callback_settings shape is still reported, exactly as the
    request-time fallback would use it."""
    metadata = {
        "callback_settings": {
            "success_callback": ["gcs_bucket"],
            "failure_callback": ["langfuse"],
            "callback_vars": {
                "gcs_bucket_name": "legacy-bucket",
                "langfuse_secret_key": "sk-lf-legacy-plaintext",
            },
        }
    }
    mock_prisma = _patch_prisma(_team_row(team_id="team-1", metadata=metadata))

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
        response = await get_team_callbacks(
            http_request=MagicMock(spec=Request),
            team_id="team-1",
            user_api_key_dict=_admin_auth(),
        )

    assert response["data"]["success_callbacks"] == ["gcs_bucket"]
    assert response["data"]["failure_callbacks"] == ["langfuse"]
    assert response["data"]["callback_vars"]["gcs_bucket_name"] == "legacy-bucket"
    # Legacy rows predate encryption at rest, so this endpoint is where the
    # plaintext secret would otherwise escape.
    assert response["data"]["callback_vars"]["langfuse_secret_key"] == "***REDACTED***"
    assert "sk-lf-legacy-plaintext" not in json.dumps(response)


@pytest.mark.asyncio
async def test_get_team_callbacks_reports_empty_for_team_without_callbacks():
    """A team with no callback config still returns the documented empty shape."""
    mock_prisma = _patch_prisma(_team_row(team_id="team-1", metadata={}))

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
        response = await get_team_callbacks(
            http_request=MagicMock(spec=Request),
            team_id="team-1",
            user_api_key_dict=_admin_auth(),
        )

    assert response["data"] == {
        "team_id": "team-1",
        "success_callbacks": [],
        "failure_callbacks": [],
        "callback_vars": {},
    }


@pytest.mark.asyncio
async def test_disable_team_logging_stops_callbacks_registered_via_api():
    """Disabling logging must stop the callbacks that are actually running.

    Callbacks registered through the API or the Admin UI live in
    metadata["logging"], and request-time resolution stops at that slot without
    reading callback_settings. Clearing only callback_settings therefore reports
    success while the team keeps sending to its logging destination. This drives
    the endpoint and then asks the real request-time resolver what the written
    row would do.
    """
    from litellm.proxy.litellm_pre_call_utils import _get_dynamic_logging_metadata

    metadata = {
        "logging": [
            {
                "callback_name": "langsmith",
                "callback_type": "success",
                "callback_vars": {"langsmith_project": "tenant-project"},
            }
        ]
    }
    mock_prisma = _patch_prisma(_team_row(team_id="team-1", metadata=metadata))

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.master_key", None),
    ):
        response = await disable_team_logging(
            http_request=MagicMock(spec=Request),
            team_id="team-1",
            user_api_key_dict=_admin_auth(),
            litellm_changed_by=None,
        )

    assert response["status"] == "success"
    written = json.loads(mock_prisma.db.litellm_teamtable.update.await_args.kwargs["data"]["metadata"])
    assert written["logging"] == []

    resolved = _get_dynamic_logging_metadata(
        UserAPIKeyAuth(api_key="hashed", team_id="team-1", team_metadata=written),
        proxy_config=MagicMock(**{"load_team_config.return_value": {}}),
    )
    assert not (resolved.success_callback if resolved else None)
    assert not (resolved.failure_callback if resolved else None)


@pytest.mark.asyncio
async def test_disable_team_logging_refreshes_cached_team(stub_team_cache_refresh):
    """The DB write alone does not stop delivery.

    Auth serves a cached team object and request-time callback resolution reads
    the metadata off it, so without this refresh a key that is already in flight
    keeps sending to the destination until the cache entry expires.
    """
    metadata = {
        "logging": [
            {
                "callback_name": "langsmith",
                "callback_type": "success",
                "callback_vars": {"langsmith_project": "tenant-project"},
            }
        ]
    }
    mock_prisma = _patch_prisma(_team_row(team_id="team-1", metadata=metadata))

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.master_key", None),
    ):
        await disable_team_logging(
            http_request=MagicMock(spec=Request),
            team_id="team-1",
            user_api_key_dict=_admin_auth(),
            litellm_changed_by=None,
        )

    stub_team_cache_refresh.assert_awaited_once()
    refreshed = stub_team_cache_refresh.await_args.kwargs["team_row"]
    assert refreshed is mock_prisma.db.litellm_teamtable.update.return_value
    # The row fed to the cache has to carry object_permission, or the refresh
    # publishes a team whose tool allowlists look empty, which reads as
    # unrestricted on the search-tool and MCP-tool checks.
    update_kwargs = mock_prisma.db.litellm_teamtable.update.await_args.kwargs
    assert update_kwargs["include"]["object_permission"] is True


@pytest.mark.asyncio
async def test_add_team_callbacks_refreshes_cached_team(stub_team_cache_refresh):
    """Registering a callback must take effect for keys that are already live."""
    mock_prisma = _patch_prisma(_team_row(team_id="team-1", metadata={"logging": []}))

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.master_key", None),
    ):
        await add_team_callbacks(
            data=AddTeamCallback(
                callback_name="langsmith",
                callback_type="success",
                callback_vars={"langsmith_project": "tenant-project"},
            ),
            http_request=MagicMock(spec=Request),
            team_id="team-1",
            user_api_key_dict=_admin_auth(),
            litellm_changed_by=None,
        )

    stub_team_cache_refresh.assert_awaited_once()
    refreshed = stub_team_cache_refresh.await_args.kwargs["team_row"]
    assert refreshed is mock_prisma.db.litellm_teamtable.update.return_value
    update_kwargs = mock_prisma.db.litellm_teamtable.update.await_args.kwargs
    assert update_kwargs["include"]["object_permission"] is True


@pytest.mark.asyncio
async def test_disable_team_logging_clears_both_metadata_shapes():
    """A team carrying both shapes ends up with neither active."""
    from litellm.proxy.litellm_pre_call_utils import _get_dynamic_logging_metadata

    metadata = {
        "logging": [
            {
                "callback_name": "langsmith",
                "callback_type": "success_and_failure",
                "callback_vars": {"langsmith_project": "tenant-project"},
            }
        ],
        "callback_settings": {
            "success_callback": ["gcs_bucket"],
            "failure_callback": ["langfuse"],
            "callback_vars": {"gcs_bucket_name": "legacy-bucket"},
        },
    }
    mock_prisma = _patch_prisma(_team_row(team_id="team-1", metadata=metadata))

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.master_key", None),
    ):
        await disable_team_logging(
            http_request=MagicMock(spec=Request),
            team_id="team-1",
            user_api_key_dict=_admin_auth(),
            litellm_changed_by=None,
        )

    written = json.loads(mock_prisma.db.litellm_teamtable.update.await_args.kwargs["data"]["metadata"])
    assert written["logging"] == []
    assert written["callback_settings"]["success_callback"] == []
    assert written["callback_settings"]["failure_callback"] == []

    resolved = _get_dynamic_logging_metadata(
        UserAPIKeyAuth(api_key="hashed", team_id="team-1", team_metadata=written),
        proxy_config=MagicMock(**{"load_team_config.return_value": {}}),
    )
    assert not (resolved.success_callback if resolved else None)
    assert not (resolved.failure_callback if resolved else None)


@pytest.mark.asyncio
async def test_disable_team_logging_leaves_team_re_enablable():
    """The emptied slot must still accept a fresh registration afterwards."""
    metadata = {
        "logging": [
            {
                "callback_name": "langsmith",
                "callback_type": "success",
                "callback_vars": {"langsmith_project": "tenant-project"},
            }
        ]
    }
    row = _team_row(team_id="team-1", metadata=metadata)
    mock_prisma = _patch_prisma(row)

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.master_key", None),
    ):
        await disable_team_logging(
            http_request=MagicMock(spec=Request),
            team_id="team-1",
            user_api_key_dict=_admin_auth(),
            litellm_changed_by=None,
        )
        row.metadata = json.loads(mock_prisma.db.litellm_teamtable.update.await_args.kwargs["data"]["metadata"])
        row.model_dump.return_value["metadata"] = row.metadata

        await add_team_callbacks(
            data=AddTeamCallback(
                callback_name="langfuse",
                callback_type="success",
                callback_vars={"langfuse_public_key": "pk-lf-new"},
            ),
            http_request=MagicMock(spec=Request),
            team_id="team-1",
            user_api_key_dict=_admin_auth(),
            litellm_changed_by=None,
        )

    written = json.loads(mock_prisma.db.litellm_teamtable.update.await_args.kwargs["data"]["metadata"])
    assert [entry["callback_name"] for entry in written["logging"]] == ["langfuse"]
