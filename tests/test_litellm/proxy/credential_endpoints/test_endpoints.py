"""Admin-gating on credential mutations.

POST and DELETE on ``/credentials`` are proxy-admin only across the board
(both logging and provider credentials). The route gate was widened so
team-admins can PATCH ``access.teams`` on existing logging destinations,
which requires also reaching the GET endpoint — but creation and deletion
of any credential remain admin-only to keep platform infrastructure under
the platform admin's control. PATCH is widened for logging credentials only
via the pure ``decide_credential_patch`` decider tested separately.
"""

import os
import sys

import pytest
from fastapi import HTTPException
from prisma.errors import ClientNotConnectedError
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
import litellm.proxy.credential_endpoints.endpoints as endpoints
from litellm.models.credentials import CredentialItem, UpdateCredentialItem
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.credential_endpoints.access_decision import OPAQUE_DENY_REASON
from litellm.types.utils import CreateCredentialItem


def _admin():
    return UserAPIKeyAuth(api_key="k", user_role=LitellmUserRoles.PROXY_ADMIN)


def _member():
    return UserAPIKeyAuth(api_key="k", user_role=LitellmUserRoles.INTERNAL_USER)


_LOGGING_INFO = {"credential_type": "logging", "description": "langfuse_otel"}


@pytest.fixture
def _connected_db(monkeypatch):
    """A working prisma_client + repository so an allowed caller reaches success."""
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-test-salt-key")
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    monkeypatch.setattr(proxy_server, "llm_router", None)
    repo = MagicMock()
    repo.create = AsyncMock()
    repo.delete_by_name = AsyncMock()
    repo.update_by_name = AsyncMock()

    async def _find_by_name(name: str) -> CredentialItem | None:
        return next(
            (credential for credential in litellm.credential_list if credential.credential_name == name),
            None,
        )

    repo.find_by_name = AsyncMock(side_effect=_find_by_name)
    monkeypatch.setattr(endpoints, "CredentialsRepository", lambda _client: repo)
    monkeypatch.setattr(endpoints.CredentialAccessor, "upsert_credentials", lambda creds: None)
    return repo


@pytest.mark.asyncio
async def test_create_logging_credential_forbidden_for_non_admin(_connected_db):
    with pytest.raises(HTTPException) as exc:
        await endpoints.create_credential(
            request=MagicMock(),
            fastapi_response=MagicMock(),
            credential=CreateCredentialItem(
                credential_name="dest",
                credential_values={"langfuse_host": "h"},
                credential_info=_LOGGING_INFO,
            ),
            user_api_key_dict=_member(),
        )
    assert exc.value.status_code == 403
    _connected_db.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_logging_credential_allowed_for_admin(_connected_db):
    result = await endpoints.create_credential(
        request=MagicMock(),
        fastapi_response=MagicMock(),
        credential=CreateCredentialItem(
            credential_name="dest",
            credential_values={"langfuse_host": "h"},
            credential_info=_LOGGING_INFO,
        ),
        user_api_key_dict=_admin(),
    )
    assert result["success"] is True
    _connected_db.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_provider_credential_forbidden_for_non_admin(_connected_db):
    """POST is proxy-admin only even for provider credentials.

    The route gate now lets non-admins reach the credentials path so they can
    PATCH ``access.teams`` on logging destinations they should be able to
    self-assign. POST remains admin-only to keep credential creation a
    platform-admin concern.
    """
    with pytest.raises(HTTPException) as exc:
        await endpoints.create_credential(
            request=MagicMock(),
            fastapi_response=MagicMock(),
            credential=CreateCredentialItem(
                credential_name="openai",
                credential_values={"api_key": "sk"},
                credential_info={"custom_llm_provider": "openai"},
            ),
            user_api_key_dict=_member(),
        )
    assert exc.value.status_code == 403
    _connected_db.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_logging_credential_forbidden_for_non_admin(_connected_db):
    with pytest.raises(HTTPException) as exc:
        await endpoints.update_credential(
            request=MagicMock(),
            fastapi_response=MagicMock(),
            credential=CredentialItem(
                credential_name="dest",
                credential_values={},
                credential_info={"access": {"global": True}},
            ),
            credential_name="dest",
            user_api_key_dict=_member(),
        )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_update_existing_logging_credential_forbidden_even_without_logging_patch(_connected_db, monkeypatch):
    """A non-admin cannot edit a stored logging credential's values, even with a patch
    that omits credential_info (the gate consults the in-memory credential too)."""
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="dest",
                credential_values={"langfuse_host": "h"},
                credential_info=_LOGGING_INFO,
            )
        ],
    )
    with pytest.raises(HTTPException) as exc:
        await endpoints.update_credential(
            request=MagicMock(),
            fastapi_response=MagicMock(),
            credential=CredentialItem(
                credential_name="dest",
                credential_values={"langfuse_host": "evil"},
                credential_info={},
            ),
            credential_name="dest",
            user_api_key_dict=_member(),
        )
    assert exc.value.status_code == 403


def test_update_db_credential_preserves_existing_info_on_partial_patch():
    """A partial credential_info patch (e.g. only access from the Edit-access modal) must
    merge into the stored info, not replace it -- otherwise the logging tag is dropped and
    the destination vanishes from the registry after the next reload."""
    from litellm.proxy.credential_endpoints.endpoints import update_db_credential

    db = CredentialItem(
        credential_name="dest",
        credential_values={},
        credential_info={
            "credential_type": "logging",
            "description": "langfuse_otel",
            "host": "h",
        },
    )
    patch = CredentialItem(
        credential_name="dest",
        credential_values={},
        credential_info={"access": {"global": True}},
    )

    merged = update_db_credential(db, patch)

    assert merged.credential_info == {
        "credential_type": "logging",
        "description": "langfuse_otel",
        "host": "h",
        "access": {"global": True},
    }


def test_update_db_credential_preserves_untouched_access_subfields():
    """Veria regression: an access patch carrying only `teams` must NOT clobber
    existing `global` / `orgs`. Pre-fix this caused a scope-tampering bug —
    the decider only allowed team-admin patches that touched access.teams,
    but the merge replaced the entire access object, silently dropping
    access.global=true and any access.orgs entries.
    """
    from litellm.proxy.credential_endpoints.endpoints import update_db_credential

    db = CredentialItem(
        credential_name="dest",
        credential_values={},
        credential_info={
            "credential_type": "logging",
            "description": "langfuse_otel",
            "host": "h",
            "access": {
                "global": True,
                "orgs": ["org-1", "org-2"],
                "teams": ["team-A"],
            },
        },
    )
    # Team-admin's allowed shape: add their team to access.teams. Crucially,
    # they don't (and per the decider can't) include global/orgs.
    patch = CredentialItem(
        credential_name="dest",
        credential_values={},
        credential_info={"access": {"teams": ["team-A", "team-T"]}},
    )

    merged = update_db_credential(db, patch)

    # access.global and access.orgs survive untouched; access.teams is updated.
    assert merged.credential_info["access"] == {
        "global": True,
        "orgs": ["org-1", "org-2"],
        "teams": ["team-A", "team-T"],
    }


@pytest.mark.asyncio
async def test_delete_logging_credential_forbidden_for_non_admin(_connected_db, monkeypatch):
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="dest",
                credential_values={},
                credential_info=_LOGGING_INFO,
            )
        ],
    )
    with pytest.raises(HTTPException) as exc:
        await endpoints.delete_credential(
            request=MagicMock(),
            fastapi_response=MagicMock(),
            credential_name="dest",
            user_api_key_dict=_member(),
        )
    assert exc.value.status_code == 403
    _connected_db.delete_by_name.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_db_only_logging_credential_forbidden_for_non_admin(_connected_db, monkeypatch):
    """A logging credential that exists ONLY in the DB (not resident in the
    in-memory ``credential_list`` -- e.g. created on another scaled instance or
    before a restart) must still gate a non-admin update. The gate falls back to
    the DB so a credential_values-only patch can't redirect a logging
    destination's endpoint without the proxy-admin check."""
    monkeypatch.setattr(litellm, "credential_list", [])  # nothing in memory
    _connected_db.find_by_name = AsyncMock(
        return_value=CredentialItem(
            credential_name="dest",
            credential_values={"langfuse_host": "h"},
            credential_info=_LOGGING_INFO,
        )
    )
    with pytest.raises(HTTPException) as exc:
        await endpoints.update_credential(
            request=MagicMock(),
            fastapi_response=MagicMock(),
            credential=CredentialItem(
                credential_name="dest",
                credential_values={"langfuse_host": "evil"},
                credential_info={},
            ),
            credential_name="dest",
            user_api_key_dict=_member(),
        )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_delete_db_only_logging_credential_forbidden_for_non_admin(_connected_db, monkeypatch):
    """Same DB-only fallback for delete: a non-admin can't delete a logging
    credential that is resident only in the DB."""
    monkeypatch.setattr(litellm, "credential_list", [])
    _connected_db.find_by_name = AsyncMock(
        return_value=CredentialItem(
            credential_name="dest",
            credential_values={},
            credential_info=_LOGGING_INFO,
        )
    )
    with pytest.raises(HTTPException) as exc:
        await endpoints.delete_credential(
            request=MagicMock(),
            fastapi_response=MagicMock(),
            credential_name="dest",
            user_api_key_dict=_member(),
        )
    assert exc.value.status_code == 403
    _connected_db.delete_by_name.assert_not_awaited()


# --- team-admin self-assign tests (LIT-3850 follow-up) ----------------------

_DEST_WITH_TEAMS = {
    "credential_type": "logging",
    "description": "tenant Langfuse",
    "host": "https://cloud.langfuse.com",
    "access": {"teams": ["team-existing"]},
}


def _team_admin_of(team_ids):
    """A non-admin caller whose user_id is admin of the named teams.

    Combined with ``_patch_team_admin_lookup`` it mimics the real
    ``_caller_grantable_team_ids`` resolution without touching the DB.
    """
    return UserAPIKeyAuth(api_key="k", user_role=LitellmUserRoles.INTERNAL_USER, user_id="ta-demo")


@pytest.fixture
def _patch_team_admin_lookup(monkeypatch):
    """Substitute the DB-backed admin-scope lookup with a configurable mock.

    ``ids`` is the caller's team-admin scope; ``org_ids`` the org-admin scope.
    Patching the single source ``_caller_admin_scope`` covers both the list
    endpoint and the PATCH decider, which reads team ids through the
    ``_caller_grantable_team_ids`` wrapper.
    """

    holder = {"ids": frozenset(), "org_ids": frozenset()}

    async def _fake(user_api_key_dict, prisma_client):
        return endpoints.CallerAdminScope(
            team_ids=frozenset(holder["ids"]),
            org_ids=frozenset(holder["org_ids"]),
        )

    monkeypatch.setattr(endpoints, "_caller_admin_scope", _fake)
    return holder


def _resident_logging_dest():
    return CredentialItem(
        credential_name="dest",
        credential_values={"langfuse_host": "h"},
        credential_info=_DEST_WITH_TEAMS,
    )


@pytest.mark.asyncio
async def test_team_admin_can_append_own_team_to_access(_connected_db, _patch_team_admin_lookup, monkeypatch):
    monkeypatch.setattr(litellm, "credential_list", [_resident_logging_dest()])
    _connected_db.find_by_name = AsyncMock(return_value=_resident_logging_dest())
    _connected_db.update_by_name = AsyncMock()
    _patch_team_admin_lookup["ids"] = frozenset({"team-T"})

    result = await endpoints.update_credential(
        request=MagicMock(),
        fastapi_response=MagicMock(),
        credential=CredentialItem(
            credential_name="dest",
            credential_values={},
            credential_info={"access": {"teams": ["team-existing", "team-T"]}},
        ),
        credential_name="dest",
        user_api_key_dict=_team_admin_of(["team-T"]),
    )
    assert result["success"] is True
    _connected_db.find_by_name.assert_awaited_once_with("dest")
    _connected_db.update_by_name.assert_awaited_once()


@pytest.mark.asyncio
async def test_team_admin_cannot_replay_stale_global_scope(_connected_db, _patch_team_admin_lookup, monkeypatch):
    cached = CredentialItem(
        credential_name="dest",
        credential_values={"langfuse_host": "h"},
        credential_info={
            **_DEST_WITH_TEAMS,
            "access": {"global": False, "teams": ["team-existing"]},
        },
    )
    authoritative = CredentialItem(
        credential_name="dest",
        credential_values={"langfuse_host": "h"},
        credential_info={
            **_DEST_WITH_TEAMS,
            "access": {"global": True, "teams": ["team-existing"]},
        },
    )
    monkeypatch.setattr(litellm, "credential_list", [cached])
    _connected_db.find_by_name = AsyncMock(return_value=authoritative)
    _connected_db.update_by_name = AsyncMock()
    _patch_team_admin_lookup["ids"] = frozenset({"team-T"})

    with pytest.raises(HTTPException) as exc:
        await endpoints.update_credential(
            request=MagicMock(),
            fastapi_response=MagicMock(),
            credential=UpdateCredentialItem(
                credential_info={
                    "access": {
                        "global": False,
                        "teams": ["team-existing", "team-T"],
                    }
                },
            ),
            credential_name="dest",
            user_api_key_dict=_team_admin_of(["team-T"]),
        )

    assert exc.value.status_code == 403
    _connected_db.find_by_name.assert_awaited_once_with("dest")
    _connected_db.update_by_name.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_credential_handles_database_lookup_failure(_connected_db):
    db_error = ClientNotConnectedError()
    _connected_db.find_by_name = AsyncMock(side_effect=db_error)
    _connected_db.update_by_name = AsyncMock()

    result = await endpoints.update_credential(
        request=MagicMock(),
        fastapi_response=MagicMock(),
        credential=UpdateCredentialItem(
            credential_info={"access": {"global": True}},
        ),
        credential_name="dest",
        user_api_key_dict=_admin(),
    )

    assert result.code == "500"
    assert result.message == str(db_error)
    _connected_db.find_by_name.assert_awaited_once_with("dest")
    _connected_db.update_by_name.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_credential_handles_missing_database_client(monkeypatch):
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "prisma_client", None)

    result = await endpoints.update_credential(
        request=MagicMock(),
        fastapi_response=MagicMock(),
        credential=UpdateCredentialItem(
            credential_info={"access": {"global": True}},
        ),
        credential_name="dest",
        user_api_key_dict=_admin(),
    )

    assert result.code == "500"


@pytest.mark.asyncio
async def test_update_credential_handles_missing_database_credential(_connected_db):
    _connected_db.find_by_name = AsyncMock(return_value=None)
    _connected_db.update_by_name = AsyncMock()

    result = await endpoints.update_credential(
        request=MagicMock(),
        fastapi_response=MagicMock(),
        credential=UpdateCredentialItem(
            credential_info={"access": {"global": True}},
        ),
        credential_name="dest",
        user_api_key_dict=_admin(),
    )

    assert result.code == "404"
    _connected_db.find_by_name.assert_awaited_once_with("dest")
    _connected_db.update_by_name.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_credential_handles_database_write_failure(_connected_db):
    db_error = ClientNotConnectedError()
    existing = CredentialItem(
        credential_name="dest",
        credential_values={"langfuse_host": "h"},
        credential_info=_LOGGING_INFO,
    )
    _connected_db.find_by_name = AsyncMock(return_value=existing)
    _connected_db.update_by_name = AsyncMock(side_effect=db_error)

    result = await endpoints.update_credential(
        request=MagicMock(),
        fastapi_response=MagicMock(),
        credential=UpdateCredentialItem(
            credential_info={"access": {"global": True}},
        ),
        credential_name="dest",
        user_api_key_dict=_admin(),
    )

    assert result.code == "500"
    assert result.message == str(db_error)
    _connected_db.find_by_name.assert_awaited_once_with("dest")
    _connected_db.update_by_name.assert_awaited_once()


@pytest.mark.asyncio
async def test_provider_credential_patch_forbidden_for_non_admin(_connected_db, monkeypatch):
    """A team-admin (or any non-admin) cannot PATCH a non-logging credential.

    The route gate was widened to let team-admins reach /credentials/{name}
    for logging-credential access edits. A provider credential is not
    is_admin_gated_credential_info, so the decider block is skipped; without
    an explicit else-branch a team-admin could rotate the upstream api_key.
    """
    provider_cred = CredentialItem(
        credential_name="openai-prod",
        credential_values={"api_key": "sk-real"},
        credential_info={"custom_llm_provider": "openai"},
    )
    monkeypatch.setattr(litellm, "credential_list", [provider_cred])
    _connected_db.find_by_name = AsyncMock(return_value=provider_cred)
    _connected_db.update_by_name = AsyncMock()

    with pytest.raises(HTTPException) as exc:
        await endpoints.update_credential(
            request=MagicMock(),
            fastapi_response=MagicMock(),
            credential=CredentialItem(
                credential_name="openai-prod",
                credential_values={"api_key": "sk-stolen"},
                credential_info={},
            ),
            credential_name="openai-prod",
            user_api_key_dict=_member(),
        )
    assert exc.value.status_code == 403
    _connected_db.update_by_name.assert_not_awaited()


@pytest.mark.asyncio
async def test_provider_credential_access_patch_bypass_forbidden(_connected_db, _patch_team_admin_lookup, monkeypatch):
    """Cursor BugBot regression: a team-admin can't sneak `access.teams` onto
    a PROVIDER credential to route through the decider instead of the admin
    gate.

    `is_admin_gated_credential_info(patch)` returns True for any patch
    containing an `access` field, so previously a team-admin could PATCH a
    provider credential with `{credential_info: {access: {teams: [...]}}}`
    and reach `decide_credential_patch`, which would Allow because the patch
    is just "add own team to access.teams". Gate must look at the STORED
    credential's type, not the patch body.
    """
    provider_cred = CredentialItem(
        credential_name="openai-prod",
        credential_values={"api_key": "sk-real"},
        credential_info={"custom_llm_provider": "openai"},
    )
    monkeypatch.setattr(litellm, "credential_list", [provider_cred])
    _connected_db.find_by_name = AsyncMock(return_value=provider_cred)
    _connected_db.update_by_name = AsyncMock()
    _patch_team_admin_lookup["ids"] = frozenset({"team-T"})

    with pytest.raises(HTTPException) as exc:
        await endpoints.update_credential(
            request=MagicMock(),
            fastapi_response=MagicMock(),
            credential=UpdateCredentialItem(
                credential_info={"access": {"teams": ["team-T"]}},
            ),
            credential_name="openai-prod",
            user_api_key_dict=_team_admin_of(["team-T"]),
        )
    assert exc.value.status_code == 403
    _connected_db.update_by_name.assert_not_awaited()


@pytest.mark.asyncio
async def test_team_admin_can_revoke_own_team_grant(_connected_db, _patch_team_admin_lookup, monkeypatch):
    """A team-admin saving an access list without their own team_id revokes it."""
    existing = CredentialItem(
        credential_name="dest",
        credential_values={"langfuse_host": "h"},
        credential_info={
            "credential_type": "logging",
            "description": "tenant Langfuse",
            "host": "https://cloud.langfuse.com",
            "access": {"teams": ["team-existing", "team-T"]},
        },
    )
    monkeypatch.setattr(litellm, "credential_list", [existing])
    _connected_db.find_by_name = AsyncMock(return_value=existing)
    _connected_db.update_by_name = AsyncMock()
    _patch_team_admin_lookup["ids"] = frozenset({"team-T"})

    result = await endpoints.update_credential(
        request=MagicMock(),
        fastapi_response=MagicMock(),
        credential=CredentialItem(
            credential_name="dest",
            credential_values={},
            credential_info={"access": {"teams": ["team-existing"]}},
        ),
        credential_name="dest",
        user_api_key_dict=_team_admin_of(["team-T"]),
    )
    assert result["success"] is True
    _connected_db.update_by_name.assert_awaited_once()


@pytest.mark.asyncio
async def test_team_admin_cannot_grant_foreign_team(_connected_db, _patch_team_admin_lookup, monkeypatch):
    monkeypatch.setattr(litellm, "credential_list", [_resident_logging_dest()])
    _patch_team_admin_lookup["ids"] = frozenset({"team-T"})

    with pytest.raises(HTTPException) as exc:
        await endpoints.update_credential(
            request=MagicMock(),
            fastapi_response=MagicMock(),
            credential=CredentialItem(
                credential_name="dest",
                credential_values={},
                credential_info={"access": {"teams": ["team-existing", "team-foreign"]}},
            ),
            credential_name="dest",
            user_api_key_dict=_team_admin_of(["team-T"]),
        )
    assert exc.value.status_code == 403
    assert "team-foreign" in exc.value.detail["error"]


@pytest.mark.asyncio
async def test_team_admin_cannot_rotate_credential_values(_connected_db, _patch_team_admin_lookup, monkeypatch):
    monkeypatch.setattr(litellm, "credential_list", [_resident_logging_dest()])
    _patch_team_admin_lookup["ids"] = frozenset({"team-T"})

    with pytest.raises(HTTPException) as exc:
        await endpoints.update_credential(
            request=MagicMock(),
            fastapi_response=MagicMock(),
            credential=CredentialItem(
                credential_name="dest",
                credential_values={"public_key": "pk-stolen"},
                credential_info={"access": {"teams": ["team-existing", "team-T"]}},
            ),
            credential_name="dest",
            user_api_key_dict=_team_admin_of(["team-T"]),
        )
    assert exc.value.status_code == 403
    # Endpoint collapses non-caller-input Deny reasons to the opaque message
    # so PATCH /credentials/{name} can't be used as an existence oracle.
    assert exc.value.detail["error"] == OPAQUE_DENY_REASON


@pytest.mark.asyncio
async def test_team_admin_cannot_flip_global(_connected_db, _patch_team_admin_lookup, monkeypatch):
    monkeypatch.setattr(litellm, "credential_list", [_resident_logging_dest()])
    _patch_team_admin_lookup["ids"] = frozenset({"team-T"})

    with pytest.raises(HTTPException) as exc:
        await endpoints.update_credential(
            request=MagicMock(),
            fastapi_response=MagicMock(),
            credential=CredentialItem(
                credential_name="dest",
                credential_values={},
                credential_info={"access": {"global": True}},
            ),
            credential_name="dest",
            user_api_key_dict=_team_admin_of(["team-T"]),
        )
    assert exc.value.status_code == 403
    assert exc.value.detail["error"] == OPAQUE_DENY_REASON


@pytest.mark.asyncio
async def test_get_credentials_shows_only_in_scope_destinations_for_non_admin(monkeypatch, _patch_team_admin_lookup):
    """Leak regression (Veria #1): a non-proxy-admin sees only destinations granted
    to a scope they administer, not every logging destination. The caller admins
    team-existing, so they see the destination granted to team-existing but never
    the provider credential and never a destination scoped to another team."""
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(  # provider credential: never visible to a non-admin
                credential_name="openai",
                credential_values={"api_key": "sk-secret"},
                credential_info={"custom_llm_provider": "openai"},
            ),
            CredentialItem(  # granted to team-existing
                credential_name="poc-langfuse",
                credential_values={"public_key": "pk-1"},
                credential_info=_DEST_WITH_TEAMS,
            ),
            CredentialItem(  # granted to a DIFFERENT team: must stay hidden
                credential_name="other-team-dest",
                credential_values={},
                credential_info={
                    "credential_type": "logging",
                    "description": "arize",
                    "access": {"teams": ["team-other"]},
                },
            ),
        ],
    )
    _patch_team_admin_lookup["ids"] = frozenset({"team-existing"})
    response = await endpoints.get_credentials(
        request=MagicMock(),
        fastapi_response=MagicMock(),
        user_api_key_dict=_team_admin_of(["team-existing"]),
    )
    names = [c["credential_name"] for c in response["credentials"]]
    assert names == ["poc-langfuse"]


@pytest.mark.asyncio
async def test_get_credentials_masks_otel_headers_only_for_non_admin(monkeypatch, _patch_team_admin_lookup):
    raw_headers = "Authorization=Bearer collector-secret,x-api-key=api-secret"
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="generic-otel",
                credential_values={
                    "otel_endpoint": "https://collector.example.com/v1/traces",
                    "otel_headers": raw_headers,
                },
                credential_info={
                    "credential_type": "logging",
                    "description": "generic",
                    "access": {"teams": ["team-existing"]},
                },
            ),
        ],
    )
    _patch_team_admin_lookup["ids"] = frozenset({"team-existing"})

    response = await endpoints.get_credentials(
        request=MagicMock(),
        fastapi_response=MagicMock(),
        user_api_key_dict=_team_admin_of(["team-existing"]),
    )

    values = response["credentials"][0]["credential_values"]
    assert values == {
        "otel_endpoint": "https://collector.example.com/v1/traces",
        "otel_headers": "********",
    }


@pytest.mark.asyncio
async def test_get_credentials_hides_out_of_scope_destination(monkeypatch, _patch_team_admin_lookup):
    """The exact leak: a team-admin of an unrelated team must see none of another
    team's destinations. Pre-fix, get_credentials returned every logging
    destination to any team/org admin regardless of the destination's access."""
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="poc-langfuse",
                credential_values={"public_key": "pk-1"},
                credential_info=_DEST_WITH_TEAMS,  # granted to team-existing
            ),
        ],
    )
    _patch_team_admin_lookup["ids"] = frozenset({"team-T"})  # NOT team-existing
    response = await endpoints.get_credentials(
        request=MagicMock(),
        fastapi_response=MagicMock(),
        user_api_key_dict=_team_admin_of(["team-T"]),
    )
    assert response["credentials"] == []


@pytest.mark.asyncio
async def test_get_credentials_shows_org_scoped_destination_to_org_admin(monkeypatch, _patch_team_admin_lookup):
    """An org-admin sees a destination granted to their org via access.orgs, matched
    against the org-admin scope (not just team ids)."""
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="org-dest",
                credential_values={},
                credential_info={
                    "credential_type": "logging",
                    "description": "arize",
                    "access": {"orgs": ["org-1"]},
                },
            ),
        ],
    )
    _patch_team_admin_lookup["ids"] = frozenset()
    _patch_team_admin_lookup["org_ids"] = frozenset({"org-1"})
    response = await endpoints.get_credentials(
        request=MagicMock(),
        fastapi_response=MagicMock(),
        user_api_key_dict=UserAPIKeyAuth(api_key="k", user_role=LitellmUserRoles.INTERNAL_USER, user_id="oa"),
    )
    names = [c["credential_name"] for c in response["credentials"]]
    assert names == ["org-dest"]


@pytest.mark.asyncio
async def test_get_credentials_forbidden_for_plain_user(monkeypatch, _patch_team_admin_lookup):
    """Veria F2 regression: a plain internal_user (no team-admin or
    org-admin status anywhere) gets 403, NOT a filtered list. The previous
    handler returned destination names, hosts, and scope metadata to any
    authenticated caller because the route gate was widened to support
    team-admin self-assignment.
    """
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="poc-langfuse",
                credential_values={"public_key": "pk-1"},
                credential_info=_DEST_WITH_TEAMS,
            ),
        ],
    )
    _patch_team_admin_lookup["ids"] = frozenset()  # admins nothing

    with pytest.raises(HTTPException) as exc:
        await endpoints.get_credentials(
            request=MagicMock(),
            fastapi_response=MagicMock(),
            user_api_key_dict=UserAPIKeyAuth(
                api_key="k",
                user_role=LitellmUserRoles.INTERNAL_USER,
                user_id="plain-user",
            ),
        )
    assert exc.value.status_code == 403
    assert "team-admin" in exc.value.detail["error"]


def test_patch_credentials_route_targets_update_credential():
    """Regression: the @router.patch decorator on /credentials/{name:path} must
    decorate update_credential, not one of the extracted helpers. A misplaced
    decorator landed once during the 7ecc1d49 split and the unit tests didn't
    catch it because they import the handler function directly; this asserts
    the FastAPI routing table actually points at update_credential.
    """
    from fastapi.routing import APIRoute

    patch_route = next(
        route
        for route in endpoints.router.routes
        if isinstance(route, APIRoute)
        and route.path == "/credentials/{credential_name:path}"
        and "PATCH" in route.methods
    )
    assert patch_route.endpoint is endpoints.update_credential


@pytest.mark.asyncio
async def test_patch_credentials_does_not_leak_credential_type(_connected_db, _patch_team_admin_lookup, monkeypatch):
    """Existence-oracle regression: a team-admin probing a credential they don't own
    must NOT be able to distinguish "logging credential, not yours" from
    "provider credential" or "doesn't exist" by comparing 403 detail strings.

    Pre-fix the decider returned reasons like "access.global is proxy-admin only"
    while `_require_proxy_admin` returned a fixed string, so the same probe
    (e.g. PATCH `{credential_info: {access: {global: true}}}`) would yield
    different bodies depending on the stored type. All three paths now return
    the same opaque message.
    """
    provider = CredentialItem(
        credential_name="openai-prod",
        credential_values={"api_key": "sk-real"},
        credential_info={"custom_llm_provider": "openai"},
    )
    logging_other = CredentialItem(
        credential_name="other-langfuse",
        credential_values={"langfuse_host": "h"},
        credential_info={
            "credential_type": "logging",
            "description": "tenant Langfuse",
            "host": "https://cloud.langfuse.com",
            "access": {"teams": ["team-other"]},
        },
    )
    monkeypatch.setattr(litellm, "credential_list", [provider, logging_other])
    _connected_db.find_by_name = AsyncMock(
        side_effect=lambda name: provider if name == "openai-prod" else logging_other
    )
    # Caller is team-admin of team-T (NOT team-other, so can't legitimately
    # edit logging_other either), probing with a patch that touches a
    # decider-protected field to maximally expose any branch divergence.
    _patch_team_admin_lookup["ids"] = frozenset({"team-T"})
    probe_patch = UpdateCredentialItem(
        credential_info={"access": {"global": True}},
    )

    bodies: list[object] = []
    for name in ("openai-prod", "other-langfuse", "does-not-exist"):
        with pytest.raises(HTTPException) as exc:
            await endpoints.update_credential(
                request=MagicMock(),
                fastapi_response=MagicMock(),
                credential=probe_patch,
                credential_name=name,
                user_api_key_dict=_team_admin_of(["team-T"]),
            )
        assert exc.value.status_code == 403
        bodies.append(exc.value.detail)

    assert bodies[0] == bodies[1] == bodies[2] == {"error": OPAQUE_DENY_REASON}


@pytest.mark.asyncio
async def test_patch_credentials_echoes_foreign_team_id_to_legit_team_admin(
    _connected_db, _patch_team_admin_lookup, monkeypatch
):
    """The one accepted leak: when a team-admin tries to grant a team_id they
    typed in the patch and don't admin, the response names that team_id so
    the UI can render a useful error. The team_id was caller input, so it
    isn't an existence oracle (the caller already knew the value).
    """
    monkeypatch.setattr(litellm, "credential_list", [_resident_logging_dest()])
    _connected_db.find_by_name = AsyncMock(return_value=_resident_logging_dest())
    _patch_team_admin_lookup["ids"] = frozenset({"team-T"})

    with pytest.raises(HTTPException) as exc:
        await endpoints.update_credential(
            request=MagicMock(),
            fastapi_response=MagicMock(),
            credential=UpdateCredentialItem(
                credential_info={"access": {"teams": ["team-existing", "team-foreign"]}},
            ),
            credential_name="dest",
            user_api_key_dict=_team_admin_of(["team-T"]),
        )
    assert exc.value.status_code == 403
    assert "team-foreign" in exc.value.detail["error"]


@pytest.mark.asyncio
async def test_get_credentials_returns_all_for_proxy_admin(monkeypatch):
    raw_headers = "Authorization=Bearer collector-secret,x-api-key=api-secret"
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="openai",
                credential_values={"api_key": "sk-secret"},
                credential_info={"custom_llm_provider": "openai"},
            ),
            CredentialItem(
                credential_name="poc-langfuse",
                credential_values={"public_key": "pk-1"},
                credential_info=_DEST_WITH_TEAMS,
            ),
            CredentialItem(
                credential_name="generic-otel",
                credential_values={"otel_headers": raw_headers},
                credential_info={
                    "credential_type": "logging",
                    "description": "generic",
                },
            ),
        ],
    )
    response = await endpoints.get_credentials(
        request=MagicMock(),
        fastapi_response=MagicMock(),
        user_api_key_dict=_admin(),
    )
    names = sorted(c["credential_name"] for c in response["credentials"])
    assert names == ["generic-otel", "openai", "poc-langfuse"]
    generic = next(c for c in response["credentials"] if c["credential_name"] == "generic-otel")
    assert generic["credential_values"]["otel_headers"] == raw_headers


@pytest.mark.asyncio
async def test_authorize_patch_malformed_stored_access_does_not_500(
    _patch_team_admin_lookup,
):
    """A stored logging destination whose ``access`` carries a key the strict model
    forbids (e.g. legacy data, or data written before the write gate rejected unknown
    keys) must not 500 every PATCH. Fail closed: a non-admin gets 403, the proxy admin
    is still allowed through. Pre-fix the unguarded ``CredentialInfo.model_validate``
    on the stored row raised an uncaught ``ValidationError`` for all callers."""
    malformed = CredentialItem(
        credential_name="legacy-dest",
        credential_values={"langfuse_host": "h"},
        credential_info={
            "credential_type": "logging",
            "access": {"global": True, "legacy_field": "x"},
        },
    )
    patch = UpdateCredentialItem(credential_info={"access": {"teams": ["team-T"]}})
    _patch_team_admin_lookup["ids"] = frozenset({"team-T"})

    with pytest.raises(HTTPException) as exc:
        await endpoints._authorize_credential_patch(
            credential_name="legacy-dest",
            patch=patch,
            existing=malformed,
            user_api_key_dict=_team_admin_of(["team-T"]),
            prisma_client=MagicMock(),
        )
    assert exc.value.status_code == 403
    assert exc.value.detail == {"error": OPAQUE_DENY_REASON}

    assert (
        await endpoints._authorize_credential_patch(
            credential_name="legacy-dest",
            patch=patch,
            existing=malformed,
            user_api_key_dict=_admin(),
            prisma_client=MagicMock(),
        )
        is None
    )
