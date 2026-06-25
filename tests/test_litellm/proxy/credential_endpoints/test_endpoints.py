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
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
import litellm.proxy.credential_endpoints.endpoints as endpoints
from litellm.models.credentials import CredentialItem
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
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
    monkeypatch.setattr(endpoints, "CredentialsRepository", lambda _client: repo)
    monkeypatch.setattr(
        endpoints.CredentialAccessor, "upsert_credentials", lambda creds: None
    )
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
async def test_update_existing_logging_credential_forbidden_even_without_logging_patch(
    _connected_db, monkeypatch
):
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


@pytest.mark.asyncio
async def test_delete_logging_credential_forbidden_for_non_admin(
    _connected_db, monkeypatch
):
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
async def test_update_db_only_logging_credential_forbidden_for_non_admin(
    _connected_db, monkeypatch
):
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
async def test_delete_db_only_logging_credential_forbidden_for_non_admin(
    _connected_db, monkeypatch
):
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
    ``_caller_team_admin_ids`` resolution without touching the DB.
    """
    return UserAPIKeyAuth(
        api_key="k", user_role=LitellmUserRoles.INTERNAL_USER, user_id="ta-demo"
    )


@pytest.fixture
def _patch_team_admin_lookup(monkeypatch):
    """Substitute the DB-backed team-admin lookup with a configurable mock."""

    holder = {"ids": frozenset()}

    async def _fake(user_api_key_dict, prisma_client):
        return holder["ids"]

    monkeypatch.setattr(endpoints, "_caller_team_admin_ids", _fake)
    return holder


def _resident_logging_dest():
    return CredentialItem(
        credential_name="dest",
        credential_values={"langfuse_host": "h"},
        credential_info=_DEST_WITH_TEAMS,
    )


@pytest.mark.asyncio
async def test_team_admin_can_append_own_team_to_access(
    _connected_db, _patch_team_admin_lookup, monkeypatch
):
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
    _connected_db.update_by_name.assert_awaited_once()


@pytest.mark.asyncio
async def test_team_admin_cannot_grant_foreign_team(
    _connected_db, _patch_team_admin_lookup, monkeypatch
):
    monkeypatch.setattr(litellm, "credential_list", [_resident_logging_dest()])
    _patch_team_admin_lookup["ids"] = frozenset({"team-T"})

    with pytest.raises(HTTPException) as exc:
        await endpoints.update_credential(
            request=MagicMock(),
            fastapi_response=MagicMock(),
            credential=CredentialItem(
                credential_name="dest",
                credential_values={},
                credential_info={
                    "access": {"teams": ["team-existing", "team-foreign"]}
                },
            ),
            credential_name="dest",
            user_api_key_dict=_team_admin_of(["team-T"]),
        )
    assert exc.value.status_code == 403
    assert "team-foreign" in exc.value.detail["error"]


@pytest.mark.asyncio
async def test_team_admin_cannot_rotate_credential_values(
    _connected_db, _patch_team_admin_lookup, monkeypatch
):
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
    assert "credential_values" in exc.value.detail["error"]


@pytest.mark.asyncio
async def test_team_admin_cannot_flip_global(
    _connected_db, _patch_team_admin_lookup, monkeypatch
):
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
    assert "global" in exc.value.detail["error"]


@pytest.mark.asyncio
async def test_get_credentials_filters_to_logging_for_non_admin(monkeypatch):
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
        ],
    )
    response = await endpoints.get_credentials(
        request=MagicMock(),
        fastapi_response=MagicMock(),
        user_api_key_dict=_team_admin_of(["team-T"]),
    )
    names = [c["credential_name"] for c in response["credentials"]]
    assert names == ["poc-langfuse"]


@pytest.mark.asyncio
async def test_get_credentials_returns_all_for_proxy_admin(monkeypatch):
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
        ],
    )
    response = await endpoints.get_credentials(
        request=MagicMock(),
        fastapi_response=MagicMock(),
        user_api_key_dict=_admin(),
    )
    names = sorted(c["credential_name"] for c in response["credentials"])
    assert names == ["openai", "poc-langfuse"]
