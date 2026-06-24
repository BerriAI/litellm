"""Admin-gating on credential mutations for logging destinations.

Logging credentials carry ``credential_info.access`` that controls where other
tenants' traces export, so create/update/delete of a logging credential is
proxy-admin only. Provider credentials keep their pre-existing (ungated) behavior.
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
async def test_create_provider_credential_not_gated(_connected_db):
    """A non-logging (provider) credential keeps its pre-existing ungated behavior."""
    result = await endpoints.create_credential(
        request=MagicMock(),
        fastapi_response=MagicMock(),
        credential=CreateCredentialItem(
            credential_name="openai",
            credential_values={"api_key": "sk"},
            credential_info={"custom_llm_provider": "openai"},
        ),
        user_api_key_dict=_member(),
    )
    assert result["success"] is True


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
