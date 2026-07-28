"""Admin-gating on credential mutations.

Every ``/credentials`` operation -- GET, POST, PATCH, DELETE, for both logging
destinations and provider credentials -- is proxy-admin only (a proxy-admin-viewer
may read). Admin-owned OTEL logging destinations and their ``access`` scoping are
managed exclusively by the proxy admin; tenants never read or mutate them over the
API. Trace routing to identity-scoped destinations happens server-side in the
resolver, independent of this surface.
"""

import os
import sys

import pytest
from fastapi import HTTPException
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
import litellm.proxy.credential_endpoints.endpoints as endpoints
from litellm.models.credentials import CredentialItem, UpdateCredentialItem
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
    """POST is proxy-admin only for provider and logging credentials alike."""
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


def test_update_db_credential_preserves_untouched_access_subfields():
    """An access patch carrying only `teams` must NOT clobber existing
    `global` / `orgs`: a top-level replace of the access object would silently
    drop access.global=true and any access.orgs entries the patch never named.
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
    # A partial patch touching only access.teams; global/orgs are not sent.
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


# --- PATCH gating (proxy-admin only) ----------------------------------------

@pytest.mark.asyncio
async def test_provider_credential_patch_forbidden_for_non_admin(
    _connected_db, monkeypatch
):
    """A non-admin cannot PATCH a provider credential (or any credential):
    without the gate a non-admin could rotate the upstream api_key."""
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
async def test_get_credentials_forbidden_for_non_admin(monkeypatch):
    """A non-proxy-admin (team-admin, org-admin, or plain internal_user) gets 403.
    Credentials, including admin-owned logging destinations, are proxy-admin only;
    the list is never exposed to a tenant over the API."""
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="poc-langfuse",
                credential_values={"public_key": "pk-1"},
                credential_info=_LOGGING_INFO,
            ),
        ],
    )
    with pytest.raises(HTTPException) as exc:
        await endpoints.get_credentials(
            request=MagicMock(),
            fastapi_response=MagicMock(),
            user_api_key_dict=_member(),
        )
    assert exc.value.status_code == 403


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
                credential_info={
                    "credential_type": "logging",
                    "description": "langfuse_otel",
                    "access": {"teams": ["team-A"]},
                },
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
    generic = next(
        c for c in response["credentials"] if c["credential_name"] == "generic-otel"
    )
    assert generic["credential_values"]["otel_headers"] == raw_headers


@pytest.mark.asyncio
async def test_get_credentials_admin_viewer_gets_full_list_fully_masked(monkeypatch):
    """PROXY_ADMIN_VIEW_ONLY keeps read parity with PROXY_ADMIN on this endpoint:
    the full credential list, including provider credentials, with every stored
    value constant-masked so the read-only role receives no usable secret."""
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
        user_api_key_dict=UserAPIKeyAuth(
            api_key="k", user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY
        ),
    )
    names = sorted(c["credential_name"] for c in response["credentials"])
    assert names == ["generic-otel", "openai"]
    assert all(
        value == "********"
        for c in response["credentials"]
        for value in c["credential_values"].values()
    )
    assert "collector-secret" not in str(response)
    assert "sk-secret" not in str(response)


