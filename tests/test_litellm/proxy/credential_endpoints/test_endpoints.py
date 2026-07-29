"""Authorization on credential endpoints, scoped to trace destinations.

Trace destinations (``credential_type == "logging"``) are proxy-admin-managed
regardless of a key's ``allowed_routes``: create/update/delete require the proxy
admin, and reads of a destination are limited to the proxy admin (or a
proxy-admin-viewer). Provider credentials are not trace destinations, so the
handlers do not gate them by role; their authorization stays at the route layer
(a non-admin only reaches ``/credentials`` when an admin delegated it via
``allowed_routes``), exactly as it was before the destinations feature. Trace
routing to identity-scoped destinations happens server-side in the resolver,
independent of this surface.
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
_PROVIDER_INFO = {"custom_llm_provider": "openai"}


def _logging_cred(name="dest"):
    return CredentialItem(
        credential_name=name,
        credential_values={"langfuse_host": "h"},
        credential_info=_LOGGING_INFO,
    )


def _provider_cred(name="openai-prod"):
    return CredentialItem(
        credential_name=name,
        credential_values={"api_key": "sk-real"},
        credential_info=_PROVIDER_INFO,
    )


@pytest.fixture
def _connected_db(monkeypatch):
    """A working prisma_client + repository so an allowed caller reaches success."""
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-test-salt-key")
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    monkeypatch.setattr(proxy_server, "llm_router", None)
    monkeypatch.setattr(litellm, "credential_list", [])
    repo = MagicMock()
    repo.create = AsyncMock()
    repo.delete_by_name = AsyncMock()
    repo.update_by_name = AsyncMock()
    repo.find_by_name = AsyncMock(return_value=None)
    monkeypatch.setattr(endpoints, "CredentialsRepository", lambda _client: repo)
    monkeypatch.setattr(endpoints.CredentialAccessor, "upsert_credentials", lambda creds: None)
    return repo


# --- create ------------------------------------------------------------------

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
async def test_create_provider_credential_allowed_for_non_admin(_connected_db):
    """Provider credentials are not trace destinations, so create is not gated by
    role in the handler: a key that route-auth admitted (delegated via
    ``allowed_routes``) keeps its pre-feature ability to create one."""
    result = await endpoints.create_credential(
        request=MagicMock(),
        fastapi_response=MagicMock(),
        credential=CreateCredentialItem(
            credential_name="openai",
            credential_values={"api_key": "sk"},
            credential_info=_PROVIDER_INFO,
        ),
        user_api_key_dict=_member(),
    )
    assert result["success"] is True
    _connected_db.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_provider_credential_cannot_smuggle_destination(_connected_db):
    """A non-admin cannot create a destination by tagging a 'provider' create with
    credential_type=logging: the gate keys off the payload's type, not its name."""
    with pytest.raises(HTTPException) as exc:
        await endpoints.create_credential(
            request=MagicMock(),
            fastapi_response=MagicMock(),
            credential=CreateCredentialItem(
                credential_name="looks-like-provider",
                credential_values={"otel_endpoint": "https://attacker/v1/traces"},
                credential_info={"credential_type": "logging", "access": {"global": True}},
            ),
            user_api_key_dict=_member(),
        )
    assert exc.value.status_code == 403
    _connected_db.create.assert_not_awaited()


# --- update ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_logging_credential_forbidden_for_non_admin(_connected_db):
    _connected_db.find_by_name = AsyncMock(return_value=_logging_cred())
    with pytest.raises(HTTPException) as exc:
        await endpoints.update_credential(
            request=MagicMock(),
            fastapi_response=MagicMock(),
            credential=UpdateCredentialItem(credential_info={"access": {"global": True}}),
            credential_name="dest",
            user_api_key_dict=_member(),
        )
    assert exc.value.status_code == 403
    _connected_db.update_by_name.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_existing_logging_credential_forbidden_even_without_logging_patch(_connected_db):
    """A non-admin cannot edit a stored destination's values, even with a patch that
    omits credential_info: the gate consults the stored (DB) credential too."""
    _connected_db.find_by_name = AsyncMock(return_value=_logging_cred())
    with pytest.raises(HTTPException) as exc:
        await endpoints.update_credential(
            request=MagicMock(),
            fastapi_response=MagicMock(),
            credential=UpdateCredentialItem(credential_values={"langfuse_host": "evil"}),
            credential_name="dest",
            user_api_key_dict=_member(),
        )
    assert exc.value.status_code == 403
    _connected_db.update_by_name.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_provider_credential_allowed_for_non_admin(_connected_db):
    """A non-admin (delegated via allowed_routes) may still patch a provider
    credential; only destinations are handler-gated."""
    _connected_db.find_by_name = AsyncMock(return_value=_provider_cred())
    result = await endpoints.update_credential(
        request=MagicMock(),
        fastapi_response=MagicMock(),
        credential=UpdateCredentialItem(credential_values={"api_key": "sk-rotated"}),
        credential_name="openai-prod",
        user_api_key_dict=_member(),
    )
    assert result["success"] is True
    _connected_db.update_by_name.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_provider_to_destination_forbidden_for_non_admin(_connected_db):
    """Converting a provider credential into a destination (patch adds a logging
    credential_type / access) requires the proxy admin, so a non-admin can't
    escalate a delegated provider credential into a global trace sink."""
    _connected_db.find_by_name = AsyncMock(return_value=_provider_cred())
    with pytest.raises(HTTPException) as exc:
        await endpoints.update_credential(
            request=MagicMock(),
            fastapi_response=MagicMock(),
            credential=UpdateCredentialItem(
                credential_info={"credential_type": "logging", "access": {"global": True}},
            ),
            credential_name="openai-prod",
            user_api_key_dict=_member(),
        )
    assert exc.value.status_code == 403
    _connected_db.update_by_name.assert_not_awaited()


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


# --- delete ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_logging_credential_forbidden_for_non_admin(_connected_db):
    _connected_db.find_by_name = AsyncMock(return_value=_logging_cred())
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
async def test_delete_provider_credential_allowed_for_non_admin(_connected_db):
    """Deleting a provider credential is not handler-gated (route-auth governs it),
    so a delegated non-admin keeps its pre-feature ability to delete one."""
    _connected_db.find_by_name = AsyncMock(return_value=_provider_cred())
    result = await endpoints.delete_credential(
        request=MagicMock(),
        fastapi_response=MagicMock(),
        credential_name="openai-prod",
        user_api_key_dict=_member(),
    )
    assert result["success"] is True
    _connected_db.delete_by_name.assert_awaited_once()


# --- reads -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_credentials_hides_destinations_from_non_admin(monkeypatch):
    """A non-admin list returns provider credentials (pre-feature behavior) but
    never trace destinations, which stay proxy-admin information."""
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="openai",
                credential_values={"api_key": "sk-secret"},
                credential_info=_PROVIDER_INFO,
            ),
            CredentialItem(
                credential_name="poc-langfuse",
                credential_values={"public_key": "pk-1"},
                credential_info=_LOGGING_INFO,
            ),
        ],
    )
    response = await endpoints.get_credentials(
        request=MagicMock(),
        fastapi_response=MagicMock(),
        user_api_key_dict=_member(),
    )
    names = [c["credential_name"] for c in response["credentials"]]
    assert names == ["openai"]  # destination hidden, provider visible


@pytest.mark.asyncio
async def test_get_credential_by_name_destination_forbidden_for_non_admin(monkeypatch):
    monkeypatch.setattr(litellm, "credential_list", [_logging_cred("poc-langfuse")])
    with pytest.raises(HTTPException) as exc:
        await endpoints.get_credential_by_name(
            request=MagicMock(),
            fastapi_response=MagicMock(),
            credential_name="poc-langfuse",
            user_api_key_dict=_member(),
        )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_get_credential_by_name_provider_allowed_for_non_admin(monkeypatch):
    """A provider credential read by name is not handler-gated (masked as before)."""
    monkeypatch.setattr(litellm, "credential_list", [_provider_cred("openai-prod")])
    result = await endpoints.get_credential_by_name(
        request=MagicMock(),
        fastapi_response=MagicMock(),
        credential_name="openai-prod",
        user_api_key_dict=_member(),
    )
    assert result.credential_name == "openai-prod"
    assert result.credential_values["api_key"] != "sk-real"  # still masked


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
    generic = next(c for c in response["credentials"] if c["credential_name"] == "generic-otel")
    # otel_headers carries the collector auth token, so the masker treats it as a
    # secret key: readable prefix only, never the full value.
    assert generic["credential_values"]["otel_headers"] != raw_headers
    assert "collector-secret" not in str(response)


@pytest.mark.asyncio
async def test_get_credentials_admin_viewer_reads_same_masked_list_as_admin(monkeypatch):
    """PROXY_ADMIN_VIEW_ONLY keeps read parity with PROXY_ADMIN on this endpoint:
    the identical credential list through the identical masker, with no raw
    secret in either response."""
    raw_headers = "Authorization=Bearer collector-secret,x-api-key=api-secret"
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="openai",
                credential_values={"api_key": "sk-secret-value"},
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
    viewer_response = await endpoints.get_credentials(
        request=MagicMock(),
        fastapi_response=MagicMock(),
        user_api_key_dict=UserAPIKeyAuth(api_key="k", user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY),
    )
    admin_response = await endpoints.get_credentials(
        request=MagicMock(),
        fastapi_response=MagicMock(),
        user_api_key_dict=_admin(),
    )
    assert viewer_response == admin_response
    names = sorted(c["credential_name"] for c in viewer_response["credentials"])
    assert names == ["generic-otel", "openai"]
    assert "collector-secret" not in str(viewer_response)
    assert "sk-secret-value" not in str(viewer_response)
