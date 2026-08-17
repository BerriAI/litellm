"""Credential endpoint behavior: partial-PATCH access merge, PATCH routing, and
secret masking on read.

Authorization on ``/credentials`` is handled at the route layer (a non-admin key
only reaches these handlers when an admin delegated the route via
``allowed_routes``); the handlers themselves do not gate by role, so no authz is
asserted here. Trace routing to identity-scoped destinations happens server-side
in the resolver, independent of this surface.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
import litellm.proxy.credential_endpoints.endpoints as endpoints
from litellm.models.credentials import CredentialItem
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.proxy_server import app


def _admin():
    return UserAPIKeyAuth(api_key="k", user_role=LitellmUserRoles.PROXY_ADMIN)


# --- credential_info replace semantics ---


@pytest.mark.asyncio
async def test_create_credential_validates_access_only_for_logging(monkeypatch):
    """validate_credential_access runs for a logging destination but never for a
    provider credential. A provider cred carrying an unrelated `access` key must not be
    rejected by the destination access-shape validator (that would 400 a valid
    provider credential the validator was never meant to see)."""
    import litellm.proxy.proxy_server as proxy_server
    from litellm.types.utils import CreateCredentialItem

    monkeypatch.setattr(proxy_server, "prisma_client", None, raising=False)

    class _Validated(Exception):
        pass

    def _spy(_info):
        raise _Validated()

    monkeypatch.setattr(endpoints, "validate_credential_access", _spy)

    async def _create(credential):
        return await endpoints.create_credential(
            request=MagicMock(),
            fastapi_response=MagicMock(),
            credential=credential,
            user_api_key_dict=_admin(),
        )

    logging_cred = CreateCredentialItem(
        credential_name="dest",
        credential_values={"otel_endpoint": "http://collector:4318"},
        credential_info={"credential_type": "logging", "description": "generic", "access": {"global": True}},
    )
    with pytest.raises(_Validated):
        await _create(logging_cred)

    provider_cred = CreateCredentialItem(
        credential_name="openai",
        credential_values={"api_key": "sk"},
        credential_info={"custom_llm_provider": "openai", "access": {"bogus": True}},
    )
    # Validator is skipped; the handler proceeds and fails on the (None) prisma client,
    # a 500 -- never the _Validated sentinel.
    with pytest.raises(Exception) as excinfo:
        await _create(provider_cred)
    assert not isinstance(excinfo.value, _Validated)


# --- PATCH routing regression ------------------------------------------------

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
async def test_get_credentials_masks_secret_values(monkeypatch):
    """GET /credentials masks secret-bearing values; in particular a destination's
    otel_headers (which carries the collector auth token) is never returned raw."""
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
                credential_info={"credential_type": "logging", "description": "generic"},
            ),
        ],
    )
    response = await endpoints.get_credentials(
        request=MagicMock(),
        fastapi_response=MagicMock(),
        user_api_key_dict=_admin(),
    )
    names = sorted(c["credential_name"] for c in response["credentials"])
    assert names == ["generic-otel", "openai"]
    generic = next(c for c in response["credentials"] if c["credential_name"] == "generic-otel")
    # otel_headers carries the collector auth token, so the masker treats it as a
    # secret key: readable prefix only, never the full value.
    assert generic["credential_values"]["otel_headers"] != raw_headers
    assert "collector-secret" not in str(response)
    assert "sk-secret" not in str(response)


@pytest.mark.asyncio
async def test_get_credentials_reports_whether_each_destination_actually_builds(monkeypatch):
    """The dashboard needs the resolver's own verdict, not a second implementation of it.

    Its Scope column read ``credential_info.access`` alone, so a destination the resolver
    excludes (no backend name, or values its adapter rejects) still rendered a scope badge
    and read as live. Reproducing the adapter rules in the frontend would drift from them;
    this field is computed by ``destination_for_credential``, the same function the
    request-time resolver and the team/org disclosure use.
    """
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="builds",
                credential_values={"otel_endpoint": "http://collector.internal:4318/v1/traces"},
                credential_info={"credential_type": "logging", "description": "generic", "access": {"global": True}},
            ),
            CredentialItem(
                credential_name="no-backend",
                credential_values={"otel_endpoint": "http://collector.internal:4318/v1/traces"},
                credential_info={"credential_type": "logging", "access": {"global": True}},
            ),
            CredentialItem(
                credential_name="adapter-rejects",
                credential_values={"langfuse_public_key": "pk-only"},
                credential_info={
                    "credential_type": "logging",
                    "description": "langfuse_otel",
                    "access": {"global": True},
                },
            ),
            CredentialItem(
                credential_name="openai",
                credential_values={"api_key": "sk-secret"},
                credential_info={"custom_llm_provider": "openai"},
            ),
        ],
    )
    response = await endpoints.get_credentials(
        request=MagicMock(),
        fastapi_response=MagicMock(),
        user_api_key_dict=_admin(),
    )
    verdicts = {c["credential_name"]: c.get("resolves_to_destination") for c in response["credentials"]}
    assert verdicts["builds"] is True
    assert verdicts["no-backend"] is False
    assert verdicts["adapter-rejects"] is False
    # A provider credential is not a destination and gets no verdict at all, rather than a
    # False that would render it as a broken destination.
    assert verdicts["openai"] is None


client = TestClient(app)


def _as_admin():
    return UserAPIKeyAuth(api_key="test-key", user_role="proxy_admin")


def _patch_credential(name: str, body: dict):
    missing = object()
    previous_override = app.dependency_overrides.get(user_api_key_auth, missing)
    app.dependency_overrides[user_api_key_auth] = _as_admin
    try:
        return client.patch(
            f"/credentials/{name}",
            json=body,
            headers={"Authorization": "Bearer test-key"},
        )
    finally:
        if previous_override is missing:
            app.dependency_overrides.pop(user_api_key_auth, None)
        else:
            app.dependency_overrides[user_api_key_auth] = previous_override


def test_update_credential_answers_404_when_the_credential_does_not_exist():
    """Regression: the handler used to ``return handle_exception_on_proxy(e)``, which makes
    the exception the response body and lets FastAPI answer 200, so a write the handler
    rejected read as a success to every caller that checks the status. The dashboard's API
    client branches on the status, so it reported a failed edit as applied."""
    with patch("litellm.proxy.proxy_server.prisma_client", MagicMock()), patch(
        "litellm.proxy.credential_endpoints.endpoints.CredentialsRepository"
    ) as repository:
        repository.return_value.find_by_name = AsyncMock(return_value=None)

        response = _patch_credential(
            "definitely-not-there",
            {"credential_name": "definitely-not-there", "credential_values": {"api_key": "sk-x"}, "credential_info": {}},
        )

    assert response.status_code == 404, f"rejected write answered {response.status_code}: {response.text}"
    assert "error" in response.json()


def test_update_credential_answers_500_when_the_database_is_not_connected():
    """The other rejection this handler raises must carry its own status too."""
    with patch("litellm.proxy.proxy_server.prisma_client", None):
        response = _patch_credential(
            "any-name",
            {"credential_name": "any-name", "credential_values": {"api_key": "sk-x"}, "credential_info": {}},
        )

    assert response.status_code == 500, f"rejected write answered {response.status_code}: {response.text}"


def test_update_credential_still_answers_200_on_a_successful_write():
    """The fix must not turn a legitimate update into an error; the dashboard and the
    Playwright credentials spec both assert the success path."""
    stored = CredentialItem(
        credential_name="existing",
        credential_values={"api_key": "sk-old"},
        credential_info={"custom_llm_provider": "openai"},
    )
    with patch("litellm.proxy.proxy_server.prisma_client", MagicMock()), patch(
        "litellm.proxy.proxy_server.master_key", "sk-test-master"
    ), patch("litellm.proxy.credential_endpoints.endpoints.CredentialsRepository") as repository:
        repository.return_value.find_by_name = AsyncMock(return_value=stored)
        repository.return_value.update_by_name = AsyncMock(return_value=None)

        response = _patch_credential(
            "existing",
            {"credential_name": "existing", "credential_values": {"api_key": "sk-new"}, "credential_info": {}},
        )

    assert response.status_code == 200, response.text
    assert response.json()["success"] is True
