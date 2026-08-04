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

import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
import litellm.proxy.credential_endpoints.endpoints as endpoints
from litellm.models.credentials import CredentialItem
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth


def _admin():
    return UserAPIKeyAuth(api_key="k", user_role=LitellmUserRoles.PROXY_ADMIN)


# --- credential_info replace semantics, identical for every credential kind ---

@pytest.mark.parametrize(
    "info, patch_info, expected",
    [
        (
            {"custom_llm_provider": "openai", "stale": "keepout"},
            {"custom_llm_provider": "azure"},
            {"custom_llm_provider": "azure"},
        ),
        (
            {"credential_type": "logging", "description": "arize", "access": {"global": True}},
            {"credential_type": "logging", "description": "arize", "access": {"teams": ["t1"]}},
            {"credential_type": "logging", "description": "arize", "access": {"teams": ["t1"]}},
        ),
    ],
    ids=["provider", "logging-destination"],
)
def test_update_db_credential_replaces_info_wholesale(info, patch_info, expected):
    """``credential_info`` is replaced, never merged, for every credential kind.

    The caller sends the whole object (the body model requires it), so a replace is
    lossless and omitted keys are a deliberate removal. Special-casing logging
    destinations with a subfield merge is what let a fragment reach the write path and
    silently delete sibling metadata such as ``custom_llm_provider``.
    """
    from litellm.proxy.credential_endpoints.endpoints import update_db_credential

    merged = update_db_credential(
        CredentialItem(credential_name="c", credential_values={}, credential_info=info),
        CredentialItem(credential_name="c", credential_values={}, credential_info=patch_info),
    )

    assert merged.credential_info == expected


def test_sync_in_memory_credential_mirrors_the_db_row(monkeypatch):
    """Regression: the routing-live copy must equal the row that was written.

    The in-memory mirror used to merge ``credential_info`` while the DB write replaced
    it, so a field dropped from the row stayed visible in ``litellm.credential_list``
    and on ``GET /credentials`` -- the loss only surfaced on the next reload, long after
    the request that caused it.
    """
    from litellm.proxy.credential_endpoints.endpoints import _sync_in_memory_credential

    existing = CredentialItem(
        credential_name="openai-prod",
        credential_values={"api_key": "enc"},
        credential_info={"custom_llm_provider": "openai", "keepme": "important"},
    )
    monkeypatch.setattr(litellm, "credential_list", [existing])
    patch = CredentialItem(
        credential_name="openai-prod",
        credential_values={},
        credential_info={"description": "just a label"},
    )
    merged = CredentialItem(
        credential_name="openai-prod",
        credential_values={"api_key": "enc"},
        credential_info={"description": "just a label"},
    )

    _sync_in_memory_credential(old_name="openai-prod", merged=merged, patch=patch)

    in_memory = next(c for c in litellm.credential_list if c.credential_name == "openai-prod")
    assert in_memory.credential_info == merged.credential_info


# --- access-shape validation is scoped to logging destinations ---------------

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
async def test_update_credential_raises_rejections_instead_of_answering_200(monkeypatch):
    """Regression: the handler returned the ProxyException instead of raising it, so
    FastAPI serialized it as the body of a 200. A rejected access shape then read as a
    successful write to any client checking the status, including the destination edit
    modal. The rejection has to reach the caller as a 4xx.
    """
    from litellm.proxy._types import ProxyException

    class _Repo:
        def __init__(self, _client):
            pass

        async def find_by_name(self, name):
            return CredentialItem(
                credential_name=name,
                credential_values={},
                credential_info={"credential_type": "logging", "description": "generic"},
            )

    monkeypatch.setattr(endpoints, "CredentialsRepository", _Repo)
    monkeypatch.setattr(endpoints, "validate_credential_access", _boom_400)
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "prisma_client", object(), raising=False)

    with pytest.raises(ProxyException) as excinfo:
        await endpoints.update_credential(
            request=MagicMock(),
            fastapi_response=MagicMock(),
            credential=CredentialItem(
                credential_name="dest",
                credential_values={},
                credential_info={"credential_type": "logging", "access": "everyone"},
            ),
            credential_name="dest",
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
        )
    assert excinfo.value.code in ("400", 400)


def _boom_400(_info):
    from fastapi import HTTPException

    raise HTTPException(status_code=400, detail={"error": "credential_info.access must be an object"})


# --- secret masking on read --------------------------------------------------

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
