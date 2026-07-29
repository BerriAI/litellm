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


# --- partial-PATCH access merge (destinations keep untouched access subfields) ---

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
