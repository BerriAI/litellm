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


# --- provider credentials keep base replace semantics (merge is logging-only) ---

def test_update_db_credential_replaces_info_for_provider_credential():
    """The subfield merge is scoped to logging destinations. A provider credential
    patch replaces credential_info wholesale (base behavior): omitted keys drop, so a
    partial patch is a full replace, not a merge -- merging non-destination creds would
    silently resurrect stale provider metadata the caller meant to remove."""
    from litellm.proxy.credential_endpoints.endpoints import update_db_credential

    db = CredentialItem(
        credential_name="openai",
        credential_values={},
        credential_info={"custom_llm_provider": "openai", "stale": "keepout"},
    )
    patch = CredentialItem(
        credential_name="openai",
        credential_values={},
        credential_info={"custom_llm_provider": "azure"},
    )

    merged = update_db_credential(db, patch)

    assert merged.credential_info == {"custom_llm_provider": "azure"}


def test_sync_in_memory_credential_merges_provider_info(monkeypatch):
    """The in-memory mirror merges credential_info for every credential (pre-PR
    parity). A partial provider PATCH (e.g. only description) must not drop
    custom_llm_provider from litellm.credential_list; a wholesale replace made the
    credential unresolvable (401) until the next scheduled DB reload."""
    from litellm.proxy.credential_endpoints.endpoints import _sync_in_memory_credential
    from litellm.types.utils import UpdateCredentialItem

    existing = CredentialItem(
        credential_name="openai-prod",
        credential_values={"api_key": "enc"},
        credential_info={"custom_llm_provider": "openai", "keepme": "important"},
    )
    monkeypatch.setattr(litellm, "credential_list", [existing])
    patch = UpdateCredentialItem(credential_info={"description": "just a label"})
    merged = CredentialItem(
        credential_name="openai-prod",
        credential_values={"api_key": "enc"},
        credential_info={"description": "just a label"},
    )

    _sync_in_memory_credential(old_name="openai-prod", merged=merged, patch=patch)

    in_memory = next(c for c in litellm.credential_list if c.credential_name == "openai-prod")
    assert in_memory.credential_info == {
        "custom_llm_provider": "openai",
        "keepme": "important",
        "description": "just a label",
    }


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
