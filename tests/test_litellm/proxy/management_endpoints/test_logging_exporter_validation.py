"""Tests for ``validate_credential_access`` -- the shape check on a logging
destination's ``credential_info.access`` at create/update time.

Which identities a destination fires for is governed entirely by ``access`` and
evaluated by the request-time resolver; there is no separate assignment/enable
surface, so this module only guards that a write stores a well-formed ``access``.
"""

import pytest
from fastapi import HTTPException

from litellm.proxy.management_endpoints.logging_exporter_validation import (
    validate_credential_access,
)


def test_validate_credential_access_accepts_valid_object():
    validate_credential_access({"access": {"global": False, "teams": ["t1", "t2"], "orgs": ["o1"]}})


def test_validate_credential_access_noop_without_access():
    validate_credential_access({"credential_type": "logging"})
    validate_credential_access(None)


@pytest.mark.parametrize(
    "access",
    [
        5,  # not an object
        {"global": "yes"},  # global must be bool
        {"teams": "t1"},  # teams must be a list
        {"orgs": [1, 2]},  # orgs must be strings
    ],
)
def test_validate_credential_access_rejects_bad_shape(access):
    with pytest.raises(HTTPException) as exc:
        validate_credential_access({"access": access})
    assert exc.value.status_code == 400


def test_validate_credential_access_rejects_unknown_field():
    """Unknown access keys must be rejected at write time so a destination can never
    be stored in a shape the strict ``CredentialAccess`` read model later refuses to
    parse (which would 500 every subsequent PATCH)."""
    with pytest.raises(HTTPException) as exc:
        validate_credential_access({"access": {"global": True, "legacy_field": "x"}})
    assert exc.value.status_code == 400
    assert "legacy_field" in exc.value.detail["error"]
