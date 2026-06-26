"""Validation for admin-owned logging-exporter assignment on key/team/org.

The single ``validate_logging_exporter_assignment`` runs across all four
endpoints (``/team/new``, ``/team/update``, ``/key/generate``, ``/key/update``,
``/organization/*``); each call site computes the relevant
``caller_is_team_admin`` / ``caller_is_org_admin`` flags from the loaded
team or org and passes them in. Proxy admin always passes.
"""

import os
import sys

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm.models.credentials import CredentialItem
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.management_endpoints.logging_exporter_validation import (
    is_admin_gated_credential_info,
    validate_credential_access,
    validate_logging_exporter_assignment,
)


@pytest.fixture
def _registry():
    original = litellm.credential_list
    litellm.credential_list = [
        CredentialItem(
            credential_name="langfuse-eu",
            credential_values={},
            credential_info={
                "credential_type": "logging",
                "description": "langfuse_otel",
            },
        ),
        CredentialItem(
            credential_name="openai-key",
            credential_values={},
            credential_info={"custom_llm_provider": "openai"},  # provider credential
        ),
    ]
    try:
        yield
    finally:
        litellm.credential_list = original


def _admin():
    return UserAPIKeyAuth(api_key="k", user_role=LitellmUserRoles.PROXY_ADMIN)


def _non_admin():
    return UserAPIKeyAuth(api_key="k", user_role=LitellmUserRoles.INTERNAL_USER)


def _ok(metadata):
    return {"logging_exporters": metadata}


# --- Role allow paths -------------------------------------------------------


def test_proxy_admin_always_allowed(_registry):
    """No flags needed; proxy_admin role suffices."""
    validate_logging_exporter_assignment(_ok(["langfuse-eu"]), _admin())


def test_team_admin_flag_allows_non_admin(_registry):
    """A non-admin caller flagged caller_is_team_admin=True passes."""
    validate_logging_exporter_assignment(
        _ok(["langfuse-eu"]),
        _non_admin(),
        caller_is_team_admin=True,
    )


def test_org_admin_flag_allows_non_admin(_registry):
    """A non-admin caller flagged caller_is_org_admin=True passes."""
    validate_logging_exporter_assignment(
        _ok(["langfuse-eu"]),
        _non_admin(),
        caller_is_org_admin=True,
    )


def test_both_flags_set_allows_non_admin(_registry):
    """Setting both flags is fine; they're independent OR-ed allows."""
    validate_logging_exporter_assignment(
        _ok(["langfuse-eu"]),
        _non_admin(),
        caller_is_team_admin=True,
        caller_is_org_admin=True,
    )


def test_non_admin_with_no_flags_is_forbidden(_registry):
    """The headline deny: internal_user with no team/org admin context."""
    with pytest.raises(HTTPException) as exc:
        validate_logging_exporter_assignment(_ok(["langfuse-eu"]), _non_admin())
    assert exc.value.status_code == 403


def test_proxy_admin_overrides_falsy_flags(_registry):
    """proxy_admin role wins even when both flags are False."""
    validate_logging_exporter_assignment(
        _ok(["langfuse-eu"]),
        _admin(),
        caller_is_team_admin=False,
        caller_is_org_admin=False,
    )


# --- Shape / registry checks (run regardless of who's calling) --------------


def test_unknown_credential_rejected_for_admin(_registry):
    with pytest.raises(HTTPException) as exc:
        validate_logging_exporter_assignment(_ok(["does-not-exist"]), _admin())
    assert exc.value.status_code == 400


def test_unknown_credential_rejected_for_team_admin(_registry):
    with pytest.raises(HTTPException) as exc:
        validate_logging_exporter_assignment(
            _ok(["does-not-exist"]),
            _non_admin(),
            caller_is_team_admin=True,
        )
    assert exc.value.status_code == 400


def test_provider_credential_rejected(_registry):
    """openai-key exists but is provider-typed, not a logging destination."""
    with pytest.raises(HTTPException) as exc:
        validate_logging_exporter_assignment(_ok(["openai-key"]), _admin())
    assert exc.value.status_code == 400


def test_non_list_is_rejected(_registry):
    with pytest.raises(HTTPException) as exc:
        validate_logging_exporter_assignment(
            {"logging_exporters": "langfuse-eu"}, _admin()
        )
    assert exc.value.status_code == 400


def test_noop_when_field_absent(_registry):
    """An update that does not touch logging_exporters skips the gate even
    for a non-admin with no flags."""
    validate_logging_exporter_assignment({"some_other_key": 1}, _non_admin())
    validate_logging_exporter_assignment(None, _non_admin())


# --- is_admin_gated_credential_info / validate_credential_access ------------


@pytest.mark.parametrize(
    "credential_info, gated",
    [
        ({"credential_type": "logging"}, True),
        ({"access": {"global": True}}, True),
        ({"credential_type": "logging", "access": {"teams": ["t"]}}, True),
        ({"custom_llm_provider": "openai"}, False),
        ({}, False),
        (None, False),
    ],
)
def test_is_admin_gated_credential_info(credential_info, gated):
    assert is_admin_gated_credential_info(credential_info) is gated


def test_validate_credential_access_accepts_valid_object():
    validate_credential_access(
        {"access": {"global": False, "teams": ["t1", "t2"], "orgs": ["o1"]}}
    )


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
