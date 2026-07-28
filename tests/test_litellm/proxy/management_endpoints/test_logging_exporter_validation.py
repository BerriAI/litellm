"""Validation for admin-owned logging-exporter assignment on key/team/org.

The single ``validate_logging_exporter_assignment`` runs across every write path
(``/team/new``, ``/team/update``, ``/key/generate``, ``/key/update``,
``/organization/*``). Assigning ``logging_exporters`` is proxy-admin only: a
non-proxy-admin write is rejected, and every named exporter must resolve to a
registered logging credential.
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
    validate_logging_exporter_field,
)


@pytest.fixture
def _registry():
    original = litellm.credential_list
    litellm.credential_list = [
        # global: visible to (and assignable by) every scope.
        CredentialItem(
            credential_name="langfuse-eu",
            credential_values={},
            credential_info={
                "credential_type": "logging",
                "description": "langfuse_otel",
                "access": {"global": True},
            },
        ),
        # scoped to one team / one org: assignable only within that scope.
        CredentialItem(
            credential_name="arize-ds",
            credential_values={},
            credential_info={
                "credential_type": "logging",
                "description": "arize",
                "access": {"teams": ["ds-team"], "orgs": ["ds-org"]},
            },
        ),
        # proxy-wide auto default: access.global makes it visible to every scope,
        # auto_enable makes it fire without being named.
        CredentialItem(
            credential_name="central-default",
            credential_values={},
            credential_info={
                "credential_type": "logging",
                "description": "arize",
                "auto_enable": True,
                "access": {"global": True},
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


def test_non_admin_with_no_flags_is_forbidden(_registry):
    """The headline deny: internal_user with no team/org admin context."""
    with pytest.raises(HTTPException) as exc:
        validate_logging_exporter_assignment(_ok(["langfuse-eu"]), _non_admin())
    assert exc.value.status_code == 403


# --- Scope checks: a non-proxy-admin may only name destinations granted to them -


# --- Shape / registry checks (run regardless of who's calling) --------------


def test_unknown_credential_rejected_for_admin(_registry):
    with pytest.raises(HTTPException) as exc:
        validate_logging_exporter_assignment(_ok(["does-not-exist"]), _admin())
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


# --- Veria F4: removal-via-omission ----------------------------------------
#
# Update endpoints replace stored metadata wholesale. A caller can wipe an
# admin-assigned `logging_exporters` by sending a `metadata` payload that
# omits the field. The validator must catch this when ``existing_metadata``
# is passed.


def test_removal_via_omission_blocked_for_non_admin(_registry):
    """A non-admin with no flags cannot wipe an admin-assigned exporter by
    submitting metadata without logging_exporters."""
    with pytest.raises(HTTPException) as exc:
        validate_logging_exporter_assignment(
            {"some_other_key": 1},  # no logging_exporters in the new payload
            _non_admin(),
            existing_metadata={"logging_exporters": ["langfuse-eu"]},
        )
    assert exc.value.status_code == 403


def test_removal_via_omission_allowed_for_proxy_admin(_registry):
    """Proxy admin may drop the exporter via omission."""
    validate_logging_exporter_assignment(
        {"some_other_key": 1},
        _admin(),
        existing_metadata={"logging_exporters": ["langfuse-eu"]},
    )


def test_explicit_empty_list_blocked_for_non_admin(_registry):
    """A non-admin submitting `logging_exporters: []` over a non-empty stored
    value is a removal write and must be gated."""
    with pytest.raises(HTTPException) as exc:
        validate_logging_exporter_assignment(
            {"logging_exporters": []},
            _non_admin(),
            existing_metadata={"logging_exporters": ["langfuse-eu"]},
        )
    assert exc.value.status_code == 403


def test_explicit_null_blocked_for_non_admin(_registry):
    """`logging_exporters: null` over a non-empty stored value is also a
    removal; the validator's shape check would reject it as non-list, but
    F4's authorization gate must fire first."""
    with pytest.raises(HTTPException) as exc:
        validate_logging_exporter_assignment(
            {"logging_exporters": None},
            _non_admin(),
            existing_metadata={"logging_exporters": ["langfuse-eu"]},
        )
    assert exc.value.status_code == 403


def test_unchanged_value_is_noop(_registry):
    """A metadata payload that re-sends the SAME logging_exporters value is
    a noop and skips the gate even for a non-admin -- there is no net change
    to authorize."""
    validate_logging_exporter_assignment(
        {"logging_exporters": ["langfuse-eu"]},
        _non_admin(),
        existing_metadata={"logging_exporters": ["langfuse-eu"]},
    )


def test_omitted_on_both_sides_is_noop(_registry):
    """A metadata update that doesn't touch logging_exporters on a row that
    never had one is a noop."""
    validate_logging_exporter_assignment(
        {"some_other_key": 1},
        _non_admin(),
        existing_metadata={"some_other_key": 0},
    )


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


def test_validate_credential_access_rejects_unknown_field():
    """Unknown access keys must be rejected at write time so a destination can never
    be stored in a shape the strict ``CredentialAccess`` read model later refuses to
    parse (which would 500 every subsequent PATCH)."""
    with pytest.raises(HTTPException) as exc:
        validate_credential_access({"access": {"global": True, "legacy_field": "x"}})
    assert exc.value.status_code == 400
    assert "legacy_field" in exc.value.detail["error"]


# --- validate_logging_exporter_field (the column-backed adapter) ------------
#
# The endpoints now pass a typed list off the request's ``logging_exporters``
# field instead of a metadata dict. The adapter must gate the same way, and the
# typed field's None-means-omitted semantics must not open a bypass.


def test_field_none_is_noop_for_non_admin(_registry):
    """A request that omits logging_exporters (None) must not require authorization."""
    validate_logging_exporter_field(None, _non_admin())


def test_field_set_by_non_admin_is_forbidden(_registry):
    with pytest.raises(HTTPException) as exc:
        validate_logging_exporter_field(["langfuse-eu"], _non_admin())
    assert exc.value.status_code == 403


def test_field_proxy_admin_can_assign(_registry):
    """Proxy admin may assign any registered logging destination."""
    validate_logging_exporter_field(["arize-ds"], _admin())


def test_field_empty_clear_over_existing_is_gated_for_non_admin(_registry):
    """Clearing an admin-assigned value ([] over a non-empty stored column) is a
    change and must be authorized; a non-admin cannot silently wipe it."""
    with pytest.raises(HTTPException) as exc:
        validate_logging_exporter_field(
            [],
            _non_admin(),
            existing_exporters=["langfuse-eu"],
        )
    assert exc.value.status_code == 403


def test_field_unchanged_value_is_noop(_registry):
    """Re-sending the same column value is a no-op even for a non-admin."""
    validate_logging_exporter_field(
        ["langfuse-eu"],
        _non_admin(),
        existing_exporters=["langfuse-eu"],
    )


