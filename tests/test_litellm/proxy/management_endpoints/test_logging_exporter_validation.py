"""Validation for admin-owned logging-exporter assignment on key/team/org."""

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
    validate_key_logging_exporter_assignment,
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
            credential_info={"custom_llm_provider": "openai"},  # a provider credential
        ),
    ]
    try:
        yield
    finally:
        litellm.credential_list = original


def _admin():
    return UserAPIKeyAuth(api_key="k", user_role=LitellmUserRoles.PROXY_ADMIN)


def _member():
    return UserAPIKeyAuth(api_key="k", user_role=LitellmUserRoles.INTERNAL_USER)


def test_admin_with_known_logging_credential_is_allowed(_registry):
    validate_logging_exporter_assignment(
        {"logging_exporters": ["langfuse-eu"]}, _admin()
    )


def test_noop_when_assignment_absent(_registry):
    # an update that does not touch logging_exporters is never gated/validated
    validate_logging_exporter_assignment({"some_other_key": 1}, _member())
    validate_logging_exporter_assignment(None, _member())


def test_non_admin_is_forbidden(_registry):
    with pytest.raises(HTTPException) as exc:
        validate_logging_exporter_assignment(
            {"logging_exporters": ["langfuse-eu"]}, _member()
        )
    assert exc.value.status_code == 403


def test_unknown_credential_is_rejected(_registry):
    with pytest.raises(HTTPException) as exc:
        validate_logging_exporter_assignment(
            {"logging_exporters": ["does-not-exist"]}, _admin()
        )
    assert exc.value.status_code == 400


def test_provider_credential_is_not_a_valid_logging_exporter(_registry):
    # openai-key exists but is a provider credential, not a logging destination
    with pytest.raises(HTTPException) as exc:
        validate_logging_exporter_assignment(
            {"logging_exporters": ["openai-key"]}, _admin()
        )
    assert exc.value.status_code == 400


def test_non_list_is_rejected(_registry):
    with pytest.raises(HTTPException) as exc:
        validate_logging_exporter_assignment(
            {"logging_exporters": "langfuse-eu"}, _admin()
        )
    assert exc.value.status_code == 400


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


# --- key-scoped validator (LIT-3850 follow-up) ------------------------------


def test_key_validator_team_admin_of_key_team_passes(_registry):
    """team-admin of the key's team may write metadata.logging_exporters."""
    validate_key_logging_exporter_assignment(
        metadata={"logging_exporters": ["langfuse-eu"]},
        user_api_key_dict=_member(),
        is_team_admin_of_key_team=True,
    )


def test_key_validator_proxy_admin_always_passes(_registry):
    validate_key_logging_exporter_assignment(
        metadata={"logging_exporters": ["langfuse-eu"]},
        user_api_key_dict=_admin(),
        is_team_admin_of_key_team=False,
    )


def test_key_validator_non_team_admin_is_forbidden(_registry):
    """A non-admin who isn't team-admin of the key's team is rejected."""
    with pytest.raises(HTTPException) as exc:
        validate_key_logging_exporter_assignment(
            metadata={"logging_exporters": ["langfuse-eu"]},
            user_api_key_dict=_member(),
            is_team_admin_of_key_team=False,
        )
    assert exc.value.status_code == 403


def test_key_validator_personal_key_rejects_non_admin(_registry):
    """A personal key (no team) means is_team_admin_of_key_team=False; only admin passes."""
    with pytest.raises(HTTPException) as exc:
        validate_key_logging_exporter_assignment(
            metadata={"logging_exporters": ["langfuse-eu"]},
            user_api_key_dict=_member(),
            is_team_admin_of_key_team=False,
        )
    assert exc.value.status_code == 403


def test_key_validator_unknown_credential_rejected_even_for_team_admin(_registry):
    with pytest.raises(HTTPException) as exc:
        validate_key_logging_exporter_assignment(
            metadata={"logging_exporters": ["does-not-exist"]},
            user_api_key_dict=_member(),
            is_team_admin_of_key_team=True,
        )
    assert exc.value.status_code == 400


def test_key_validator_noop_when_absent(_registry):
    validate_key_logging_exporter_assignment(
        metadata={"other": 1},
        user_api_key_dict=_member(),
        is_team_admin_of_key_team=False,
    )
    validate_key_logging_exporter_assignment(
        metadata=None,
        user_api_key_dict=_member(),
        is_team_admin_of_key_team=False,
    )
