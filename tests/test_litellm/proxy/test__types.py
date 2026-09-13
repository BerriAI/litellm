import json

import pytest
from pydantic import ValidationError

from litellm.proxy._types import (
    ROLES_WITHIN_ORG,
    GenerateKeyRequest,
    KeyRequest,
    LiteLLM_AuditLogs,
    LiteLLM_TeamMembership,
    LitellmUserRoles,
    OrganizationMemberUpdateRequest,
    ResetSpendRequest,
    UpdateKeyRequest,
    UpdateUserRequest,
    UserAPIKeyAuth,
)

SERVER_ONLY_MARKERS = (
    "mcp_admitted_user_subject",
    "mcp_source_team_rpm_limits",
    "mcp_session_resource_server_id",
    "via_virtual_key",
)


@pytest.mark.parametrize("marker", SERVER_ONLY_MARKERS)
def test_a_caller_cannot_forge_a_server_only_marker_through_the_constructor(marker):
    auth = UserAPIKeyAuth(**{marker: "forged-by-caller"})

    assert getattr(auth, marker) != "forged-by-caller"


@pytest.mark.parametrize("marker", SERVER_ONLY_MARKERS)
def test_a_caller_cannot_forge_a_server_only_marker_through_model_validate(marker):
    auth = UserAPIKeyAuth.model_validate({marker: "forged-by-caller"})

    assert getattr(auth, marker) != "forged-by-caller"


@pytest.mark.parametrize("marker", SERVER_ONLY_MARKERS)
def test_the_server_sets_a_marker_by_assignment_after_construction(marker):
    auth = UserAPIKeyAuth()

    setattr(auth, marker, "set-by-the-server")

    assert getattr(auth, marker) == "set-by-the-server"


def test_a_virtual_key_is_hashed_out_of_the_auth_object():
    raw_key = "sk-1234567890abcdefghij"

    auth = UserAPIKeyAuth(api_key=raw_key)

    assert auth.api_key != raw_key
    assert auth.token == auth.api_key


def test_a_bearer_prefixed_key_hashes_the_same_as_the_bare_key():
    raw_key = "sk-1234567890abcdefghij"

    assert UserAPIKeyAuth(api_key=f"Bearer {raw_key}").token == UserAPIKeyAuth(api_key=raw_key).token


def test_an_absent_api_key_leaves_the_token_unset():
    auth = UserAPIKeyAuth()

    assert auth.api_key is None
    assert auth.token is None


AUDIENCE_CASES = (
    ("https://litellm.example.com", False, True),
    (None, True, True),
    (None, False, False),
    ("https://litellm.example.com", True, False),
)


@pytest.mark.parametrize(("audience", "disable_audience_validation", "is_accepted"), AUDIENCE_CASES)
def test_a_jwt_issuer_must_name_an_audience_or_opt_out_of_one_but_never_both(
    audience, disable_audience_validation, is_accepted
):
    from litellm.proxy._types import JWTIssuerConfig

    fields = {
        "issuer": "https://idp.example.com",
        "audience": audience,
        "disable_audience_validation": disable_audience_validation,
    }

    if is_accepted:
        config = JWTIssuerConfig(**fields)
        assert config.audience == audience
        assert config.disable_audience_validation is disable_audience_validation
        return

    with pytest.raises(ValidationError):
        JWTIssuerConfig(**fields)


@pytest.mark.parametrize("sent", (True, False))
def test_a_boolean_spend_reset_is_refused_rather_than_read_as_a_number(sent):
    with pytest.raises(ValidationError):
        ResetSpendRequest(reset_to=sent)


@pytest.mark.parametrize(("sent", "expected"), ((0, 0.0), (12, 12.0), (4.25, 4.25), ("7.5", 7.5)))
def test_a_numeric_spend_reset_is_kept_as_that_number(sent, expected):
    assert ResetSpendRequest(reset_to=sent).reset_to == expected


TEMP_BUDGET_CASES = (
    (None, None, True),
    (10.0, "2026-01-01T00:00:00", True),
    (10.0, None, False),
    (None, "2026-01-01T00:00:00", False),
)


@pytest.mark.parametrize(("increase", "expiry", "is_accepted"), TEMP_BUDGET_CASES)
def test_a_temporary_budget_needs_both_an_amount_and_an_expiry(increase, expiry, is_accepted):
    fields = {"key": "sk-abc", "temp_budget_increase": increase, "temp_budget_expiry": expiry}

    if is_accepted:
        assert UpdateKeyRequest(**fields).temp_budget_increase == increase
        return

    with pytest.raises(ValidationError):
        UpdateKeyRequest(**fields)


KEY_IDENTIFIER_CASES = (
    ({"key": "sk-abc"}, True),
    ({"key_alias": "my-alias"}, True),
    ({"key": "sk-abc", "key_alias": "my-alias"}, True),
    ({}, False),
)


@pytest.mark.parametrize(("fields", "is_accepted"), KEY_IDENTIFIER_CASES)
def test_a_key_update_must_say_which_key_it_updates(fields, is_accepted):
    if is_accepted:
        assert UpdateKeyRequest(**fields) is not None
        return

    with pytest.raises(ValidationError):
        UpdateKeyRequest(**fields)


KEY_LOOKUP_CASES = (
    ({"keys": ["sk-abc"]}, True),
    ({"key_aliases": ["my-alias"]}, True),
    ({}, False),
    ({"keys": []}, False),
    ({"keys": [], "key_aliases": []}, False),
)


@pytest.mark.parametrize(("fields", "is_accepted"), KEY_LOOKUP_CASES)
def test_a_key_lookup_naming_nothing_is_refused_rather_than_matching_everything(fields, is_accepted):
    if is_accepted:
        assert KeyRequest(**fields) is not None
        return

    with pytest.raises(ValidationError):
        KeyRequest(**fields)


@pytest.mark.parametrize("role", ROLES_WITHIN_ORG)
def test_an_organization_member_may_hold_a_role_that_exists_within_an_organization(role):
    request = OrganizationMemberUpdateRequest(organization_id="org-1", user_id="user-1", role=role)

    assert request.role == role


ROLES_OUTSIDE_ORG = tuple(role for role in LitellmUserRoles if role not in ROLES_WITHIN_ORG)


@pytest.mark.parametrize("role", ROLES_OUTSIDE_ORG)
def test_an_organization_member_cannot_be_given_a_role_that_lives_outside_the_organization(role):
    with pytest.raises(ValidationError):
        OrganizationMemberUpdateRequest(organization_id="org-1", user_id="user-1", role=role)


def test_an_empty_max_budget_from_a_form_post_reads_as_no_budget_not_as_zero():
    assert GenerateKeyRequest(max_budget="").max_budget is None


@pytest.mark.parametrize("sent", (0, 0.0, 25.5))
def test_a_max_budget_that_was_actually_sent_is_kept(sent):
    assert GenerateKeyRequest(max_budget=sent).max_budget == sent


USER_IDENTIFIER_CASES = (
    ({"user_id": "user-1"}, True),
    ({"user_email": "user@example.com"}, True),
    ({"user_id": "user-1", "user_email": "user@example.com"}, True),
    ({}, False),
)


@pytest.mark.parametrize(("fields", "is_accepted"), USER_IDENTIFIER_CASES)
def test_a_user_update_must_say_which_user_it_updates(fields, is_accepted):
    if is_accepted:
        assert UpdateUserRequest(**fields) is not None
        return

    with pytest.raises(ValidationError):
        UpdateUserRequest(**fields)


def _audit_log(**overrides) -> LiteLLM_AuditLogs:
    fields = {
        "id": "audit-1",
        "updated_at": "2026-01-01T00:00:00",
        "changed_by": "user-1",
        "action": "updated",
        "table_name": "LiteLLM_VerificationToken",
        "object_id": "key-1",
        **overrides,
    }
    return LiteLLM_AuditLogs(**fields)


SECRET = "sk-verysecretvalue1234567890"
SECRET_MASKED = "sk-v********************7890"


@pytest.mark.parametrize("field", ("before_value", "updated_values"))
def test_an_audit_log_does_not_store_the_key_it_recorded_a_change_to(field):
    log = _audit_log(**{field: json.dumps({"key": SECRET})})

    assert json.loads(getattr(log, field)) == {"key": SECRET_MASKED}


@pytest.mark.parametrize("field", ("before_value", "updated_values"))
def test_an_audit_log_keeps_the_non_secret_fields_it_recorded(field):
    sent = {"key": SECRET, "max_budget": 50, "models": ["gpt-4o"]}

    log = _audit_log(**{field: json.dumps(sent)})

    assert json.loads(getattr(log, field)) == {
        "key": SECRET_MASKED,
        "max_budget": 50,
        "models": ["gpt-4o"],
    }


@pytest.mark.parametrize("field", ("before_value", "updated_values"))
def test_an_audit_log_leaves_a_change_it_has_no_record_of_alone(field):
    assert getattr(_audit_log(**{field: None}), field) is None


@pytest.mark.parametrize(("sent", "expected"), ((123, "123"), (None, None), ("user-1", "user-1")))
def test_an_audit_log_records_who_made_the_change_as_text(sent, expected):
    assert _audit_log(changed_by=sent).changed_by == expected


def test_team_membership_budget_table_optional_no_crash():
    data = {
        "user_id": "test-user",
        "team_id": "test-team",
        "budget_id": None,
    }
    result = LiteLLM_TeamMembership.model_validate(data)
    assert result.litellm_budget_table is None


def test_team_membership_budget_table_present_still_works():
    data = {
        "user_id": "test-user",
        "team_id": "test-team",
        "budget_id": "some-budget-id",
        "litellm_budget_table": None,
    }
    result = LiteLLM_TeamMembership.model_validate(data)
    assert result.litellm_budget_table is None
