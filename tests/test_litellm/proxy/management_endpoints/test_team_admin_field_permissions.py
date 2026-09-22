import pytest
from fastapi import HTTPException

from litellm.models.team import BudgetLimitEntry
from litellm.models.verification_token import LiteLLM_VerificationToken
from litellm.proxy._types import LiteLLM_ModelTable, LiteLLM_TeamTable, UpdateKeyRequest, UpdateTeamRequest
from litellm.proxy.management_endpoints.team_admin_field_permissions import (
    TeamAdminEditAllowed,
    TeamAdminEditingDisabled,
    TeamAdminFieldNotPermitted,
    TeamAdminKeyEditAllowed,
    TeamAdminMemberKeyEditingDisabled,
    changed_key_fields,
    changed_team_fields,
    resolve_team_admin_editable_fields,
    team_admin_edit_verdict,
    team_admin_key_edit_verdict,
    team_admin_key_request_or_raise,
    team_admin_may_edit_member_key_budgets,
    team_admin_may_manage_projects,
    team_admin_request_or_raise,
)

_SUPPORTED = frozenset({"tpm_limit", "rpm_limit", "team_alias"})


def _team(**overrides):
    return LiteLLM_TeamTable(team_id="team-1", **overrides)


class TestResolveTeamAdminEditableFields:
    def test_missing_setting_means_nothing_editable(self):
        assert resolve_team_admin_editable_fields({}, _SUPPORTED) == frozenset()

    def test_keeps_only_supported_names(self):
        configured = {"team_admin_editable_team_fields": ["tpm_limit", "blocked", "organization_id"]}
        assert resolve_team_admin_editable_fields(configured, _SUPPORTED) == frozenset({"tpm_limit"})

    @pytest.mark.parametrize("raw", ["tpm_limit", 7, {"tpm_limit": True}, [1, 2]])
    def test_malformed_setting_fails_closed(self, raw):
        assert resolve_team_admin_editable_fields({"team_admin_editable_team_fields": raw}, _SUPPORTED) == frozenset()

    def test_projects_permission_is_not_a_team_field(self):
        configured = {"team_admin_editable_team_fields": ["projects", "tpm_limit"]}
        assert resolve_team_admin_editable_fields(configured, _SUPPORTED) == frozenset({"tpm_limit"})


class TestTeamAdminMayManageProjects:
    def test_missing_setting_denies(self):
        assert team_admin_may_manage_projects({}) is False

    def test_team_fields_alone_do_not_grant_projects(self):
        assert team_admin_may_manage_projects({"team_admin_editable_team_fields": ["tpm_limit", "max_budget"]}) is False

    def test_projects_entry_grants(self):
        assert team_admin_may_manage_projects({"team_admin_editable_team_fields": ["max_budget", "projects"]}) is True

    @pytest.mark.parametrize("raw", ["projects", 7, [1, 2]])
    def test_malformed_setting_denies(self, raw):
        assert team_admin_may_manage_projects({"team_admin_editable_team_fields": raw}) is False


class TestChangedTeamFields:
    def test_team_id_alone_changes_nothing(self):
        assert changed_team_fields(UpdateTeamRequest(team_id="team-1"), _team()) == frozenset()

    def test_column_echoing_stored_value_is_not_a_change(self):
        data = UpdateTeamRequest(team_id="team-1", tpm_limit=5, team_alias="alpha", max_budget=None)
        assert changed_team_fields(data, _team(tpm_limit=5, team_alias="alpha")) == frozenset()

    def test_column_with_different_value_is_a_change(self):
        data = UpdateTeamRequest(team_id="team-1", tpm_limit=6, team_alias="alpha")
        assert changed_team_fields(data, _team(tpm_limit=5, team_alias="alpha")) == frozenset({"tpm_limit"})

    def test_explicit_null_clearing_a_stored_column_is_a_change(self):
        data = UpdateTeamRequest(team_id="team-1", max_budget=None)
        assert changed_team_fields(data, _team(max_budget=30.0)) == frozenset({"max_budget"})

    def test_folded_field_sent_top_level_is_named_not_metadata(self):
        data = UpdateTeamRequest(team_id="team-1", guardrails=["b"])
        assert changed_team_fields(data, _team(metadata={"guardrails": ["a"]})) == frozenset({"guardrails"})

    def test_folded_field_sent_inside_metadata_is_named_not_metadata(self):
        data = UpdateTeamRequest(team_id="team-1", metadata={"guardrails": ["b"]})
        assert changed_team_fields(data, _team(metadata={"guardrails": ["a"]})) == frozenset({"guardrails"})

    def test_custom_metadata_key_change_is_attributed_to_metadata(self):
        data = UpdateTeamRequest(team_id="team-1", metadata={"guardrails": ["a"], "cost_center": "b"})
        existing = _team(metadata={"guardrails": ["a"], "cost_center": "a"})
        assert changed_team_fields(data, existing) == frozenset({"metadata"})

    def test_metadata_echo_with_top_level_override_only_names_the_override(self):
        data = UpdateTeamRequest(team_id="team-1", guardrails=["b"], metadata={"guardrails": ["a"], "cost_center": "a"})
        existing = _team(metadata={"guardrails": ["a"], "cost_center": "a"})
        assert changed_team_fields(data, existing) == frozenset({"guardrails"})

    def test_dropping_a_stored_key_from_submitted_metadata_is_a_change(self):
        data = UpdateTeamRequest(team_id="team-1", metadata={"cost_center": "a"})
        existing = _team(metadata={"cost_center": "a", "tags": ["x"], "logging": [{"callback": "langfuse"}]})
        assert changed_team_fields(data, existing) == frozenset({"tags", "logging"})

    def test_server_managed_metadata_key_is_ignored(self):
        data = UpdateTeamRequest(team_id="team-1", metadata={"cost_center": "a"})
        existing = _team(metadata={"cost_center": "a", "team_member_budget_id": "budget-1"})
        assert changed_team_fields(data, existing) == frozenset()

    def test_model_aliases_compare_against_the_model_table(self):
        table = LiteLLM_ModelTable(model_aliases='{"fast": "gpt-4o-mini"}', created_by="a", updated_by="a")
        same = UpdateTeamRequest(team_id="team-1", model_aliases={"fast": "gpt-4o-mini"})
        different = UpdateTeamRequest(team_id="team-1", model_aliases={"fast": "gpt-4o"})
        assert changed_team_fields(same, _team(litellm_model_table=table)) == frozenset()
        assert changed_team_fields(different, _team(litellm_model_table=table)) == frozenset({"model_aliases"})

    def test_empty_model_aliases_against_no_model_table_is_not_a_change(self):
        assert changed_team_fields(UpdateTeamRequest(team_id="team-1", model_aliases={}), _team()) == frozenset()

    def test_field_without_a_stored_counterpart_counts_as_changed_when_sent(self):
        data = UpdateTeamRequest(team_id="team-1", team_member_budget=10.0)
        assert changed_team_fields(data, _team()) == frozenset({"team_member_budget"})


class TestTeamAdminEditVerdict:
    def test_no_permitted_fields_disables_editing_even_for_a_no_op(self):
        verdict = team_admin_edit_verdict(UpdateTeamRequest(team_id="team-1"), _team(), frozenset())
        assert verdict == TeamAdminEditingDisabled()

    def test_allowed_request_keeps_only_the_changed_fields(self):
        data = UpdateTeamRequest(team_id="team-1", tpm_limit=6, team_alias="alpha", budget_duration="30d")
        existing = _team(team_alias="alpha", budget_duration="30d")
        verdict = team_admin_edit_verdict(data, existing, frozenset({"tpm_limit"}))
        assert isinstance(verdict, TeamAdminEditAllowed)
        assert verdict.request.model_dump(exclude_unset=True) == {"team_id": "team-1", "tpm_limit": 6}

    def test_permitted_field_changed_inside_metadata_keeps_the_metadata(self):
        data = UpdateTeamRequest(team_id="team-1", metadata={"guardrails": ["b"]}, team_alias="alpha")
        existing = _team(team_alias="alpha", metadata={"guardrails": ["a"]})
        verdict = team_admin_edit_verdict(data, existing, frozenset({"guardrails"}))
        assert isinstance(verdict, TeamAdminEditAllowed)
        assert verdict.request.model_dump(exclude_unset=True) == {
            "team_id": "team-1",
            "metadata": {"guardrails": ["b"]},
        }

    def test_first_blocked_field_in_sorted_order_is_reported(self):
        data = UpdateTeamRequest(team_id="team-1", tpm_limit=6, rpm_limit=6, blocked=True)
        verdict = team_admin_edit_verdict(data, _team(), frozenset({"tpm_limit"}))
        assert verdict == TeamAdminFieldNotPermitted(field="blocked")


class TestTeamAdminRequestOrRaise:
    def test_allowed_hands_back_its_request(self):
        request = UpdateTeamRequest(team_id="team-1", tpm_limit=6)
        assert team_admin_request_or_raise(TeamAdminEditAllowed(request=request)) is request

    def test_disabled_is_a_403_pointing_at_the_proxy_admin(self):
        with pytest.raises(HTTPException) as exc:
            team_admin_request_or_raise(TeamAdminEditingDisabled())
        assert exc.value.status_code == 403
        assert "cannot edit team settings" in exc.value.detail
        assert "Settings > UI > Team admin editable fields" in exc.value.detail

    def test_field_not_permitted_is_a_403_naming_the_field(self):
        with pytest.raises(HTTPException) as exc:
            team_admin_request_or_raise(TeamAdminFieldNotPermitted(field="blocked"))
        assert exc.value.status_code == 403
        assert "'blocked'" in exc.value.detail


def _key(**overrides):
    return LiteLLM_VerificationToken(token="hashed", **overrides)


class TestTeamAdminMayEditMemberKeyBudgets:
    def test_missing_setting_denies(self):
        assert team_admin_may_edit_member_key_budgets({}) is False

    def test_team_fields_alone_do_not_grant(self):
        configured = {"team_admin_editable_team_fields": ["tpm_limit", "max_budget", "projects"]}
        assert team_admin_may_edit_member_key_budgets(configured) is False

    def test_member_key_budgets_entry_grants(self):
        configured = {"team_admin_editable_team_fields": ["member_key_budgets"]}
        assert team_admin_may_edit_member_key_budgets(configured) is True

    @pytest.mark.parametrize("raw", ["member_key_budgets", 7, [1, 2]])
    def test_malformed_setting_denies(self, raw):
        assert team_admin_may_edit_member_key_budgets({"team_admin_editable_team_fields": raw}) is False


class TestChangedKeyFields:
    def test_key_alone_changes_nothing(self):
        assert changed_key_fields(UpdateKeyRequest(key="sk-1"), _key()) == frozenset()

    def test_columns_echoing_stored_values_are_not_a_change(self):
        data = UpdateKeyRequest(key="sk-1", max_budget=10.0, models=["m"], tpm_limit=5)
        existing = _key(max_budget=10.0, models=["m"], tpm_limit=5)
        assert changed_key_fields(data, existing) == frozenset()

    def test_column_with_different_value_is_a_change(self):
        data = UpdateKeyRequest(key="sk-1", max_budget=0)
        assert changed_key_fields(data, _key(max_budget=10.0)) == frozenset({"max_budget"})

    def test_metadata_folded_field_echo_is_not_a_change(self):
        data = UpdateKeyRequest(key="sk-1", tag_rpm_limit={"fast": 3})
        existing = _key(metadata={"tag_rpm_limit": {"fast": 3}})
        assert changed_key_fields(data, existing) == frozenset()

    def test_metadata_folded_field_difference_is_named_not_metadata(self):
        data = UpdateKeyRequest(key="sk-1", tag_rpm_limit={"fast": 4})
        existing = _key(metadata={"tag_rpm_limit": {"fast": 3}})
        assert changed_key_fields(data, existing) == frozenset({"tag_rpm_limit"})

    def test_budget_limits_echo_ignores_order_and_reset_at(self):
        windows = [
            {"budget_duration": "1d", "max_budget": 5.0, "reset_at": "2030-01-01T00:00:00"},
            {"budget_duration": "7d", "max_budget": 50.0, "reset_at": "2030-01-07T00:00:00"},
        ]
        data = UpdateKeyRequest(
            key="sk-1",
            budget_limits=[
                BudgetLimitEntry(budget_duration="7d", max_budget=50.0),
                BudgetLimitEntry(budget_duration="1d", max_budget=5.0),
            ],
        )
        assert changed_key_fields(data, _key(budget_limits=windows)) == frozenset()

    def test_budget_limits_difference_is_a_change(self):
        data = UpdateKeyRequest(key="sk-1", budget_limits=[BudgetLimitEntry(budget_duration="1d", max_budget=9.0)])
        existing = _key(budget_limits=[{"budget_duration": "1d", "max_budget": 5.0, "reset_at": "2030-01-01"}])
        assert changed_key_fields(data, existing) == frozenset({"budget_limits"})

    def test_explicit_null_clearing_a_stored_column_is_a_change(self):
        data = UpdateKeyRequest(key="sk-1", budget_duration=None)
        assert changed_key_fields(data, _key(budget_duration="30d")) == frozenset({"budget_duration"})

    def test_field_without_a_stored_counterpart_counts_as_changed_when_sent(self):
        data = UpdateKeyRequest(key="sk-1", duration="1h")
        assert changed_key_fields(data, _key()) == frozenset({"duration"})


class TestTeamAdminKeyEditVerdict:
    def test_disabled_even_for_a_no_op(self):
        verdict = team_admin_key_edit_verdict(UpdateKeyRequest(key="sk-1"), _key(), enabled=False)
        assert verdict == TeamAdminMemberKeyEditingDisabled()

    def test_budget_only_change_is_allowed(self):
        data = UpdateKeyRequest(key="sk-1", max_budget=0, budget_duration="30d")
        verdict = team_admin_key_edit_verdict(data, _key(max_budget=10.0), enabled=True)
        assert verdict == TeamAdminKeyEditAllowed(changed=frozenset({"max_budget", "budget_duration"}))

    def test_key_alias_change_is_blocked_and_named(self):
        data = UpdateKeyRequest(key="sk-1", key_alias="renamed")
        verdict = team_admin_key_edit_verdict(data, _key(key_alias="member"), enabled=True)
        assert verdict == TeamAdminFieldNotPermitted(field="key_alias")

    def test_spend_is_blocked(self):
        data = UpdateKeyRequest(key="sk-1", spend=0)
        verdict = team_admin_key_edit_verdict(data, _key(spend=3.5), enabled=True)
        assert verdict == TeamAdminFieldNotPermitted(field="spend")

    def test_budget_plus_non_budget_names_the_non_budget_field(self):
        data = UpdateKeyRequest(key="sk-1", max_budget=0, key_alias="renamed")
        verdict = team_admin_key_edit_verdict(data, _key(max_budget=10.0, key_alias="member"), enabled=True)
        assert verdict == TeamAdminFieldNotPermitted(field="key_alias")


class TestTeamAdminKeyRequestOrRaise:
    def test_allowed_returns_none(self):
        verdict = TeamAdminKeyEditAllowed(changed=frozenset({"max_budget"}))
        assert team_admin_key_request_or_raise(verdict) is None

    def test_disabled_is_a_403_pointing_at_member_key_budgets(self):
        with pytest.raises(HTTPException) as exc:
            team_admin_key_request_or_raise(TeamAdminMemberKeyEditingDisabled())
        assert exc.value.status_code == 403
        assert "member_key_budgets" in exc.value.detail
        assert "Settings > UI > Team admin editable fields" in exc.value.detail

    def test_field_not_permitted_is_a_403_naming_the_field(self):
        with pytest.raises(HTTPException) as exc:
            team_admin_key_request_or_raise(TeamAdminFieldNotPermitted(field="key_alias"))
        assert exc.value.status_code == 403
        assert "'key_alias'" in exc.value.detail
