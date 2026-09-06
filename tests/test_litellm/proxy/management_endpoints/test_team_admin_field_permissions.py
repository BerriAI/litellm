import pytest
from fastapi import HTTPException

from litellm.proxy._types import LiteLLM_ModelTable, LiteLLM_TeamTable, UpdateTeamRequest
from litellm.proxy.management_endpoints.team_admin_field_permissions import (
    TeamAdminEditAllowed,
    TeamAdminEditingDisabled,
    TeamAdminFieldNotPermitted,
    changed_team_fields,
    raise_for_team_admin_edit_verdict,
    resolve_team_admin_editable_fields,
    team_admin_edit_verdict,
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

    def test_changes_within_permitted_fields_are_allowed(self):
        data = UpdateTeamRequest(team_id="team-1", tpm_limit=6, team_alias="alpha")
        verdict = team_admin_edit_verdict(data, _team(team_alias="alpha"), frozenset({"tpm_limit"}))
        assert verdict == TeamAdminEditAllowed()

    def test_first_blocked_field_in_sorted_order_is_reported(self):
        data = UpdateTeamRequest(team_id="team-1", tpm_limit=6, rpm_limit=6, blocked=True)
        verdict = team_admin_edit_verdict(data, _team(), frozenset({"tpm_limit"}))
        assert verdict == TeamAdminFieldNotPermitted(field="blocked")


class TestRaiseForTeamAdminEditVerdict:
    def test_allowed_does_not_raise(self):
        assert raise_for_team_admin_edit_verdict(TeamAdminEditAllowed()) is None

    def test_disabled_is_a_403_pointing_at_the_proxy_admin(self):
        with pytest.raises(HTTPException) as exc:
            raise_for_team_admin_edit_verdict(TeamAdminEditingDisabled())
        assert exc.value.status_code == 403
        assert "cannot edit team settings" in exc.value.detail
        assert "Settings > UI > Team admin editable fields" in exc.value.detail

    def test_field_not_permitted_is_a_403_naming_the_field(self):
        with pytest.raises(HTTPException) as exc:
            raise_for_team_admin_edit_verdict(TeamAdminFieldNotPermitted(field="blocked"))
        assert exc.value.status_code == 403
        assert "'blocked'" in exc.value.detail
