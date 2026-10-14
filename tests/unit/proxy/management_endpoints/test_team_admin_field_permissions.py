from typing import Final

import pytest
from fastapi import HTTPException

from litellm.models.team import BudgetLimitEntry
from litellm.models.verification_token import LiteLLM_VerificationToken
from litellm.proxy._types import (
    LiteLLM_ModelTable,
    LiteLLM_TeamTable,
    LiteLLMKeyType,
    UpdateKeyRequest,
    UpdateTeamRequest,
)
from litellm.proxy.management_endpoints.team_admin_field_permissions import (
    SUPPORTED_TEAM_ADMIN_EDITABLE_TEAM_FIELDS,
    TeamAdminDeclaredMetadataKeyCleared,
    TeamAdminEditAllowed,
    TeamAdminEditingDisabled,
    TeamAdminFieldNotPermitted,
    TeamAdminKeyEditAllowed,
    TeamAdminMemberKeyEditingDisabled,
    TeamAdminMetadataKeyProxyControlled,
    TeamMemberDefaults,
    changed_key_fields,
    changed_team_fields,
    resolve_team_admin_editable_fields,
    team_admin_edit_verdict,
    team_admin_key_edit_verdict,
    team_admin_key_request_or_raise,
    team_admin_may_create_member_keys,
    team_admin_may_edit_member_key_budgets,
    team_admin_may_manage_projects,
    team_admin_may_raise_max_budget,
    team_admin_member_key_creation_or_raise,
    team_admin_member_key_creation_verdict,
    TeamAdminMemberKeyCreationAllowed,
    TeamAdminMemberKeyCreationDisabled,
    TeamAdminMemberKeyOwnerNotPermitted,
    TeamAdminMemberKeyTypeNotPermitted,
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

    def test_raise_max_budget_permission_is_not_a_team_field(self):
        configured = {"team_admin_editable_team_fields": ["raise_max_budget", "max_budget"]}
        assert resolve_team_admin_editable_fields(configured, frozenset({"max_budget"})) == frozenset({"max_budget"})


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

    def test_top_level_require_trace_id_change_is_attributed_to_the_folded_field(self):
        existing: Final = _team(metadata={"require_trace_id": True})
        data: Final = UpdateTeamRequest(team_id="team-1", require_trace_id=False)

        assert changed_team_fields(data, existing) == frozenset({"require_trace_id"})
        assert team_admin_edit_verdict(data, existing, frozenset({"tpm_limit"})) == TeamAdminMetadataKeyProxyControlled(
            key="require_trace_id"
        )
        assert changed_team_fields(UpdateTeamRequest(team_id="team-1", require_trace_id=True), existing) == frozenset()

    def test_metadata_replacement_cannot_drop_stored_require_trace_id(self):
        data: Final = UpdateTeamRequest(team_id="team-1", metadata={})
        existing: Final = _team(metadata={"require_trace_id": True})

        assert changed_team_fields(data, existing) == frozenset({"require_trace_id"})
        assert team_admin_edit_verdict(data, existing, frozenset({"tpm_limit"})) == TeamAdminMetadataKeyProxyControlled(
            key="require_trace_id"
        )

    def test_resending_stored_require_trace_id_is_not_a_change(self):
        data: Final = UpdateTeamRequest(team_id="team-1", metadata={"require_trace_id": True})

        assert changed_team_fields(data, _team(metadata={"require_trace_id": True})) == frozenset()

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
        data = UpdateTeamRequest(team_id="team-1", team_member_key_duration="30d")
        assert changed_team_fields(data, _team()) == frozenset({"team_member_key_duration"})


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
        verdict = team_admin_edit_verdict(
            data, existing, frozenset({"guardrails"}), proxy_controlled_metadata_keys=frozenset()
        )
        assert isinstance(verdict, TeamAdminEditAllowed)
        assert verdict.request.model_dump(exclude_unset=True) == {
            "team_id": "team-1",
            "metadata": {"guardrails": ["b"]},
        }

    def test_first_blocked_field_in_sorted_order_is_reported(self):
        data = UpdateTeamRequest(team_id="team-1", tpm_limit=6, rpm_limit=6, blocked=True)
        verdict = team_admin_edit_verdict(data, _team(), frozenset({"tpm_limit"}))
        assert verdict == TeamAdminFieldNotPermitted(field="blocked")


def _permitted(*fields: str) -> frozenset[str]:
    return resolve_team_admin_editable_fields(
        {"team_admin_editable_team_fields": list(fields)}, SUPPORTED_TEAM_ADMIN_EDITABLE_TEAM_FIELDS
    )


_STORED_MEMBER_DEFAULTS: Final = TeamMemberDefaults(
    team_member_budget=10.0, team_member_budget_duration="30d", team_member_tpm_limit=100, team_member_rpm_limit=5
)

_MEMBER_DEFAULT_EDITS: Final = (
    pytest.param("team_member_budget", 25.0, 10.0, id="team_member_budget"),
    pytest.param("team_member_budget_duration", "7d", "30d", id="team_member_budget_duration"),
    pytest.param("team_member_tpm_limit", 200, 100, id="team_member_tpm_limit"),
    pytest.param("team_member_rpm_limit", 9, 5, id="team_member_rpm_limit"),
)


class TestTeamMemberDefaultEdits:
    @pytest.mark.parametrize("field,new,_stored", _MEMBER_DEFAULT_EDITS)
    def test_enabled_field_change_is_allowed(self, field: str, new: object, _stored: object):
        data: Final = UpdateTeamRequest.model_validate({"team_id": "team-1", field: new})
        verdict: Final = team_admin_edit_verdict(data, _team(), _permitted(field), _STORED_MEMBER_DEFAULTS)
        assert isinstance(verdict, TeamAdminEditAllowed)
        assert verdict.request.model_dump(exclude_unset=True) == {"team_id": "team-1", field: new}

    @pytest.mark.parametrize("field,new,_stored", _MEMBER_DEFAULT_EDITS)
    def test_change_without_the_field_enabled_names_it(self, field: str, new: object, _stored: object):
        data: Final = UpdateTeamRequest.model_validate({"team_id": "team-1", field: new})
        verdict: Final = team_admin_edit_verdict(data, _team(), _permitted("tpm_limit"), _STORED_MEMBER_DEFAULTS)
        assert verdict == TeamAdminFieldNotPermitted(field=field)

    @pytest.mark.parametrize("field,_new,stored", _MEMBER_DEFAULT_EDITS)
    def test_resending_the_stored_value_is_dropped_without_the_field_enabled(
        self, field: str, _new: object, stored: object
    ):
        data: Final = UpdateTeamRequest.model_validate({"team_id": "team-1", field: stored, "tpm_limit": 6})
        verdict: Final = team_admin_edit_verdict(
            data, _team(tpm_limit=5), _permitted("tpm_limit"), _STORED_MEMBER_DEFAULTS
        )
        assert isinstance(verdict, TeamAdminEditAllowed)
        assert verdict.request.model_dump(exclude_unset=True) == {"team_id": "team-1", "tpm_limit": 6}

    def test_nulls_resent_for_a_team_without_member_defaults_are_dropped(self):
        data: Final = UpdateTeamRequest(
            team_id="team-1",
            team_member_budget=None,
            team_member_budget_duration=None,
            team_member_tpm_limit=None,
            team_member_rpm_limit=None,
            tpm_limit=6,
        )
        verdict: Final = team_admin_edit_verdict(data, _team(tpm_limit=5), _permitted("tpm_limit"), None)
        assert isinstance(verdict, TeamAdminEditAllowed)
        assert verdict.request.model_dump(exclude_unset=True) == {"team_id": "team-1", "tpm_limit": 6}

    def test_first_member_budget_keeps_an_explicit_never_resets_duration(self):
        data: Final = UpdateTeamRequest(team_id="team-1", team_member_budget=5.0, team_member_budget_duration=None)
        permitted: Final = _permitted("team_member_budget", "team_member_budget_duration")
        verdict: Final = team_admin_edit_verdict(data, _team(budget_duration="7d"), permitted, None)
        assert isinstance(verdict, TeamAdminEditAllowed)
        assert verdict.request.model_dump(exclude_unset=True) == {
            "team_id": "team-1",
            "team_member_budget": 5.0,
            "team_member_budget_duration": None,
        }

    def test_first_member_budget_drops_a_duration_the_team_admin_may_not_set(self):
        data: Final = UpdateTeamRequest(team_id="team-1", team_member_budget=5.0, team_member_budget_duration=None)
        verdict: Final = team_admin_edit_verdict(
            data, _team(budget_duration="7d"), _permitted("team_member_budget"), None
        )
        assert isinstance(verdict, TeamAdminEditAllowed)
        assert verdict.request.model_dump(exclude_unset=True) == {"team_id": "team-1", "team_member_budget": 5.0}

    def test_existing_member_budget_drops_a_resent_duration_next_to_a_changed_budget(self):
        data: Final = UpdateTeamRequest(team_id="team-1", team_member_budget=5.0, team_member_budget_duration="7d")
        stored: Final = TeamMemberDefaults(team_member_budget=1.0, team_member_budget_duration="7d")
        permitted: Final = _permitted("team_member_budget", "team_member_budget_duration")
        verdict: Final = team_admin_edit_verdict(data, _team(), permitted, stored)
        assert isinstance(verdict, TeamAdminEditAllowed)
        assert verdict.request.model_dump(exclude_unset=True) == {"team_id": "team-1", "team_member_budget": 5.0}


class TestSoftBudgetEdits:
    def test_enabled_soft_budget_change_is_allowed(self):
        data: Final = UpdateTeamRequest(team_id="team-1", soft_budget=20.0)
        verdict: Final = team_admin_edit_verdict(data, _team(soft_budget=10.0), _permitted("soft_budget"))
        assert isinstance(verdict, TeamAdminEditAllowed)
        assert verdict.request.model_dump(exclude_unset=True) == {"team_id": "team-1", "soft_budget": 20.0}

    def test_soft_budget_change_without_it_enabled_names_it(self):
        data: Final = UpdateTeamRequest(team_id="team-1", soft_budget=20.0)
        verdict: Final = team_admin_edit_verdict(data, _team(soft_budget=10.0), _permitted("max_budget"))
        assert verdict == TeamAdminFieldNotPermitted(field="soft_budget")

    def test_unchanged_soft_budget_resend_is_dropped_when_not_permitted(self):
        data: Final = UpdateTeamRequest(team_id="team-1", tpm_limit=6, soft_budget=10)
        verdict: Final = team_admin_edit_verdict(data, _team(soft_budget=10.0), _permitted("tpm_limit"))
        assert isinstance(verdict, TeamAdminEditAllowed)
        assert verdict.request.model_dump(exclude_unset=True) == {"team_id": "team-1", "tpm_limit": 6}


_PROXY_READ_TEAM_METADATA_KEYS: Final = (
    "opted_out_global_guardrails",
    "disable_global_guardrails",
    "guardrails",
    "policies",
    "tags",
    "spend_logs_metadata",
    "model_config",
    "callback_settings",
    "logging",
    "priority",
    "zguard_policy_id",
    "team_member_key_duration",
    "secret_manager_settings",
    "model_rpm_limit",
    "model_tpm_limit",
    "mcp_rpm_limit",
    "default_estimated_output_tokens",
    "default_estimated_output_tokens_per_model",
    "allowed_vector_store_indexes",
    "enforced_file_expires_after",
    "enforced_batch_output_expires_after",
    "require_trace_id",
    "allowed_passthrough_routes",
    "denied_passthrough_routes",
    "allowed_tools",
    "allow_client_mock_response",
    "allow_client_message_redaction_opt_out",
    "allow_client_pricing_override",
    "allow_litellm_changed_by_header",
    "otel_service_name_override",
    "otel_service_name",
    "soft_budget_alerting_emails",
    "team_member_max_budget_alert_emails",
    "batch_enqueued_token_limit",
    "max_batch_file_records",
    "max_batch_file_uploads_per_day",
    "max_file_downloads_per_minute",
    "scim_managed",
    "scim_data",
    "phoenix_project_name_override",
    "phoenix_project_name",
    "opik",
)

_STORED_METADATA: Final = {"cost_center": "eng", "region": "eu", "priority": "high", "guardrails": ["pii"]}


def _metadata_verdict(stored: dict[str, object], sent: dict[str, object], declared: frozenset[str] = frozenset()):
    return team_admin_edit_verdict(
        UpdateTeamRequest(team_id="team-1", metadata=sent),
        _team(metadata=stored),
        _permitted("metadata"),
        declared_metadata_keys=declared,
    )


class TestTeamMetadataEdits:
    def test_metadata_change_without_metadata_enabled_names_metadata(self):
        verdict: Final = team_admin_edit_verdict(
            UpdateTeamRequest(team_id="team-1", metadata={**_STORED_METADATA, "project": "x"}),
            _team(metadata=_STORED_METADATA),
            _permitted("tpm_limit"),
        )
        assert verdict == TeamAdminFieldNotPermitted(field="metadata")

    @pytest.mark.parametrize(
        "sent",
        [
            pytest.param({**_STORED_METADATA, "project": "x"}, id="add"),
            pytest.param({**_STORED_METADATA, "cost_center": "sales"}, id="edit"),
            pytest.param({"region": "eu", "priority": "high", "guardrails": ["pii"]}, id="delete"),
        ],
    )
    def test_ordinary_key_add_edit_and_delete_are_allowed(self, sent: dict[str, object]):
        verdict: Final = _metadata_verdict(_STORED_METADATA, sent)
        assert isinstance(verdict, TeamAdminEditAllowed)
        assert verdict.request.metadata == sent

    @pytest.mark.parametrize("key", _PROXY_READ_TEAM_METADATA_KEYS)
    def test_adding_a_key_the_proxy_reads_names_it(self, key: str):
        verdict: Final = _metadata_verdict(_STORED_METADATA, {**_STORED_METADATA, key: "set-by-team-admin"})
        assert verdict == TeamAdminMetadataKeyProxyControlled(key=key)

    @pytest.mark.parametrize("key", ["opted_out_global_guardrails", "callback_settings", "priority", "guardrails"])
    def test_changing_or_removing_a_stored_proxy_controlled_key_names_it(self, key: str):
        stored: Final = {**_STORED_METADATA, key: ["stored"]}
        changed: Final = _metadata_verdict(stored, {**stored, key: ["changed"]})
        removed: Final = _metadata_verdict(stored, {name: value for name, value in stored.items() if name != key})
        assert changed == TeamAdminMetadataKeyProxyControlled(key=key)
        assert removed == TeamAdminMetadataKeyProxyControlled(key=key)

    def test_proxy_controlled_field_sent_top_level_names_it(self):
        verdict: Final = team_admin_edit_verdict(
            UpdateTeamRequest(team_id="team-1", tags=["team-admin"]), _team(metadata={}), _permitted("metadata")
        )
        assert verdict == TeamAdminMetadataKeyProxyControlled(key="tags")

    def test_unchanged_resend_of_the_full_stored_metadata_is_not_an_edit(self):
        verdict: Final = team_admin_edit_verdict(
            UpdateTeamRequest(team_id="team-1", tpm_limit=6, metadata=dict(_STORED_METADATA)),
            _team(metadata=_STORED_METADATA),
            _permitted("tpm_limit"),
        )
        assert isinstance(verdict, TeamAdminEditAllowed)
        assert verdict.request.model_dump(exclude_unset=True) == {"team_id": "team-1", "tpm_limit": 6}

    def test_system_managed_key_left_out_is_not_a_removal(self):
        stored: Final = {**_STORED_METADATA, "team_member_budget_id": "budget-1"}
        verdict: Final = _metadata_verdict(stored, {**_STORED_METADATA, "project": "x"})
        assert isinstance(verdict, TeamAdminEditAllowed)

    def test_system_managed_key_sent_with_another_value_names_it(self):
        stored: Final = {**_STORED_METADATA, "team_member_budget_id": "budget-1"}
        verdict: Final = _metadata_verdict(stored, {**stored, "team_member_budget_id": "budget-2"})
        assert verdict == TeamAdminMetadataKeyProxyControlled(key="team_member_budget_id")

    def test_declared_key_value_edit_is_allowed(self):
        verdict: Final = _metadata_verdict(
            _STORED_METADATA, {**_STORED_METADATA, "cost_center": "sales"}, frozenset({"cost_center"})
        )
        assert isinstance(verdict, TeamAdminEditAllowed)
        assert verdict.request.metadata == {**_STORED_METADATA, "cost_center": "sales"}

    @pytest.mark.parametrize(
        "sent",
        [
            pytest.param({"region": "eu", "priority": "high", "guardrails": ["pii"]}, id="removed"),
            pytest.param({**_STORED_METADATA, "cost_center": None}, id="null"),
            pytest.param({**_STORED_METADATA, "cost_center": ""}, id="empty"),
            pytest.param({**_STORED_METADATA, "cost_center": "   "}, id="blank"),
        ],
    )
    def test_clearing_a_stored_declared_key_names_it(self, sent: dict[str, object]):
        verdict: Final = _metadata_verdict(_STORED_METADATA, sent, frozenset({"cost_center"}))
        assert verdict == TeamAdminDeclaredMetadataKeyCleared(key="cost_center")

    def test_declared_key_never_set_is_not_a_removal_when_absent(self):
        verdict: Final = _metadata_verdict(
            _STORED_METADATA, {**_STORED_METADATA, "project": "x"}, frozenset({"department"})
        )
        assert isinstance(verdict, TeamAdminEditAllowed)

    @pytest.mark.parametrize(
        "stored,sent",
        [
            pytest.param(True, 1, id="true-to-1"),
            pytest.param(1, True, id="1-to-true"),
            pytest.param({"a": 0}, {"a": False}, id="nested-0-to-false"),
        ],
    )
    def test_retyping_a_proxy_controlled_value_names_it(self, stored: object, sent: object):
        verdict: Final = _metadata_verdict(
            {"allow_client_mock_response": stored, "a": 1}, {"allow_client_mock_response": sent, "a": 2}
        )
        assert verdict == TeamAdminMetadataKeyProxyControlled(key="allow_client_mock_response")

    @pytest.mark.parametrize(
        "stored,sent",
        [
            pytest.param({"a": 1}, {"a": 2, "callback_settings": None}, id="add-as-null"),
            pytest.param({"a": 1, "callback_settings": None}, {"a": 2}, id="remove-stored-null"),
        ],
    )
    def test_a_null_valued_proxy_controlled_key_appearing_or_disappearing_names_it(
        self, stored: dict[str, object], sent: dict[str, object]
    ):
        assert _metadata_verdict(stored, sent) == TeamAdminMetadataKeyProxyControlled(key="callback_settings")

    @pytest.mark.parametrize(
        "stored,sent",
        [
            pytest.param(5.0, 5, id="whole-number-as-int"),
            pytest.param({"tiers": [1.0, {"b": 2.0}]}, {"tiers": [1, {"b": 2}]}, id="nested-whole-numbers"),
            pytest.param({"x": 1, "y": {"p": "a", "q": "b"}}, {"y": {"q": "b", "p": "a"}, "x": 1}, id="key-order"),
        ],
    )
    def test_resending_an_equal_proxy_controlled_value_is_not_a_change(self, stored: object, sent: object):
        verdict: Final = _metadata_verdict({"priority": stored, "a": 1}, {"priority": sent, "a": 2})
        assert isinstance(verdict, TeamAdminEditAllowed)

    def test_null_metadata_on_a_team_with_member_defaults_names_the_system_key(self):
        verdict: Final = team_admin_edit_verdict(
            UpdateTeamRequest(team_id="team-1", metadata=None),
            _team(metadata={"region": "eu", "team_member_budget_id": "budget-1"}),
            _permitted("metadata"),
        )
        assert verdict == TeamAdminMetadataKeyProxyControlled(key="team_member_budget_id")

    def test_proxy_controlled_rule_wins_over_declared(self):
        verdict: Final = _metadata_verdict(
            _STORED_METADATA, {**_STORED_METADATA, "priority": "low"}, frozenset({"priority"})
        )
        assert verdict == TeamAdminMetadataKeyProxyControlled(key="priority")


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

    @pytest.mark.parametrize(
        "verdict",
        [TeamAdminMetadataKeyProxyControlled(key="priority"), TeamAdminDeclaredMetadataKeyCleared(key="priority")],
    )
    def test_metadata_key_verdicts_are_a_403_naming_the_key(self, verdict):
        with pytest.raises(HTTPException) as exc:
            team_admin_request_or_raise(verdict)
        assert exc.value.status_code == 403
        assert "'priority'" in exc.value.detail

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


class TestTeamAdminMayCreateMemberKeys:
    def test_missing_setting_denies(self):
        assert team_admin_may_create_member_keys({}) is False

    def test_other_permissions_do_not_grant(self):
        configured = {"team_admin_editable_team_fields": ["tpm_limit", "projects", "member_key_budgets"]}
        assert team_admin_may_create_member_keys(configured) is False

    def test_create_member_keys_entry_grants(self):
        configured = {"team_admin_editable_team_fields": ["create_member_keys"]}
        assert team_admin_may_create_member_keys(configured) is True

    @pytest.mark.parametrize("raw", ["create_member_keys", 7, [1, 2]])
    def test_malformed_setting_denies(self, raw):
        assert team_admin_may_create_member_keys({"team_admin_editable_team_fields": raw}) is False


_MEMBER_KEYS_ON = {"team_admin_editable_team_fields": ["create_member_keys"]}


class TestTeamAdminMemberKeyCreationVerdict:
    @pytest.mark.parametrize("owner_role", ["internal_user", "internal_user_viewer"])
    @pytest.mark.parametrize("key_type", [None, LiteLLMKeyType.DEFAULT, LiteLLMKeyType.LLM_API])
    def test_an_ordinary_member_and_an_llm_key_are_allowed(self, owner_role, key_type):
        verdict = team_admin_member_key_creation_verdict(_MEMBER_KEYS_ON, key_type=key_type, owner_role=owner_role)
        assert verdict == TeamAdminMemberKeyCreationAllowed()
        assert team_admin_member_key_creation_or_raise(verdict) is None

    def test_disabled_is_a_403_naming_the_permission_and_where_to_enable_it(self):
        verdict = team_admin_member_key_creation_verdict(
            {"team_admin_editable_team_fields": ["member_key_budgets"]}, key_type=None, owner_role="internal_user"
        )
        assert verdict == TeamAdminMemberKeyCreationDisabled()
        with pytest.raises(HTTPException) as exc:
            team_admin_member_key_creation_or_raise(verdict)
        assert exc.value.status_code == 403
        assert "'create_member_keys'" in exc.value.detail
        assert "Settings > UI > Team admin editable fields" in exc.value.detail

    @pytest.mark.parametrize("owner_role", ["proxy_admin", "proxy_admin_viewer", "org_admin", "team", "customer", None])
    def test_an_owner_who_is_not_an_ordinary_internal_user_is_refused(self, owner_role):
        """A key authenticates with its owner's role, so a privileged owner would hand that role to the team admin."""
        verdict = team_admin_member_key_creation_verdict(_MEMBER_KEYS_ON, key_type=None, owner_role=owner_role)
        assert verdict == TeamAdminMemberKeyOwnerNotPermitted(owner_role=owner_role)
        with pytest.raises(HTTPException) as exc:
            team_admin_member_key_creation_or_raise(verdict)
        assert exc.value.status_code == 403
        assert f"'{owner_role}'" in exc.value.detail

    @pytest.mark.parametrize("key_type", [LiteLLMKeyType.MANAGEMENT, LiteLLMKeyType.READ_ONLY])
    def test_a_key_type_that_reaches_beyond_llm_routes_is_refused(self, key_type):
        verdict = team_admin_member_key_creation_verdict(_MEMBER_KEYS_ON, key_type=key_type, owner_role="internal_user")
        assert verdict == TeamAdminMemberKeyTypeNotPermitted(key_type=key_type)
        with pytest.raises(HTTPException) as exc:
            team_admin_member_key_creation_or_raise(verdict)
        assert exc.value.status_code == 403
        assert f"'{key_type.value}'" in exc.value.detail


class TestTeamAdminMayRaiseMaxBudget:
    def test_missing_setting_denies(self):
        assert team_admin_may_raise_max_budget({}) is False

    def test_max_budget_alone_denies(self):
        configured = {"team_admin_editable_team_fields": ["tpm_limit", "max_budget", "projects"]}
        assert team_admin_may_raise_max_budget(configured) is False

    def test_raise_without_max_budget_denies(self):
        """A config file can list the entry on its own; it is inert until max_budget is granted too."""
        configured = {"team_admin_editable_team_fields": ["raise_max_budget", "tpm_limit"]}
        assert team_admin_may_raise_max_budget(configured) is False

    def test_max_budget_with_raise_grants(self):
        configured = {"team_admin_editable_team_fields": ["raise_max_budget", "max_budget"]}
        assert team_admin_may_raise_max_budget(configured) is True

    @pytest.mark.parametrize("raw", ["raise_max_budget", 7, [1, 2], {"max_budget": True}])
    def test_malformed_setting_denies(self, raw: object):
        assert team_admin_may_raise_max_budget({"team_admin_editable_team_fields": raw}) is False


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

    def test_spend_echo_is_blocked_even_when_unchanged(self):
        data = UpdateKeyRequest(key="sk-1", spend=4.5, max_budget=0)
        verdict = team_admin_key_edit_verdict(data, _key(spend=4.5, max_budget=10.0), enabled=True)
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
