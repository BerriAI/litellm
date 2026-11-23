//! A key's team as the auth cache holds it, as far as the native host reads it. Each field is
//! taken apart by name, so a field added to the cached team does not compile here until it is
//! honored or named as one the native app does not serve.

use std::sync::Arc;

use litellm_redis_store::auth_cache::CachedTeam;

use super::key::Carried;

/// A team the native host read for a key, with what the host reads of it.
#[derive(Clone, Debug, PartialEq)]
pub struct Team {
    /// The team's `models` list, which the auth core decides model access on.
    models: Arc<[String]>,
}

impl Team {
    pub(crate) fn models(&self) -> &[String] {
        &self.models
    }
}

/// The native app honors a team's `models` list and nothing else of it. `Err` names the first
/// field that holds something Python enforces and the native app does not: the block, a budget, a
/// limit, an alias table, an access group, a permission, the organization above it or anything in
/// its metadata. The roster and the member defaults only reach a key with an owner, which the
/// owner's own step declines, and whether a request may change guardrails is asked of a body
/// `admit` never admits. The rest only describes the team or what it spent.
pub(super) fn read(team: &CachedTeam) -> Result<Team, &'static str> {
    let CachedTeam {
        team_alias: _,
        team_id: _,
        admins: _,
        members: _,
        members_with_roles: _,
        team_member_permissions: _,
        default_team_member_models: _,
        allow_team_guardrail_config: _,
        spend: _,
        model_spend: _,
        updated_at: _,
        created_at: _,
        last_refreshed_at: _,
        models,
        blocked,
        organization_id,
        metadata,
        tpm_limit,
        rpm_limit,
        tpd_limit,
        max_parallel_requests,
        max_budget,
        soft_budget,
        budget_duration,
        budget_limits,
        budget_reset_at,
        model_max_budget,
        router_settings,
        access_group_ids,
        policies,
        model_id,
        litellm_model_table,
        object_permission,
        object_permission_id,
        unread,
    } = team;
    let unread_models = super::key::unread_models(models);
    let unread = &unread.iter().any(|(_, value)| value.is_carried());
    let unserved = unread_models.then_some("models").or_else(|| {
        carried![
            blocked,
            organization_id,
            metadata,
            tpm_limit,
            rpm_limit,
            tpd_limit,
            max_parallel_requests,
            max_budget,
            soft_budget,
            budget_duration,
            budget_limits,
            budget_reset_at,
            model_max_budget,
            router_settings,
            access_group_ids,
            policies,
            model_id,
            litellm_model_table,
            object_permission,
            object_permission_id,
            unread,
        ]
    });
    match unserved {
        Some(field) => Err(field),
        None => Ok(Team {
            models: Arc::from(models.as_slice()),
        }),
    }
}

#[cfg(test)]
mod tests {
    use litellm_python_compat::json;
    use litellm_redis_store::FromValue;
    use rstest::rstest;
    use serde_json::{Value as Fields, json};

    use super::*;

    /// `LiteLLM_TeamTableCachedObj.model_dump(mode="json")` of a team `/team/new` made with a
    /// model list and nothing else.
    const CACHED: &str = r#"{
        "team_alias": "Team", "team_id": "team", "organization_id": null, "admins": [],
        "members": [], "members_with_roles": [{"user_id": "admin", "user_email": null, "role": "admin"}],
        "team_member_permissions": [], "metadata": {}, "tpm_limit": null, "rpm_limit": null,
        "tpd_limit": null, "max_budget": null, "soft_budget": null, "budget_duration": null,
        "budget_limits": null, "models": ["claude"], "blocked": false, "router_settings": {},
        "access_group_ids": [], "default_team_member_models": [], "spend": 0.0,
        "max_parallel_requests": null, "budget_reset_at": null, "model_id": null, "model_spend": {},
        "model_max_budget": {}, "policies": [], "allow_team_guardrail_config": false,
        "litellm_model_table": null, "object_permission": null, "object_permission_id": null,
        "updated_at": "2026-11-19T10:00:00.123000Z", "created_at": "2026-11-19T10:00:00.123000Z",
        "last_refreshed_at": 1763546400.5
    }"#;

    /// The cached team with `change` set on it.
    fn cached(change: Fields) -> CachedTeam {
        let mut entry: Fields = serde_json::from_str(CACHED).unwrap();
        entry
            .as_object_mut()
            .unwrap()
            .extend(change.as_object().unwrap().clone());
        CachedTeam::from_value(&json::loads(&entry.to_string()).unwrap()).unwrap()
    }

    #[rstest]
    fn a_team_python_cached_with_a_model_list_is_read_with_that_list() {
        let team = read(&cached(json!({}))).unwrap();

        assert_eq!(team.models(), ["claude".to_owned()]);
    }

    #[rstest]
    #[case::the_teams_models(json!({"models": ["all-team-models"]}))]
    #[case::a_field_this_build_does_not_name_left_empty(json!({"future_limit": null}))]
    #[case::a_guardrail_permission(json!({"allow_team_guardrail_config": true}))]
    #[case::member_defaults(json!({"default_team_member_models": ["gpt"]}))]
    fn a_field_the_native_app_honors_or_never_reads_is_no_reason_to_decline(
        #[case] change: Fields,
    ) {
        assert!(read(&cached(change)).is_ok());
    }

    #[rstest]
    #[case::a_block(json!({"blocked": true}), "blocked")]
    #[case::an_organization(json!({"organization_id": "org"}), "organization_id")]
    #[case::metadata(json!({"metadata": {"guardrails": ["pii"]}}), "metadata")]
    #[case::tpm_limit(json!({"tpm_limit": 0}), "tpm_limit")]
    #[case::rpm_limit(json!({"rpm_limit": 1}), "rpm_limit")]
    #[case::tpd_limit(json!({"tpd_limit": 1}), "tpd_limit")]
    #[case::max_parallel_requests(json!({"max_parallel_requests": 1}), "max_parallel_requests")]
    #[case::max_budget(json!({"max_budget": 0.0}), "max_budget")]
    #[case::soft_budget(json!({"soft_budget": 1.0}), "soft_budget")]
    #[case::budget_duration(json!({"budget_duration": "30d"}), "budget_duration")]
    #[case::budget_limits(
        json!({"budget_limits": [{"budget_duration": "1d", "max_budget": 1.0, "reset_at": null}]}),
        "budget_limits"
    )]
    #[case::budget_reset_at(json!({"budget_reset_at": "2100-01-01T00:00:00Z"}), "budget_reset_at")]
    #[case::model_max_budget(json!({"model_max_budget": {"claude": 1}}), "model_max_budget")]
    #[case::router_settings(json!({"router_settings": {"num_retries": 1}}), "router_settings")]
    #[case::access_group_ids(json!({"access_group_ids": ["group"]}), "access_group_ids")]
    #[case::policies(json!({"policies": ["policy"]}), "policies")]
    #[case::model_id(json!({"model_id": 1}), "model_id")]
    #[case::an_alias_table(
        json!({"litellm_model_table": {"model_aliases": {"gpt": "claude"}, "created_by": "a", "updated_by": "a"}}),
        "litellm_model_table"
    )]
    #[case::an_object_permission(
        json!({"object_permission": {"object_permission_id": "p"}}),
        "object_permission"
    )]
    #[case::object_permission_id(json!({"object_permission_id": "p"}), "object_permission_id")]
    #[case::a_provider_pattern(json!({"models": ["anthropic/*"]}), "models")]
    #[case::no_default_models(json!({"models": ["no-default-models"]}), "models")]
    #[case::a_field_this_build_does_not_name(json!({"future_limit": 1}), "unread")]
    fn a_team_field_the_native_app_does_not_honor_is_named(
        #[case] change: Fields,
        #[case] field: &str,
    ) {
        assert_eq!(read(&cached(change)).err(), Some(field));
    }
}
