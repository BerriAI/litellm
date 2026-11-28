//! Reads a caller's team from the auth cache into a core record. Each entry is taken apart by
//! name, so a field added to it does not compile here until it is honored or declined. Nothing
//! here charges or limits a team, so a row that holds a limit, a budget or anything else the
//! native app does not apply is declined, naming the field.

use litellm_gateway_auth_core::{
    authz::ModelList, identity::jwt::Team, record::TeamRecord,
    routes::passthrough::PassthroughGrants,
};
use litellm_redis_store::auth_cache::TeamEntry;

use super::carried::{Carried, carried, newer};

/// The team's record, or the first field it holds that the native app does not apply.
pub(super) fn team(team: &TeamEntry) -> Result<Team<TeamRecord>, &'static str> {
    let TeamEntry {
        team_alias,
        team_id,
        organization_id,
        admins: _,
        members: _,
        members_with_roles,
        team_member_permissions: _,
        metadata,
        tpm_limit,
        rpm_limit,
        tpd_limit,
        max_budget,
        soft_budget,
        budget_duration,
        budget_limits,
        models,
        blocked,
        router_settings,
        access_group_ids,
        default_team_member_models,
        spend: _,
        max_parallel_requests,
        budget_reset_at,
        model_id,
        model_spend: _,
        model_max_budget,
        policies,
        allow_team_guardrail_config,
        litellm_model_table,
        object_permission,
        object_permission_id,
        updated_at: _,
        created_at: _,
        last_refreshed_at: _,
        unread,
    } = team;
    let unread = &newer(unread);
    let unread_models = ModelList::read(models) == ModelList::Unread;
    let held = unread_models.then_some("models").or_else(|| {
        carried![
            organization_id,
            metadata,
            tpm_limit,
            rpm_limit,
            tpd_limit,
            max_budget,
            soft_budget,
            budget_duration,
            budget_limits,
            blocked,
            router_settings,
            access_group_ids,
            default_team_member_models,
            max_parallel_requests,
            budget_reset_at,
            model_id,
            model_max_budget,
            policies,
            allow_team_guardrail_config,
            litellm_model_table,
            object_permission,
            object_permission_id,
            unread,
        ]
    });
    if let Some(field) = held {
        return Err(field);
    }
    Ok(Team {
        team_id: team_id.clone(),
        models: Some(models.clone()),
        passthrough: PassthroughGrants::default(),
        members: members_with_roles
            .iter()
            .filter_map(|member| member.user_id.clone())
            .collect(),
        row: TeamRecord {
            team_id: team_id.clone(),
            team_alias: team_alias.clone(),
            models: models.clone(),
            ..TeamRecord::default()
        },
    })
}

#[cfg(test)]
mod tests {
    use litellm_python_compat::json;
    use litellm_redis_store::FromValue;
    use rstest::rstest;

    use super::*;

    fn cached(entry: &str) -> TeamEntry {
        TeamEntry::from_value(&json::loads(entry).unwrap()).unwrap()
    }

    #[rstest]
    #[case::that_holds_something(r#"{"team_id": "squad", "future_limit": 1}"#, Some("unread"))]
    #[case::left_empty(r#"{"team_id": "squad", "future_limit": null}"#, None)]
    fn a_cached_team_declines_over_a_field_this_build_does_not_name_only_when_it_holds_something(
        #[case] entry: &str,
        #[case] declined: Option<&str>,
    ) {
        assert_eq!(team(&cached(entry)).err(), declined);
    }
}
