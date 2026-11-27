//! Reads a caller's team from the auth cache into a core record. Each entry is taken apart by
//! name, so a field added to it does not compile here until it is copied onto the record, read
//! for nothing or declined. A column that holds a structure no record field has a type for yet is
//! declined here, naming the field, and [`unserved`] names the first copied column the native app
//! does not serve yet.

use litellm_gateway_auth_core::{
    authz::ModelList,
    identity::jwt::Team,
    record::{Budget, MinuteLimits, TeamRecord},
    routes::passthrough::PassthroughGrants,
};
use litellm_redis_store::auth_cache::TeamEntry;

use super::carried::{Carried, carried, newer};

/// The team's record, or the first field it holds that no record field has a type for yet.
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
        spend,
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
            metadata,
            budget_limits,
            router_settings,
            budget_reset_at,
            model_id,
            model_max_budget,
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
            organization_id: organization_id.clone(),
            limits: MinuteLimits {
                rpm: *rpm_limit,
                tpm: *tpm_limit,
            },
            tpd_limit: *tpd_limit,
            max_parallel_requests: *max_parallel_requests,
            budget: Budget {
                max: *max_budget,
                spend: spend.unwrap_or_default(),
            },
            soft_budget: *soft_budget,
            budget_duration: budget_duration.clone(),
            blocked: *blocked,
            access_group_ids: access_group_ids.clone().unwrap_or_default(),
            default_team_member_models: default_team_member_models.clone().unwrap_or_default(),
            policies: policies.clone().unwrap_or_default(),
            allow_team_guardrail_config: allow_team_guardrail_config.unwrap_or_default(),
            ..TeamRecord::default()
        },
    })
}

/// The first column of `team` that holds something nothing in the native app reads yet, which
/// sends its caller to Python. The record is taken apart by name, so a column added to it does
/// not compile here until it is named as read or listed. Whoever makes the native app honor a
/// column takes it off the list in the same change.
pub(crate) fn unserved(team: &TeamRecord) -> Option<&'static str> {
    let TeamRecord {
        team_id: _,
        team_alias: _,
        models: _,
        organization_id,
        limits: MinuteLimits {
            rpm: rpm_limit,
            tpm: tpm_limit,
        },
        tpd_limit,
        max_parallel_requests,
        budget: Budget {
            max: max_budget,
            spend: _,
        },
        soft_budget,
        budget_duration,
        blocked,
        access_group_ids,
        default_team_member_models,
        policies,
        allow_team_guardrail_config,
        member_max_budget,
    } = team;
    carried![
        organization_id,
        tpm_limit,
        rpm_limit,
        tpd_limit,
        max_parallel_requests,
        max_budget,
        soft_budget,
        budget_duration,
        blocked,
        access_group_ids,
        default_team_member_models,
        policies,
        allow_team_guardrail_config,
        member_max_budget,
    ]
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

    #[test]
    fn a_teams_organization_is_copied_and_named_unserved() {
        let team = team(&cached(r#"{"team_id": "squad", "organization_id": "org"}"#)).unwrap();

        assert_eq!(
            (team.row.organization_id.as_deref(), unserved(&team.row)),
            (Some("org"), Some("organization_id"))
        );
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
