//! Reads a caller's team, user and their team membership, organization and project from the auth
//! cache into core records. Each entry is taken apart by name, so a field added to it does not
//! compile here until it is copied onto the record, read for nothing or declined. A column that
//! holds a structure no record field has a type for yet is declined here, naming the field, and
//! [`unserved`], [`unserved_user`], [`unserved_membership`], [`unserved_org`] and
//! [`unserved_project`] name the first copied column the native app does not serve yet.

use litellm_gateway_auth_core::{
    AffiliationRead,
    authz::ModelList,
    identity::jwt::{Org, Team, User},
    record::{
        Budget, MembershipRecord, MinuteLimits, OrgRecord, ProjectRecord, TeamRecord, UserRecord,
    },
    routes::passthrough::PassthroughGrants,
};
use litellm_python_compat::Value;
use litellm_redis_store::auth_cache::{
    Bag, BudgetEntry, OrganizationEntry, ProjectEntry, TeamEntry, TeamMembershipEntry, UserEntry,
};

use super::{
    carried::{Carried, carried, newer},
    metadata::Metadata,
};

/// The `metadata` key SCIM sets to `False` on a user it deactivated.
const SCIM_ACTIVE: &str = "scim_active";

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
    let Some(Metadata {
        model_rpm_limit,
        model_tpm_limit,
        priority,
    }) = metadata
        .as_ref()
        .map_or(Some(Metadata::default()), Metadata::read)
    else {
        return Err("metadata");
    };
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
            model_rpm_limit,
            model_tpm_limit,
            priority,
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
        model_rpm_limit,
        model_tpm_limit,
        priority,
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
        model_rpm_limit,
        model_tpm_limit,
        priority,
        member_max_budget,
    ]
}

/// The user's record, or the first field it holds that the native gateway cannot handle yet.
pub(super) fn user(user: &UserEntry) -> Result<User<UserRecord>, &'static str> {
    let UserEntry {
        user_id,
        user_alias,
        team_id,
        sso_user_id: _,
        organization_id,
        object_permission_id,
        password_reset_required: _,
        last_breach_check_at: _,
        teams,
        user_role,
        max_budget,
        rollover_max_budget,
        rollover_credit,
        budget_limits,
        spend,
        user_email,
        models,
        metadata,
        max_parallel_requests,
        tpm_limit,
        rpm_limit,
        budget_duration,
        budget_reset_at,
        allowed_cache_controls,
        policies,
        model_spend: _,
        model_max_budget,
        created_at: _,
        updated_at: _,
        organization_memberships,
        object_permission,
        unread,
    } = user;
    let unread = &newer(unread);
    let deactivated = scim_deactivated(metadata.as_ref()).ok_or("metadata")?;
    let held = carried![
        object_permission_id,
        budget_limits,
        budget_reset_at,
        model_max_budget,
        organization_memberships,
        object_permission,
        unread,
    ];
    if let Some(field) = held {
        return Err(field);
    }
    Ok(User {
        user_id: user_id.clone(),
        user_email: user_email.clone(),
        user_role: user_role.clone(),
        teams: teams.clone(),
        deactivated,
        row: UserRecord {
            user_id: user_id.clone(),
            user_alias: user_alias.clone(),
            user_email: user_email.clone(),
            user_role: user_role.clone(),
            teams: teams.clone(),
            team_id: team_id.clone(),
            organization_id: organization_id.clone(),
            models: models.clone(),
            limits: MinuteLimits {
                rpm: *rpm_limit,
                tpm: *tpm_limit,
            },
            max_parallel_requests: *max_parallel_requests,
            budget: Budget {
                max: *max_budget,
                spend: *spend,
            },
            rollover_max_budget: rollover_max_budget.0,
            rollover_credit: rollover_credit.0,
            budget_duration: budget_duration.clone(),
            allowed_cache_controls: allowed_cache_controls.clone(),
            policies: policies.clone(),
        },
    })
}

/// Whether SCIM deactivated the user, as `metadata.get("scim_active") is False` reads it. `None`
/// when the metadata holds anything else.
fn scim_deactivated(metadata: Option<&Bag>) -> Option<bool> {
    let entries = metadata.map(Bag::entries).unwrap_or_default();
    match entries {
        [] => Some(false),
        [(Value::Str(key), active)] if key == SCIM_ACTIVE => {
            Some(matches!(active, Value::Bool(false)))
        }
        _ => None,
    }
}

/// The first column of `user` that holds something nothing in the native app reads yet, which
/// sends its caller to Python, as [`unserved`] names a team's. The user's email, role and teams
/// say who it is, and the auth core reads them from the user a JWT names, so they are not listed.
pub(crate) fn unserved_user(user: &UserRecord) -> Option<&'static str> {
    let UserRecord {
        user_id: _,
        user_alias: _,
        user_email: _,
        user_role: _,
        teams: _,
        team_id,
        organization_id,
        models,
        limits: MinuteLimits {
            rpm: rpm_limit,
            tpm: tpm_limit,
        },
        max_parallel_requests,
        budget: Budget {
            max: max_budget,
            spend: _,
        },
        rollover_max_budget,
        rollover_credit,
        budget_duration,
        allowed_cache_controls,
        policies,
    } = user;
    carried![
        team_id,
        organization_id,
        models,
        tpm_limit,
        rpm_limit,
        max_parallel_requests,
        max_budget,
        rollover_max_budget,
        rollover_credit,
        budget_duration,
        allowed_cache_controls,
        policies,
    ]
}

/// The membership's record, or the first field it or its budget row holds that no record field has
/// a type for yet. A JWT's read declines a membership with any budget of its own.
pub(super) fn membership(
    membership: &TeamMembershipEntry,
    read: AffiliationRead,
) -> Result<MembershipRecord, &'static str> {
    let TeamMembershipEntry {
        user_id,
        team_id,
        budget_id,
        spend,
        total_spend: _,
        rollover_credit,
        litellm_budget_table,
        unread,
    } = membership;
    let unread = &newer(unread);
    let rollover_credit = &rollover_credit.0;
    let has_budget = &litellm_budget_table.is_some();
    if read == AffiliationRead::Jwt
        && let Some(field) = carried![budget_id, rollover_credit]
            .or(has_budget.then_some("litellm_budget_table"))
            .or_else(|| carried![unread])
    {
        return Err(field);
    }
    if let Some(field) = carried![unread] {
        return Err(field);
    }
    let spend = spend.unwrap_or_default();
    let BudgetRow {
        limits,
        budget,
        soft_budget,
        tpd_limit,
        max_parallel_requests,
        budget_duration,
    } = match litellm_budget_table {
        Some(row) => budget_row(&row.budget, spend)?,
        None => BudgetRow {
            budget: Budget { max: None, spend },
            ..BudgetRow::default()
        },
    };
    Ok(MembershipRecord {
        user_id: user_id.clone(),
        team_id: team_id.clone(),
        limits,
        budget,
        rollover_credit: *rollover_credit,
        soft_budget,
        tpd_limit,
        max_parallel_requests,
        budget_duration,
    })
}

/// The first column of `membership` that holds something nothing in the native app reads yet,
/// which sends its caller to Python, taken apart by name as [`unserved`] takes a team.
pub(crate) fn unserved_membership(membership: &MembershipRecord) -> Option<&'static str> {
    let MembershipRecord {
        user_id: _,
        team_id: _,
        limits: MinuteLimits {
            rpm: rpm_limit,
            tpm: tpm_limit,
        },
        budget: Budget {
            max: max_budget,
            spend: _,
        },
        rollover_credit,
        soft_budget,
        tpd_limit,
        max_parallel_requests,
        budget_duration,
    } = membership;
    carried![
        rollover_credit,
        max_budget,
        soft_budget,
        rpm_limit,
        tpm_limit,
        tpd_limit,
        max_parallel_requests,
        budget_duration
    ]
}

/// The organization's record, or the first field it or its budget row holds that no record field
/// has a type for yet.
pub(super) fn org(org: &OrganizationEntry) -> Result<Org<OrgRecord>, &'static str> {
    let OrganizationEntry {
        organization_id,
        organization_alias,
        budget_id: _,
        spend,
        metadata,
        models,
        model_spend: _,
        created_by: _,
        updated_by: _,
        users: _,
        litellm_budget_table,
        object_permission,
        object_permission_id,
        unread,
    } = org;
    let unread = &newer(unread);
    if let Some(field) = carried![object_permission, object_permission_id, metadata, unread] {
        return Err(field);
    }
    let organization_id = organization_id.as_ref().ok_or("organization_id")?;
    let limits = litellm_budget_table
        .as_ref()
        .ok_or("litellm_budget_table")?;
    let BudgetRow {
        limits,
        budget,
        soft_budget,
        tpd_limit,
        max_parallel_requests,
        budget_duration,
    } = budget_row(limits, *spend)?;
    Ok(Org {
        organization_id: organization_id.clone(),
        row: OrgRecord {
            organization_id: organization_id.clone(),
            organization_alias: organization_alias.clone(),
            models: models.clone(),
            limits,
            budget,
            soft_budget,
            tpd_limit,
            max_parallel_requests,
            budget_duration,
        },
    })
}

/// A budget row's plain values, which the record of whatever the row budgets copies.
#[derive(Default)]
struct BudgetRow {
    limits: MinuteLimits,
    budget: Budget,
    soft_budget: Option<f64>,
    tpd_limit: Option<i64>,
    max_parallel_requests: Option<i64>,
    budget_duration: Option<String>,
}

/// The budget row's plain values with `spend` spent against it, or the first field it holds that
/// no record field has a type for yet.
fn budget_row(entry: &BudgetEntry, spend: f64) -> Result<BudgetRow, &'static str> {
    let BudgetEntry {
        budget_id: _,
        soft_budget,
        max_budget,
        rollover_max_budget,
        max_parallel_requests,
        tpm_limit,
        rpm_limit,
        tpd_limit,
        model_max_budget,
        model_tpm_limit,
        model_rpm_limit,
        budget_duration,
        allowed_models,
        temp_budget_increase,
        temp_budget_expiry,
        unread,
    } = entry;
    let rollover_max_budget = &rollover_max_budget.0;
    let model_tpm_limit = &model_tpm_limit
        .as_ref()
        .is_some_and(|map| !map.0.is_empty());
    let model_rpm_limit = &model_rpm_limit
        .as_ref()
        .is_some_and(|map| !map.0.is_empty());
    let unread = &newer(unread);
    if let Some(field) = carried![
        rollover_max_budget,
        model_max_budget,
        model_tpm_limit,
        model_rpm_limit,
        allowed_models,
        temp_budget_increase,
        temp_budget_expiry,
        unread,
    ] {
        return Err(field);
    }
    Ok(BudgetRow {
        budget: Budget {
            max: *max_budget,
            spend,
        },
        soft_budget: *soft_budget,
        limits: MinuteLimits {
            rpm: *rpm_limit,
            tpm: *tpm_limit,
        },
        tpd_limit: *tpd_limit,
        max_parallel_requests: *max_parallel_requests,
        budget_duration: budget_duration.clone(),
    })
}

/// The first column of `org` that holds something nothing in the native app reads yet, which
/// sends its caller to Python, taken apart by name as [`unserved`] takes a team.
pub(crate) fn unserved_org(org: &OrgRecord) -> Option<&'static str> {
    let OrgRecord {
        organization_id: _,
        organization_alias: _,
        models,
        budget: Budget {
            max: max_budget,
            spend: _,
        },
        soft_budget,
        limits: MinuteLimits {
            rpm: rpm_limit,
            tpm: tpm_limit,
        },
        tpd_limit,
        max_parallel_requests,
        budget_duration,
    } = org;
    carried![
        models,
        max_budget,
        soft_budget,
        rpm_limit,
        tpm_limit,
        tpd_limit,
        max_parallel_requests,
        budget_duration
    ]
}

/// The project's record, or the first field it holds that no record field has a type for yet.
pub(super) fn project(project: &ProjectEntry) -> Result<ProjectRecord, &'static str> {
    let ProjectEntry {
        project_id,
        project_alias,
        description: _,
        team_id,
        budget_id,
        metadata,
        models,
        spend: _,
        model_spend: _,
        model_rpm_limit,
        model_tpm_limit,
        blocked,
        object_permission_id,
        created_by: _,
        updated_by: _,
        created_at: _,
        updated_at: _,
        litellm_budget_table,
        object_permission,
        last_refreshed_at: _,
        unread,
    } = project;
    let unread = &newer(unread);
    let held = carried![
        budget_id,
        metadata,
        model_rpm_limit,
        model_tpm_limit,
        object_permission_id,
        litellm_budget_table,
        object_permission,
        unread,
    ];
    if let Some(field) = held {
        return Err(field);
    }
    Ok(ProjectRecord {
        project_id: project_id.clone(),
        project_alias: project_alias.clone(),
        team_id: team_id.clone(),
        models: models.clone(),
        blocked: *blocked,
        ..ProjectRecord::default()
    })
}

/// The first column of `project` that holds something nothing in the native app reads yet, which
/// sends its caller to Python, taken apart by name as [`unserved`] takes a team.
pub(crate) fn unserved_project(project: &ProjectRecord) -> Option<&'static str> {
    let ProjectRecord {
        project_id: _,
        project_alias: _,
        team_id,
        models,
        blocked,
        budget: Budget {
            max: max_budget,
            spend: _,
        },
    } = project;
    carried![team_id, models, blocked, max_budget]
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use litellm_python_compat::json;
    use litellm_redis_store::FromValue;
    use rstest::rstest;

    use super::*;

    fn cached(entry: &str) -> TeamEntry {
        TeamEntry::from_value(&json::loads(entry).unwrap()).unwrap()
    }

    fn cached_user(entry: &str) -> UserEntry {
        UserEntry::from_value(&json::loads(entry).unwrap()).unwrap()
    }

    #[rstest]
    #[case::object_permission_id(
        r#"{"object_permission_id": "permission"}"#,
        "object_permission_id"
    )]
    #[case::budget_limits(
        r#"{"budget_limits": [{"budget_duration": "30d", "max_budget": 5.0}]}"#,
        "budget_limits"
    )]
    #[case::budget_reset_at(r#"{"budget_reset_at": "2100-01-01T00:00:00"}"#, "budget_reset_at")]
    #[case::model_max_budget(r#"{"model_max_budget": {"claude": 1.0}}"#, "model_max_budget")]
    #[case::organization_memberships(
        r#"{"organization_memberships": [{
            "user_id": "user-1",
            "organization_id": "org",
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00"
        }]}"#,
        "organization_memberships"
    )]
    #[case::object_permission(
        r#"{"object_permission": {"object_permission_id": "permission"}}"#,
        "object_permission"
    )]
    #[case::a_field_a_newer_pod_wrote(r#"{"token_budget": 5}"#, "unread")]
    fn a_user_holding_a_structure_is_declined_over_that_field(
        #[case] held: &str,
        #[case] field: &str,
    ) {
        let entry = held.replacen('{', r#"{"user_id": "user-1", "#, 1);

        assert_eq!(user(&cached_user(&entry)).err(), Some(field));
    }

    #[rstest]
    #[case::deactivated(r#"{"scim_active": false}"#, Ok(true))]
    #[case::active(r#"{"scim_active": true}"#, Ok(false))]
    #[case::active_beside_other_metadata(
        r#"{"scim_active": true, "department": "eng"}"#,
        Err("metadata")
    )]
    #[case::another_flag_that_is_off(r#"{"suspended": false}"#, Err("metadata"))]
    fn a_user_is_deactivated_only_by_scim_and_declined_over_any_other_metadata(
        #[case] metadata: &str,
        #[case] deactivated: Result<bool, &str>,
    ) {
        let entry = format!(r#"{{"user_id": "user-1", "metadata": {metadata}}}"#);

        assert_eq!(
            user(&cached_user(&entry)).map(|user| user.deactivated),
            deactivated
        );
    }

    #[test]
    fn a_teams_organization_is_copied_and_named_unserved() {
        let team = team(&cached(r#"{"team_id": "squad", "organization_id": "org"}"#)).unwrap();

        assert_eq!(
            (team.row.organization_id.as_deref(), unserved(&team.row)),
            (Some("org"), Some("organization_id"))
        );
    }

    #[test]
    fn a_teams_metadata_limits_and_priority_are_copied() {
        let team = team(&cached(
            r#"{"team_id": "squad", "metadata": {"model_rpm_limit": {"claude": 2}, "model_tpm_limit": {"claude": 100}, "priority": "prod"}}"#,
        ))
        .unwrap();

        assert_eq!(
            (
                team.row.model_rpm_limit,
                team.row.model_tpm_limit,
                team.row.priority
            ),
            (
                BTreeMap::from([("claude".to_owned(), 2)]),
                BTreeMap::from([("claude".to_owned(), 100)]),
                Some("prod".to_owned())
            )
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

    fn cached_membership(entry: &str) -> TeamMembershipEntry {
        TeamMembershipEntry::from_value(&json::loads(entry).unwrap()).unwrap()
    }

    #[rstest]
    #[case::budget_id(r#"{"budget_id": "budget"}"#, "budget_id")]
    #[case::rollover_credit(r#"{"rollover_credit": 1.5}"#, "rollover_credit")]
    #[case::litellm_budget_table(
        r#"{"litellm_budget_table": {"max_budget": 5.0, "created_at": "2026-01-01T00:00:00"}}"#,
        "litellm_budget_table"
    )]
    #[case::a_field_a_newer_pod_wrote(r#"{"token_budget": 5}"#, "unread")]
    fn a_membership_holding_a_budget_of_its_own_is_declined_over_that_field(
        #[case] held: &str,
        #[case] field: &str,
    ) {
        let entry = format!(
            r#"{{"user_id": "user-1", "team_id": "team", {}"#,
            &held[1..]
        );

        assert_eq!(
            membership(&cached_membership(&entry), AffiliationRead::Jwt),
            Err(field)
        );
    }

    #[test]
    fn a_membership_is_read_as_the_user_and_the_team_it_joins() {
        let entry = r#"{"user_id": "user-1", "team_id": "team", "spend": 2.0, "total_spend": 5.0}"#;

        assert_eq!(
            membership(&cached_membership(entry), AffiliationRead::Jwt),
            Ok(MembershipRecord {
                user_id: "user-1".to_owned(),
                team_id: "team".to_owned(),
                budget: Budget {
                    max: None,
                    spend: 2.0,
                },
                ..MembershipRecord::default()
            })
        );
    }
}
