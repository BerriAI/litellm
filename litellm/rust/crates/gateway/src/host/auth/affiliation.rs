//! Reads a caller's team, key owner and their membership, organization and project from the auth
//! cache into core records. Each entry is taken apart by name, so a field added to it does not
//! compile here until it is copied onto the record, read for nothing or declined. A column that
//! holds a structure no record field has a type for yet is declined here, naming the field, and
//! [`unserved`], [`unserved_user`], [`unserved_membership`], [`unserved_org`] and
//! [`unserved_project`] name the first copied column the native app does not serve yet.

use jiff::Timestamp;
use litellm_gateway_auth_core::{
    AffiliationRead,
    authz::ModelList,
    identity::jwt::{Org, Team, User},
    record::{
        Budget, BudgetExtras, BudgetWindow, MembershipRecord, MinuteLimits, OrgRecord,
        ProjectRecord, TeamModelAliasTable, TeamRecord, UserRecord,
    },
    routes::passthrough::PassthroughGrants,
};
use litellm_python_compat::{Value, json};
use litellm_redis_store::auth_cache::{
    Bag, BudgetEntry, BudgetLimitEntry, ModelAliases, ModelTableEntry, OrganizationEntry,
    ProjectEntry, TeamEntry, TeamMembershipEntry, UserEntry,
};
use serde::de::DeserializeOwned;

use super::{
    carried::{Carried, carried, newer},
    metadata::{self, TeamMetadata},
    object_permission,
};

/// The `metadata` key SCIM sets to `False` on a user it deactivated.
const SCIM_ACTIVE: &str = "scim_active";

fn read_bag<T: DeserializeOwned>(bag: &Bag) -> Option<T> {
    serde_json::from_str(&json::dumps(&Value::Dict(bag.entries().to_vec())).ok()?).ok()
}

fn column<T: Default + DeserializeOwned>(
    bag: Option<&Bag>,
    field: &'static str,
) -> Result<T, &'static str> {
    bag.map_or_else(|| Ok(T::default()), |bag| read_bag(bag).ok_or(field))
}

pub(crate) fn windows(
    windows: Option<&[BudgetLimitEntry]>,
) -> Result<Vec<BudgetWindow>, &'static str> {
    windows
        .unwrap_or_default()
        .iter()
        .map(|window| {
            let timestamp = |time: litellm_python_compat::datetime::IsoDateTime| {
                Timestamp::from_microsecond(time.utc_microseconds()).map_err(|_| "budget_limits")
            };
            Ok(BudgetWindow {
                budget_duration: window.budget_duration.clone(),
                max_budget: window.max_budget,
                reset_at: window.reset_at.map(timestamp).transpose()?,
                window_start: window.window_start.0.map(timestamp).transpose()?,
            })
        })
        .collect()
}

fn model_table(table: &ModelTableEntry) -> Result<TeamModelAliasTable, &'static str> {
    let model_aliases = match table.model_aliases.as_ref() {
        Some(ModelAliases::Text(text)) => {
            serde_json::from_str(text).map_err(|_| "litellm_model_table")?
        }
        Some(ModelAliases::Map(map)) => read_bag(map).ok_or("litellm_model_table")?,
        None => Default::default(),
    };
    Ok(TeamModelAliasTable {
        id: table.id,
        model_aliases,
    })
}

/// A team as the native app reads it: its record, and the budget row its metadata names as the
/// default budget of a member whose membership sets none.
pub(super) struct ReadTeam {
    pub(super) team: Team<TeamRecord>,
    pub(super) member_budget_id: Option<String>,
}

/// The team, or the first field it holds that no record field has a type for yet.
pub(super) fn team(team: &TeamEntry) -> Result<ReadTeam, &'static str> {
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
        budget_reset_at: _,
        model_id,
        model_spend,
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
    let held = unread_models
        .then_some("models")
        .or_else(|| carried![unread]);
    if let Some(field) = held {
        return Err(field);
    }
    if model_id.is_some() && litellm_model_table.is_none() {
        return Err("model_id");
    }
    let Some(TeamMetadata {
        model_rpm_limit,
        model_tpm_limit,
        priority,
        team_member_budget_id,
        policies: policy_metadata,
    }) = metadata
        .as_ref()
        .map_or(Some(TeamMetadata::default()), TeamMetadata::read)
    else {
        return Err("metadata");
    };
    let team = Team {
        team_id: team_id.clone(),
        models: Some(models.clone()),
        passthrough: PassthroughGrants::default(),
        members: members_with_roles
            .iter()
            .filter_map(|member| member.user_id.clone())
            .collect(),
        row: TeamRecord {
            team_id: team_id.clone(),
            metadata: (policy_metadata != Default::default()).then(|| Box::new(policy_metadata)),
            object_permission: object_permission::record(
                object_permission_id.as_deref(),
                object_permission.as_ref(),
            )?,
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
            budget_limits: windows(budget_limits.as_deref())?,
            model_max_budget: column(model_max_budget.as_ref(), "model_max_budget")?,
            model_spend: column(model_spend.as_ref(), "model_spend")?,
            model_id: *model_id,
            litellm_model_table: litellm_model_table.as_ref().map(model_table).transpose()?,
            router_settings: router_settings
                .as_ref()
                .filter(|bag| !bag.is_empty())
                .map(|bag| read_bag(bag).ok_or("router_settings"))
                .transpose()?,
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
    };
    Ok(ReadTeam {
        team,
        member_budget_id: team_member_budget_id,
    })
}

/// The first column of `team` that holds something nothing in the native app reads yet, which
/// sends its caller to Python. The record is taken apart by name, so a column added to it does
/// not compile here until it is named as read or listed. Whoever makes the native app honor a
/// column takes it off the list in the same change.
pub(crate) fn unserved(team: &TeamRecord) -> Option<&'static str> {
    let TeamRecord {
        team_id: _,
        object_permission,
        metadata,
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
        budget_limits: _,
        model_max_budget,
        model_spend: _,
        model_id,
        litellm_model_table,
        router_settings,
        blocked,
        access_group_ids,
        default_team_member_models,
        policies,
        allow_team_guardrail_config,
        model_rpm_limit,
        model_tpm_limit,
        priority,
        member_max_budget: _,
        member_budget_extras,
    } = team;
    let litellm_model_table = &litellm_model_table.is_some();
    let router_settings = &router_settings.is_some();
    let object_permission = &object_permission.is_some();
    let metadata = &metadata.is_some();
    carried![
        object_permission,
        metadata,
        organization_id,
        tpm_limit,
        rpm_limit,
        tpd_limit,
        max_parallel_requests,
        max_budget,
        soft_budget,
        budget_duration,
        model_max_budget,
        model_id,
        litellm_model_table,
        router_settings,
        blocked,
        access_group_ids,
        default_team_member_models,
        policies,
        allow_team_guardrail_config,
        model_rpm_limit,
        model_tpm_limit,
        priority,
    ]
    .or_else(|| unserved_budget_extras(member_budget_extras))
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
        budget_reset_at: _,
        allowed_cache_controls,
        policies,
        model_spend,
        model_max_budget,
        created_at: _,
        updated_at: _,
        organization_memberships,
        object_permission,
        unread,
    } = user;
    let unread = &newer(unread);
    let deactivated = scim_deactivated(metadata.as_ref()).ok_or("metadata")?;
    let held = carried![organization_memberships, unread];
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
            object_permission: object_permission::record(
                object_permission_id.as_deref(),
                object_permission.as_ref(),
            )?,
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
            budget_limits: windows(budget_limits.as_deref())?,
            model_max_budget: column(model_max_budget.as_ref(), "model_max_budget")?,
            model_spend: column(model_spend.as_ref(), "model_spend")?,
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
        object_permission,
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
        budget: _,
        rollover_max_budget: _,
        rollover_credit: _,
        budget_duration: _,
        budget_limits: _,
        model_max_budget,
        model_spend: _,
        allowed_cache_controls,
        policies,
    } = user;
    let object_permission = &object_permission.is_some();
    carried![
        object_permission,
        team_id,
        organization_id,
        models,
        tpm_limit,
        rpm_limit,
        max_parallel_requests,
        model_max_budget,
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
        budget_extras,
    } = budget_row_or_spend(litellm_budget_table.as_ref().map(|row| &row.budget), spend)?;
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
        budget_extras,
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
        budget: _,
        rollover_credit: _,
        soft_budget,
        tpd_limit,
        max_parallel_requests,
        budget_duration: _,
        budget_extras:
            BudgetExtras {
                model_max_budget,
                model_rpm_limit,
                model_tpm_limit,
                allowed_models,
                rollover_max_budget: _,
                temp_budget_increase: _,
                temp_budget_expiry: _,
            },
    } = membership;
    carried![
        model_max_budget,
        model_rpm_limit,
        model_tpm_limit,
        allowed_models,
        soft_budget,
        rpm_limit,
        tpm_limit,
        tpd_limit,
        max_parallel_requests
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
        model_spend,
        created_by: _,
        updated_by: _,
        users: _,
        litellm_budget_table,
        object_permission,
        object_permission_id,
        unread,
    } = org;
    let unread = &newer(unread);
    if let Some(field) = carried![unread] {
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
        budget_extras,
    } = budget_row(limits, *spend)?;
    Ok(Org {
        organization_id: organization_id.clone(),
        row: OrgRecord {
            organization_id: organization_id.clone(),
            metadata: metadata::policies(metadata.as_ref())?,
            object_permission: object_permission::record(
                object_permission_id.as_deref(),
                object_permission.as_ref(),
            )?,
            organization_alias: organization_alias.clone(),
            models: models.clone(),
            limits,
            budget,
            soft_budget,
            tpd_limit,
            max_parallel_requests,
            budget_duration,
            model_spend: column(model_spend.as_ref(), "model_spend")?,
            budget_extras,
        },
    })
}

/// A budget row's plain values, which the record of whatever the row budgets copies.
#[derive(Default)]
pub(super) struct BudgetRow {
    pub(super) limits: MinuteLimits,
    pub(super) budget: Budget,
    pub(super) soft_budget: Option<f64>,
    pub(super) tpd_limit: Option<i64>,
    pub(super) max_parallel_requests: Option<i64>,
    pub(super) budget_duration: Option<String>,
    pub(super) budget_extras: BudgetExtras,
}

/// [`budget_row`] when the row has a budget row, and otherwise only `spend`, with no budget.
pub(super) fn budget_row_or_spend(
    entry: Option<&BudgetEntry>,
    spend: f64,
) -> Result<BudgetRow, &'static str> {
    match entry {
        Some(entry) => budget_row(entry, spend),
        None => Ok(BudgetRow {
            budget: Budget { max: None, spend },
            ..BudgetRow::default()
        }),
    }
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
    let unread = &newer(unread);
    if let Some(field) = carried![unread] {
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
        budget_extras: BudgetExtras {
            model_max_budget: column(model_max_budget.as_ref(), "model_max_budget")?,
            model_rpm_limit: model_rpm_limit
                .as_ref()
                .map(|map| map.0.iter().cloned().collect())
                .unwrap_or_default(),
            model_tpm_limit: model_tpm_limit
                .as_ref()
                .map(|map| map.0.iter().cloned().collect())
                .unwrap_or_default(),
            allowed_models: allowed_models.clone().unwrap_or_default(),
            rollover_max_budget: rollover_max_budget.0,
            temp_budget_increase: *temp_budget_increase,
            temp_budget_expiry: temp_budget_expiry
                .map(|time| {
                    Timestamp::from_microsecond(time.utc_microseconds())
                        .map_err(|_| "temp_budget_expiry")
                })
                .transpose()?,
        },
    })
}

pub(super) fn unserved_budget_extras(extras: &BudgetExtras) -> Option<&'static str> {
    let BudgetExtras {
        model_max_budget,
        model_rpm_limit,
        model_tpm_limit,
        allowed_models,
        rollover_max_budget,
        temp_budget_increase,
        temp_budget_expiry,
    } = extras;
    let temp_budget_expiry = &temp_budget_expiry.is_some();
    carried![
        rollover_max_budget,
        model_max_budget,
        model_tpm_limit,
        model_rpm_limit,
        allowed_models,
        temp_budget_increase,
        temp_budget_expiry,
    ]
}

/// The first column of `org` that holds something nothing in the native app reads yet, which
/// sends its caller to Python, taken apart by name as [`unserved`] takes a team.
pub(crate) fn unserved_org(org: &OrgRecord) -> Option<&'static str> {
    let OrgRecord {
        object_permission,
        metadata,
        model_spend: _,
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
        budget_extras,
    } = org;
    let object_permission = &object_permission.is_some();
    let metadata = &metadata.is_some();
    carried![object_permission, metadata]
        .or_else(|| unserved_budget_extras(budget_extras))
        .or_else(|| {
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
        })
}

/// The project's record, or the first field it holds that no record field has a type for yet.
pub(super) fn project(project: &ProjectEntry) -> Result<ProjectRecord, &'static str> {
    let ProjectEntry {
        project_id,
        project_alias,
        description: _,
        team_id,
        budget_id: _,
        metadata,
        models,
        spend,
        model_spend,
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
    let held = carried![unread];
    if let Some(field) = held {
        return Err(field);
    }
    let BudgetRow {
        limits,
        budget,
        soft_budget,
        tpd_limit,
        max_parallel_requests,
        budget_duration,
        budget_extras,
    } = budget_row_or_spend(litellm_budget_table.as_ref(), *spend)?;
    Ok(ProjectRecord {
        project_id: project_id.clone(),
        metadata: metadata::policies(metadata.as_ref())?,
        object_permission: object_permission::record(
            object_permission_id.as_deref(),
            object_permission.as_ref(),
        )?,
        project_alias: project_alias.clone(),
        team_id: team_id.clone(),
        models: models.clone(),
        blocked: *blocked,
        limits,
        budget,
        soft_budget,
        tpd_limit,
        max_parallel_requests,
        budget_duration,
        budget_extras,
        model_spend: column(model_spend.as_ref(), "model_spend")?,
        model_rpm_limit: column(model_rpm_limit.as_ref(), "model_rpm_limit")?,
        model_tpm_limit: column(model_tpm_limit.as_ref(), "model_tpm_limit")?,
    })
}

/// The first column of `project` that holds something nothing in the native app reads yet, which
/// sends its caller to Python, taken apart by name as [`unserved`] takes a team.
pub(crate) fn unserved_project(project: &ProjectRecord) -> Option<&'static str> {
    let ProjectRecord {
        object_permission,
        metadata,
        model_spend: _,
        model_rpm_limit,
        model_tpm_limit,
        project_id: _,
        project_alias: _,
        team_id,
        models,
        blocked,
        limits: MinuteLimits {
            rpm: rpm_limit,
            tpm: tpm_limit,
        },
        budget: Budget {
            max: max_budget,
            spend: _,
        },
        soft_budget,
        tpd_limit,
        max_parallel_requests,
        budget_duration,
        budget_extras,
    } = project;
    let object_permission = &object_permission.is_some();
    let metadata = &metadata.is_some();
    carried![object_permission, metadata]
        .or_else(|| unserved_budget_extras(budget_extras))
        .or_else(|| {
            carried![
                team_id,
                models,
                blocked,
                max_budget,
                soft_budget,
                rpm_limit,
                tpm_limit,
                tpd_limit,
                max_parallel_requests,
                budget_duration,
                model_rpm_limit,
                model_tpm_limit,
            ]
        })
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
    #[case::missing("{}", None)]
    #[case::null(
        r#"{"model_max_budget":null,"allowed_models":null,"temp_budget_expiry":null}"#,
        None
    )]
    #[case::model_budget(r#"{"model_max_budget":{"claude":2}}"#, Some("model_max_budget"))]
    #[case::rpm(r#"{"model_rpm_limit":{"claude":2}}"#, Some("model_rpm_limit"))]
    #[case::tpm(r#"{"model_tpm_limit":{"claude":100}}"#, Some("model_tpm_limit"))]
    #[case::models(r#"{"allowed_models":["claude"]}"#, Some("allowed_models"))]
    #[case::rollover(r#"{"rollover_max_budget":0}"#, Some("rollover_max_budget"))]
    #[case::increase(r#"{"temp_budget_increase":2}"#, Some("temp_budget_increase"))]
    #[case::expiry(
        r#"{"temp_budget_expiry":"2100-01-01T00:00:00"}"#,
        Some("temp_budget_expiry")
    )]
    fn budget_extras_are_read_and_named_unserved(#[case] input: &str, #[case] field: Option<&str>) {
        let entry = BudgetEntry::from_value(&json::loads(input).unwrap()).unwrap();
        let row = budget_row(&entry, 3.0).unwrap();
        assert_eq!(unserved_budget_extras(&row.budget_extras), field);
        assert_eq!(row.budget.spend, 3.0);
    }

    #[rstest]
    fn a_malformed_model_budget_declines_the_budget_row() {
        let input = json::loads(r#"{"model_max_budget":{"claude":[]}}"#).unwrap();
        let entry = BudgetEntry::from_value(&input).unwrap();
        assert_eq!(budget_row(&entry, 0.0).err(), Some("model_max_budget"));
    }

    #[rstest]
    #[case::object_permission_id(
        r#"{"object_permission_id": "permission"}"#,
        "object_permission_id"
    )]
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

        let result = user(&cached_user(&entry));
        let unserved = result
            .as_ref()
            .map_or_else(|field| Some(*field), |user| unserved_user(&user.row));
        assert_eq!(unserved, Some(field));
    }

    #[rstest]
    #[case::budget_limits(
        r#"{"budget_limits": [{"budget_duration": "30d", "max_budget": 5.0}]}"#,
        None
    )]
    #[case::model_max_budget(r#"{"model_max_budget": {"claude": 1.0}}"#, Some("model_max_budget"))]
    fn a_users_budget_policies_are_copied_and_only_per_model_is_named_unserved(
        #[case] held: &str,
        #[case] field: Option<&str>,
    ) {
        let entry = held.replacen(
            '{',
            r#"{"user_id":"user-1","model_spend":{"claude":0.5},"#,
            1,
        );
        let owner = user(&cached_user(&entry)).unwrap().row;
        assert_eq!(owner.model_spend["claude"], 0.5);
        assert_eq!(unserved_user(&owner), field);
    }

    #[rstest]
    #[case::limits(r#"{"model_rpm_limit":{"claude":2},"model_tpm_limit":{"claude":100}}"#, Ok((Some(2), Some(100))))]
    #[case::absent(r#"{"model_rpm_limit":null}"#, Ok((None, None)))]
    #[case::malformed(r#"{"model_tpm_limit":{"claude":[]}}"#, Err("model_tpm_limit"))]
    fn a_projects_direct_limits_are_read_from_its_own_row(
        #[case] held: &str,
        #[case] expected: Result<(Option<i64>, Option<i64>), &str>,
    ) {
        let entry = held.replacen('{', r#"{"project_id":"project","#, 1);
        let row = ProjectEntry::from_value(&json::loads(&entry).unwrap()).unwrap();
        assert_eq!(
            project(&row).map(|project| (
                project.model_rpm_limit.get("claude").copied(),
                project.model_tpm_limit.get("claude").copied(),
            )),
            expected
        );
    }

    #[rstest]
    fn a_teams_window_spend_and_routing_configuration_are_read() {
        let team = team(&cached(r#"{
            "team_id":"squad",
            "budget_limits":[{"budget_duration":"30d","max_budget":5,"window_start":"2026-11-01T00:00:00Z"}],
            "model_max_budget":{"claude":{"budget_limit":2,"time_period":"7d"}},
            "model_spend":{"claude":1.5},
            "model_id":7,
            "litellm_model_table":{"id":7,"created_by":"admin","updated_by":"admin",
                                  "model_aliases":"{\"friendly\":\"claude\"}"},
            "router_settings":{"timeout":4,"future_setting":true}
        }"#)).unwrap().team.row;
        assert_eq!(
            team.budget_limits,
            vec![BudgetWindow {
                budget_duration: "30d".to_owned(),
                max_budget: 5.0,
                reset_at: None,
                window_start: Some("2026-11-01T00:00:00Z".parse().unwrap()),
            }]
        );
        let expected_budget = litellm_gateway_auth_core::record::ModelBudgetConfig {
            max_budget: Some(2.0),
            budget_duration: Some("7d".to_owned()),
            rpm_limit: None,
            tpm_limit: None,
        };
        assert_eq!(
            team.model_max_budget["claude"],
            litellm_gateway_auth_core::record::ModelBudget::Config(expected_budget),
        );
        assert_eq!(team.model_spend["claude"], 1.5);
        assert_eq!(team.model_id, Some(7));
        let aliases = team.litellm_model_table.unwrap();
        assert_eq!(aliases.id, Some(7));
        assert_eq!(aliases.model_aliases["friendly"], "claude");
        let router = team.router_settings.unwrap();
        assert_eq!(router.timeout, Some(4.0));
        assert_eq!(router.additional_fields["future_setting"], true);
    }

    #[rstest]
    #[case::missing_target(r#"{"model_id":7}"#, Some("model_id"))]
    #[case::bad_budget(r#"{"model_max_budget":{"claude":[1]}}"#, Some("model_max_budget"))]
    #[case::bad_router(r#"{"router_settings":{"timeout":[]}}"#, Some("router_settings"))]
    #[case::bad_aliases(r#"{"litellm_model_table":{"created_by":"admin","updated_by":"admin","model_aliases":{"a":0}}}"#, Some("litellm_model_table"))]
    #[case::null(
        r#"{"budget_limits":null,"model_spend":null,"model_max_budget":null,
        "model_id":null,"litellm_model_table":null,"router_settings":null}"#,
        None
    )]
    fn a_team_with_missing_or_malformed_policy_data(
        #[case] fields: &str,
        #[case] decline: Option<&str>,
    ) {
        let entry = fields.replacen('{', r#"{"team_id":"squad","#, 1);
        let result = team(&cached(&entry));
        assert_eq!(
            result
                .as_ref()
                .err()
                .copied()
                .or_else(|| result.ok().and_then(|read| unserved(&read.team.row))),
            decline,
        );
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
        let team = team(&cached(r#"{"team_id": "squad", "organization_id": "org"}"#))
            .unwrap()
            .team;

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
        .unwrap()
        .team;

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
    fn a_teams_guardrails_and_callbacks_are_projected_without_serving_them() {
        let team = team(&cached(
            r#"{"team_id":"squad","metadata":{"guardrails":["pii"],"logging":[{"callback_name":"langfuse","callback_type":"failure","callback_vars":{"langfuse_secret_key":"test-only"}}]}}"#,
        ))
        .unwrap()
        .team;
        let metadata = team.row.metadata.as_ref().unwrap();
        assert_eq!(
            metadata.guardrails.as_deref(),
            Some(["pii".to_owned()].as_slice())
        );
        let [callback] = metadata.logging.as_deref().unwrap() else {
            panic!("expected one logging callback");
        };
        assert_eq!(callback.callback_name, "langfuse");
        assert_eq!(
            callback.callback_type,
            Some(litellm_gateway_auth_core::record::LoggingCallbackType::Failure)
        );
        assert_eq!(
            callback.callback_vars["langfuse_secret_key"].expose(),
            "test-only"
        );
        let debug = format!("{metadata:?}");
        assert!(!debug.contains("test-only"));
        assert!(debug.contains("[REDACTED]"));
        assert_eq!(unserved(&team.row), Some("metadata"));
    }

    #[rstest]
    fn legacy_team_callbacks_preserve_redacted_settings_without_serving_them() {
        let team = team(&cached(
            r#"{"team_id":"squad","metadata":{"callback_settings":{"success_callback":["langfuse"],"callback_vars":{"langfuse_secret_key":"legacy-test-only"}}}}"#,
        ))
        .unwrap()
        .team;
        let metadata = team.row.metadata.as_ref().unwrap();
        let settings = metadata.callback_settings.as_ref().unwrap();
        assert_eq!(
            settings.success_callback.as_deref(),
            Some(["langfuse".to_owned()].as_slice())
        );
        let vars = settings.callback_vars.as_ref().unwrap();
        assert_eq!(vars["langfuse_secret_key"].expose(), "legacy-test-only");
        let debug = format!("{metadata:?}");
        assert!(!debug.contains("legacy-test-only"));
        assert!(debug.contains("[REDACTED]"));
        assert_eq!(unserved(&team.row), Some("metadata"));
    }

    #[rstest]
    #[case::null("null", Ok(None))]
    #[case::empty("{}", Ok(None))]
    #[case::populated(r#"{"model_rpm_limit":{"claude":5}}"#, Ok(Some(5)))]
    #[case::malformed(r#"{"model_rpm_limit":{"claude":[]}}"#, Err("metadata"))]
    #[case::tags(r#"{"tags":["internal"]}"#, Err("metadata"))]
    fn organization_metadata_carries_only_known_policy(
        #[case] metadata: &str,
        #[case] expected: Result<Option<i64>, &str>,
    ) {
        let input = format!(
            r#"{{"organization_id":"org","budget_id":"budget","created_by":"test","updated_by":"test","litellm_budget_table":{{"budget_id":"budget"}},"metadata":{metadata}}}"#
        );
        let entry = OrganizationEntry::from_value(&json::loads(&input).unwrap()).unwrap();
        let row = org(&entry).map(|org| org.row);
        let read = row.as_ref().map_err(|field| *field).map(|org| {
            org.metadata
                .as_ref()
                .and_then(|metadata| metadata.model_rpm_limit.as_ref())
                .and_then(|limit| limit.get("claude").copied())
        });
        assert_eq!(read, expected);
        if let Ok(row) = &row {
            assert_eq!(unserved_org(row), expected.unwrap().map(|_| "metadata"));
        }
    }

    #[rstest]
    fn project_metadata_keeps_provider_fallback_order_and_is_unserved() {
        let value = json::loads(r#"{"project_id":"project","metadata":{"model_itpm_limit":{"claude":4},"model_config":{"defaultconfig":{"z-provider":{"litellm_credentials":"first"},"a-provider":{"litellm_credentials":"second"}}}}}"#).unwrap();
        let entry = ProjectEntry::from_value(&value).unwrap();
        let project = project(&entry).unwrap();
        let metadata = project.metadata.as_ref().unwrap();
        assert_eq!(metadata.model_itpm_limit.as_ref().unwrap()["claude"], 4);
        let credentials = &metadata.model_config.as_ref().unwrap()["defaultconfig"].0;
        let order = credentials
            .iter()
            .map(|(provider, credential)| {
                (
                    provider.as_str(),
                    credential
                        .litellm_credentials
                        .as_ref()
                        .map(|name| name.expose()),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            order,
            [
                ("z-provider", Some("first")),
                ("a-provider", Some("second"))
            ]
        );
        assert_eq!(unserved_project(&project), Some("metadata"));
        let debug = format!("{metadata:?}");
        assert!(!debug.contains("first"));
        assert!(!debug.contains("second"));
        assert!(debug.contains("[REDACTED]"));
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
