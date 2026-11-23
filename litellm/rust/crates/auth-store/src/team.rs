//! A team as the database holds it, in the shape the auth cache holds it.

use std::sync::Arc;

use litellm_python_compat::{Value, json};
use litellm_redis_store::{
    FromValue,
    auth_cache::{CachedMember, CachedTeam},
};
use litellm_repositories::{RawJson, Side, Team, TeamRepository};

use crate::{
    Error,
    key::{column, member, nullable, nullable_list, time},
};

/// The team as the database holds it, in the shape the auth cache holds it: the row with its
/// model alias table, and nothing else nested.
pub struct TeamDatabase {
    teams: Arc<TeamRepository>,
}

impl TeamDatabase {
    pub fn new(teams: Arc<TeamRepository>) -> Self {
        Self { teams }
    }

    /// The team `team_id`, or `None` when no row has that id. The object permission the team
    /// points at is not read, and `last_refreshed_at` is unset, since the host that caches the
    /// team stamps it.
    pub async fn team(&self, team_id: &str, side: Side) -> Result<Option<CachedTeam>, Error> {
        let Some(found) = self.teams.find_with_model_aliases(team_id, side).await? else {
            return Ok(None);
        };
        cached_team(&found).map(Some)
    }
}

/// The roster as the model reads it, each entry what `Member` dumps. The column's default is
/// an empty object, which the model reads as no members.
fn roster(raw: &RawJson) -> Result<Vec<CachedMember>, Error> {
    let refused = || Error::Row {
        column: "members_with_roles",
    };
    let entries = match json::loads(raw.get()) {
        Ok(Value::List(entries)) => entries,
        Ok(Value::Dict(fields)) if fields.is_empty() => Vec::new(),
        _ => return Err(refused()),
    };
    entries
        .iter()
        .map(|entry| {
            let entry = CachedMember::from_value(entry).ok();
            entry
                .and_then(|entry| member(entry).ok())
                .ok_or_else(refused)
        })
        .collect()
}

/// What `LiteLLM_TeamTableCachedObj.model_dump(mode="json")` holds for `team`, or the column
/// the model would refuse the row for.
fn cached_team(team: &Team) -> Result<CachedTeam, Error> {
    let Team {
        team_id,
        team_alias,
        organization_id,
        admins,
        members,
        members_with_roles,
        metadata,
        max_budget,
        soft_budget,
        spend,
        models,
        max_parallel_requests,
        tpm_limit,
        rpm_limit,
        tpd_limit,
        budget_duration,
        budget_reset_at,
        blocked,
        created_at,
        updated_at,
        model_spend,
        model_max_budget,
        router_settings,
        team_member_permissions,
        access_group_ids,
        policies,
        default_team_member_models,
        budget_limits,
        model_id,
        allow_team_guardrail_config,
        object_permission_id,
        litellm_model_table,
    } = team;
    Ok(CachedTeam {
        team_alias: team_alias.clone(),
        team_id: team_id.clone(),
        organization_id: organization_id.clone(),
        admins: admins.clone(),
        members: members.clone(),
        members_with_roles: roster(members_with_roles)?,
        team_member_permissions: Some(team_member_permissions.clone()),
        metadata: Some(column("metadata", metadata)?),
        tpm_limit: *tpm_limit,
        rpm_limit: *rpm_limit,
        tpd_limit: *tpd_limit,
        max_budget: *max_budget,
        soft_budget: *soft_budget,
        budget_duration: budget_duration.clone(),
        budget_limits: nullable_list("budget_limits", budget_limits.as_ref())?,
        models: models.clone(),
        blocked: *blocked,
        router_settings: nullable("router_settings", router_settings.as_ref())?,
        access_group_ids: Some(access_group_ids.clone()),
        default_team_member_models: Some(default_team_member_models.clone()),
        spend: Some(*spend),
        max_parallel_requests: max_parallel_requests.map(i64::from),
        budget_reset_at: time("budget_reset_at", *budget_reset_at)?,
        model_id: model_id.map(i64::from),
        model_spend: Some(column("model_spend", model_spend)?),
        model_max_budget: Some(column("model_max_budget", model_max_budget)?),
        policies: Some(policies.clone()),
        allow_team_guardrail_config: Some(*allow_team_guardrail_config),
        litellm_model_table: nullable("litellm_model_table", litellm_model_table.as_ref())?,
        object_permission: None,
        object_permission_id: object_permission_id.clone(),
        updated_at: time("updated_at", Some(*updated_at))?,
        created_at: time("created_at", Some(*created_at))?,
        last_refreshed_at: None,
        unread: Vec::new(),
    })
}
