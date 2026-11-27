//! A team as the database holds it, in the shape the auth cache holds it.

use std::sync::Arc;

use litellm_redis_store::auth_cache::TeamEntry;
use litellm_repositories::{ObjectPermissionRepository, Side, Team, TeamRepository};

use crate::{
    Error,
    budget_limit::windows,
    column::{nullable, nullable_or_text, time},
    member::roster,
    object_permission::ObjectPermissions,
};

/// The team as the database holds it, in the shape the auth cache holds it: the row with its
/// model alias table and the object permission it points at.
pub struct TeamDatabase {
    teams: Arc<TeamRepository>,
    object_permissions: ObjectPermissions,
}

impl TeamDatabase {
    pub fn new(
        teams: Arc<TeamRepository>,
        object_permissions: Arc<ObjectPermissionRepository>,
    ) -> Self {
        Self {
            teams,
            object_permissions: ObjectPermissions::new(object_permissions),
        }
    }

    /// The team `team_id` with the object permission it points at, as `get_team_object` nests
    /// it, or `None` when no row has that id. `last_refreshed_at` is unset, since the host that
    /// caches the team stamps it.
    pub async fn team(&self, team_id: &str, side: Side) -> Result<Option<TeamEntry>, Error> {
        let Some(row) = self.teams.find_with_model_aliases(team_id, side).await? else {
            return Ok(None);
        };
        self.nested(team_entry(row)?, side).await.map(Some)
    }

    async fn nested(&self, team: TeamEntry, side: Side) -> Result<TeamEntry, Error> {
        let object_permission = self
            .object_permissions
            .nested(team.object_permission_id.as_deref(), side)
            .await?;
        Ok(TeamEntry {
            object_permission,
            ..team
        })
    }

    /// The team named `alias` with the object permission it points at, read from a replica as
    /// `get_team_object_by_alias` reads it. Every
    /// row found is shaped before they are counted, since Python validates both rows before it
    /// refuses an alias two teams share.
    pub async fn team_by_alias(&self, alias: &str) -> Result<AliasMatch<TeamEntry>, Error> {
        let entries = self
            .teams
            .find_up_to_two_by_alias(alias)
            .await?
            .into_iter()
            .map(team_entry)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(match <[TeamEntry; 1]>::try_from(entries) {
            Ok([entry]) => AliasMatch::Found(self.nested(entry, Side::Read).await?),
            Err(entries) if entries.is_empty() => AliasMatch::Absent,
            Err(_) => AliasMatch::Ambiguous,
        })
    }
}

/// What a read by alias finds: the one row with that alias, none, or more than one, which
/// Python refuses rather than pick one.
#[derive(Debug, PartialEq)]
pub enum AliasMatch<E> {
    Found(E),
    Absent,
    Ambiguous,
}

/// What `LiteLLM_TeamTableCachedObj.model_dump(mode="json")` holds for `team`, or the column
/// the model would refuse the row for.
fn team_entry(team: Team) -> Result<TeamEntry, Error> {
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
    Ok(TeamEntry {
        team_alias,
        team_id,
        organization_id,
        admins,
        members,
        members_with_roles: roster(members_with_roles.get())?,
        team_member_permissions: Some(team_member_permissions),
        metadata: nullable_or_text("metadata", Some(&metadata))?,
        tpm_limit,
        rpm_limit,
        tpd_limit,
        max_budget,
        soft_budget,
        budget_duration,
        budget_limits: windows(nullable_or_text("budget_limits", budget_limits.as_ref())?),
        models,
        blocked,
        router_settings: nullable_or_text("router_settings", router_settings.as_ref())?,
        access_group_ids: Some(access_group_ids),
        default_team_member_models: Some(default_team_member_models),
        spend: Some(spend),
        max_parallel_requests: max_parallel_requests.map(i64::from),
        budget_reset_at: time("budget_reset_at", budget_reset_at)?,
        model_id: model_id.map(i64::from),
        model_spend: nullable("model_spend", Some(&model_spend))?,
        model_max_budget: nullable_or_text("model_max_budget", Some(&model_max_budget))?,
        policies: Some(policies),
        allow_team_guardrail_config: Some(allow_team_guardrail_config),
        litellm_model_table: nullable("litellm_model_table", litellm_model_table.as_ref())?,
        object_permission: None,
        object_permission_id,
        updated_at: time("updated_at", Some(updated_at))?,
        created_at: time("created_at", Some(created_at))?,
        last_refreshed_at: None,
        unread: Vec::new(),
    })
}
