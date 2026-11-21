//! The key view grouped by the object each column was read from. It reads from and writes to the flat
//! [`KeyViewRow`], so its JSON stays the one flat object Python caches a key as.

use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};

use super::queries::KeyViewRow;
use crate::RawJson;

/// The key as the auth flow sees it: the key's own columns, and the team, team membership, organization
/// and project values it is checked against. A group is `None` when the key's row joined no such object.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "KeyViewRow", into = "KeyViewRow")]
pub struct KeyView {
    pub key: KeyColumns,
    pub team: Option<TeamColumns>,
    pub team_member: Option<TeamMemberColumns>,
    pub organization: Option<OrganizationColumns>,
    pub project: ProjectColumns,
}

/// The key's own columns, with its budget row filling the limits the key leaves unset.
#[derive(Clone, Debug, PartialEq)]
pub struct KeyColumns {
    pub token: String,
    pub key_name: Option<String>,
    pub key_alias: Option<String>,
    pub soft_budget_cooldown: bool,
    pub spend: f64,
    pub total_spend: f64,
    pub expires: Option<NaiveDateTime>,
    pub models: Vec<String>,
    pub aliases: RawJson,
    pub config: RawJson,
    pub router_settings: Option<RawJson>,
    pub user_id: Option<String>,
    pub team_id: Option<String>,
    pub agent_id: Option<String>,
    pub project_id: Option<String>,
    pub permissions: RawJson,
    pub max_parallel_requests: Option<i32>,
    pub metadata: RawJson,
    pub blocked: Option<bool>,
    pub tpm_limit: Option<i64>,
    pub rpm_limit: Option<i64>,
    pub tpd_limit: Option<i64>,
    pub max_budget: Option<f64>,
    pub soft_budget: Option<f64>,
    pub budget_duration: Option<String>,
    pub budget_reset_at: Option<NaiveDateTime>,
    pub allowed_cache_controls: Option<Vec<String>>,
    pub allowed_routes: Option<Vec<String>>,
    pub key_type: Option<String>,
    pub access_group_ids: Option<Vec<String>>,
    pub model_spend: RawJson,
    pub model_max_budget: RawJson,
    pub budget_fallbacks: RawJson,
    pub budget_id: Option<String>,
    /// The key's organization, or its team's when the key names none.
    pub org_id: Option<String>,
    pub object_permission_id: Option<String>,
    pub created_at: Option<NaiveDateTime>,
    pub created_by: Option<String>,
    pub updated_at: Option<NaiveDateTime>,
    pub updated_by: Option<String>,
    pub settings_updated_at: Option<NaiveDateTime>,
    pub last_active: Option<NaiveDateTime>,
    pub rotation_count: Option<i32>,
    pub auto_rotate: Option<bool>,
    pub rotation_interval: Option<String>,
    pub last_rotation_at: Option<NaiveDateTime>,
    pub key_rotation_at: Option<NaiveDateTime>,
    pub budget_limits: Option<RawJson>,
}

/// The `team_*` columns of the team the key's `team_id` names. `spend` is never null on a team, so the
/// flat row's `team_spend` says whether the key joined one.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TeamColumns {
    pub spend: f64,
    pub alias: Option<String>,
    pub tpm_limit: Option<i64>,
    pub rpm_limit: Option<i64>,
    pub tpd_limit: Option<i64>,
    pub max_budget: Option<f64>,
    pub soft_budget: Option<f64>,
    pub model_max_budget: Option<RawJson>,
    pub models: Vec<String>,
    pub blocked: bool,
    pub model_aliases: Option<RawJson>,
    /// The key owner's entry in the team's roster, the flat row's `team_member`.
    pub member: Option<RawJson>,
    pub metadata: Option<RawJson>,
    pub object_permission_id: Option<String>,
}

/// The `team_member_*` columns of the key owner's membership in the key's team. `spend` is never null on
/// a membership, so the flat row's `team_member_spend` says whether the key joined one.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TeamMemberColumns {
    pub spend: f64,
    pub tpm_limit: Option<i64>,
    pub rpm_limit: Option<i64>,
}

/// The `organization_*` columns of the organization the key itself names. `alias` is never null on an
/// organization, so the flat row's `organization_alias` says whether the key joined one.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OrganizationColumns {
    pub alias: String,
    pub max_budget: Option<f64>,
    pub tpm_limit: Option<i64>,
    pub rpm_limit: Option<i64>,
    pub metadata: Option<RawJson>,
}

/// The `project_*` columns. A project's alias is nullable, so nothing says whether the key joined one.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProjectColumns {
    pub alias: Option<String>,
}

/// A flat row that holds a column of `group` while the column that marks the group as joined is null.
/// The query never returns one. A cached row another writer filled by hand can be one.
#[derive(Debug, thiserror::Error)]
#[error("the key row holds {group} columns while {marker} is null")]
pub struct Orphaned {
    group: &'static str,
    marker: &'static str,
}

/// `columns` when the group's `marker` column was set, and `None` when it was not and no other column
/// of the group holds anything.
fn joined<G: Default + PartialEq>(
    group: &'static str,
    marker: &'static str,
    is_joined: bool,
    columns: G,
) -> Result<Option<G>, Orphaned> {
    if is_joined {
        return Ok(Some(columns));
    }
    if columns == G::default() {
        return Ok(None);
    }
    Err(Orphaned { group, marker })
}

impl TryFrom<KeyViewRow> for KeyView {
    type Error = Orphaned;

    fn try_from(row: KeyViewRow) -> Result<Self, Orphaned> {
        let KeyViewRow {
            token,
            key_name,
            key_alias,
            soft_budget_cooldown,
            spend,
            total_spend,
            expires,
            models,
            aliases,
            config,
            router_settings,
            user_id,
            team_id,
            agent_id,
            project_id,
            permissions,
            max_parallel_requests,
            metadata,
            blocked,
            tpm_limit,
            rpm_limit,
            tpd_limit,
            max_budget,
            soft_budget,
            budget_duration,
            budget_reset_at,
            allowed_cache_controls,
            allowed_routes,
            key_type,
            access_group_ids,
            model_spend,
            model_max_budget,
            budget_fallbacks,
            budget_id,
            org_id,
            object_permission_id,
            created_at,
            created_by,
            updated_at,
            updated_by,
            settings_updated_at,
            last_active,
            rotation_count,
            auto_rotate,
            rotation_interval,
            last_rotation_at,
            key_rotation_at,
            budget_limits,
            team_spend,
            team_alias,
            team_tpm_limit,
            team_rpm_limit,
            team_tpd_limit,
            team_max_budget,
            team_soft_budget,
            team_model_max_budget,
            team_models,
            team_blocked,
            team_model_aliases,
            team_member,
            team_metadata,
            team_object_permission_id,
            team_member_spend,
            team_member_tpm_limit,
            team_member_rpm_limit,
            organization_alias,
            organization_max_budget,
            organization_tpm_limit,
            organization_rpm_limit,
            organization_metadata,
            project_alias,
        } = row;
        let team = TeamColumns {
            spend: team_spend.unwrap_or_default(),
            alias: team_alias,
            tpm_limit: team_tpm_limit,
            rpm_limit: team_rpm_limit,
            tpd_limit: team_tpd_limit,
            max_budget: team_max_budget,
            soft_budget: team_soft_budget,
            model_max_budget: team_model_max_budget,
            models: team_models,
            blocked: team_blocked,
            model_aliases: team_model_aliases,
            member: team_member,
            metadata: team_metadata,
            object_permission_id: team_object_permission_id,
        };
        let team_member = TeamMemberColumns {
            spend: team_member_spend.unwrap_or_default(),
            tpm_limit: team_member_tpm_limit,
            rpm_limit: team_member_rpm_limit,
        };
        let is_in_organization = organization_alias.is_some();
        let organization = OrganizationColumns {
            alias: organization_alias.unwrap_or_default(),
            max_budget: organization_max_budget,
            tpm_limit: organization_tpm_limit,
            rpm_limit: organization_rpm_limit,
            metadata: organization_metadata,
        };
        Ok(Self {
            key: KeyColumns {
                token,
                key_name,
                key_alias,
                soft_budget_cooldown,
                spend,
                total_spend,
                expires,
                models,
                aliases,
                config,
                router_settings,
                user_id,
                team_id,
                agent_id,
                project_id,
                permissions,
                max_parallel_requests,
                metadata,
                blocked,
                tpm_limit,
                rpm_limit,
                tpd_limit,
                max_budget,
                soft_budget,
                budget_duration,
                budget_reset_at,
                allowed_cache_controls,
                allowed_routes,
                key_type,
                access_group_ids,
                model_spend,
                model_max_budget,
                budget_fallbacks,
                budget_id,
                org_id,
                object_permission_id,
                created_at,
                created_by,
                updated_at,
                updated_by,
                settings_updated_at,
                last_active,
                rotation_count,
                auto_rotate,
                rotation_interval,
                last_rotation_at,
                key_rotation_at,
                budget_limits,
            },
            team: joined("team", "team_spend", team_spend.is_some(), team)?,
            team_member: joined(
                "team member",
                "team_member_spend",
                team_member_spend.is_some(),
                team_member,
            )?,
            organization: joined(
                "organization",
                "organization_alias",
                is_in_organization,
                organization,
            )?,
            project: ProjectColumns {
                alias: project_alias,
            },
        })
    }
}

impl From<KeyView> for KeyViewRow {
    fn from(view: KeyView) -> Self {
        let KeyView {
            key,
            team,
            team_member,
            organization,
            project,
        } = view;
        let KeyColumns {
            token,
            key_name,
            key_alias,
            soft_budget_cooldown,
            spend,
            total_spend,
            expires,
            models,
            aliases,
            config,
            router_settings,
            user_id,
            team_id,
            agent_id,
            project_id,
            permissions,
            max_parallel_requests,
            metadata,
            blocked,
            tpm_limit,
            rpm_limit,
            tpd_limit,
            max_budget,
            soft_budget,
            budget_duration,
            budget_reset_at,
            allowed_cache_controls,
            allowed_routes,
            key_type,
            access_group_ids,
            model_spend,
            model_max_budget,
            budget_fallbacks,
            budget_id,
            org_id,
            object_permission_id,
            created_at,
            created_by,
            updated_at,
            updated_by,
            settings_updated_at,
            last_active,
            rotation_count,
            auto_rotate,
            rotation_interval,
            last_rotation_at,
            key_rotation_at,
            budget_limits,
        } = key;
        let team_spend = team.as_ref().map(|team| team.spend);
        let team = team.unwrap_or_default();
        let team_member_spend = team_member.as_ref().map(|member| member.spend);
        let team_member = team_member.unwrap_or_default();
        let (organization_alias, organization) = match organization {
            Some(organization) => (Some(organization.alias.clone()), organization),
            None => (None, OrganizationColumns::default()),
        };
        Self {
            token,
            key_name,
            key_alias,
            soft_budget_cooldown,
            spend,
            total_spend,
            expires,
            models,
            aliases,
            config,
            router_settings,
            user_id,
            team_id,
            agent_id,
            project_id,
            permissions,
            max_parallel_requests,
            metadata,
            blocked,
            tpm_limit,
            rpm_limit,
            tpd_limit,
            max_budget,
            soft_budget,
            budget_duration,
            budget_reset_at,
            allowed_cache_controls,
            allowed_routes,
            key_type,
            access_group_ids,
            model_spend,
            model_max_budget,
            budget_fallbacks,
            budget_id,
            org_id,
            object_permission_id,
            created_at,
            created_by,
            updated_at,
            updated_by,
            settings_updated_at,
            last_active,
            rotation_count,
            auto_rotate,
            rotation_interval,
            last_rotation_at,
            key_rotation_at,
            budget_limits,
            team_spend,
            team_alias: team.alias,
            team_tpm_limit: team.tpm_limit,
            team_rpm_limit: team.rpm_limit,
            team_tpd_limit: team.tpd_limit,
            team_max_budget: team.max_budget,
            team_soft_budget: team.soft_budget,
            team_model_max_budget: team.model_max_budget,
            team_models: team.models,
            team_blocked: team.blocked,
            team_model_aliases: team.model_aliases,
            team_member: team.member,
            team_metadata: team.metadata,
            team_object_permission_id: team.object_permission_id,
            team_member_spend,
            team_member_tpm_limit: team_member.tpm_limit,
            team_member_rpm_limit: team_member.rpm_limit,
            organization_alias,
            organization_max_budget: organization.max_budget,
            organization_tpm_limit: organization.tpm_limit,
            organization_rpm_limit: organization.rpm_limit,
            organization_metadata: organization.metadata,
            project_alias: project.alias,
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::{KeyView, OrganizationColumns, TeamColumns, TeamMemberColumns};
    use crate::test_support::PLAIN_KEY_JSON;

    /// The flat object of a key in a team and an organization, with a membership and a project.
    fn joined_everything() -> Value {
        with(json!({
            "team_id": "team-1", "user_id": "user-1", "project_id": "project-1", "org_id": "org-1",
            "team_spend": 1.5, "team_alias": "Team", "team_tpm_limit": 10, "team_rpm_limit": 20,
            "team_tpd_limit": 30, "team_max_budget": 40.5, "team_soft_budget": 4.5,
            "team_model_max_budget": {"gpt-4o": 1}, "team_models": ["gpt-4o", "all-proxy-models"],
            "team_blocked": true, "team_model_aliases": {"fast": "gpt-4o"},
            "team_member": {"role": "admin", "user_id": "user-1"}, "team_metadata": {"tier": "gold"},
            "team_object_permission_id": "permission-1",
            "team_member_spend": 0.25, "team_member_tpm_limit": 5, "team_member_rpm_limit": 6,
            "organization_alias": "Org", "organization_max_budget": 100.0, "organization_tpm_limit": 7,
            "organization_rpm_limit": 8, "organization_metadata": {"plan": "pro"},
            "project_alias": "Project"
        }))
    }

    fn changed(mut flat: Value, change: Value) -> Value {
        flat.as_object_mut()
            .unwrap()
            .extend(change.as_object().unwrap().clone());
        flat
    }

    fn with(change: Value) -> Value {
        changed(serde_json::from_str(PLAIN_KEY_JSON).unwrap(), change)
    }

    /// `flat` with each time written the way this crate writes one, which is not the way pydantic does.
    fn with_times_as_written(flat: Value) -> Value {
        changed(
            flat,
            json!({
                "created_at": "2026-11-19T10:00:00.123+00:00",
                "updated_at": "2026-11-19T10:00:00.123+00:00"
            }),
        )
    }

    #[rstest]
    #[case::a_key_that_joined_nothing(with(json!({})))]
    #[case::a_key_that_joined_everything(joined_everything())]
    #[case::a_team_that_holds_only_its_spend(with(json!({"team_spend": 0.0})))]
    #[case::a_membership_without_a_team(with(json!({"team_member_spend": 0.0})))]
    #[case::an_organization_that_holds_only_its_alias(with(json!({"organization_alias": ""})))]
    fn the_flat_object_reads_into_groups_and_writes_back_unchanged(#[case] flat: Value) {
        let view: KeyView = serde_json::from_value(flat.clone()).unwrap();

        assert_eq!(
            serde_json::to_value(&view).unwrap(),
            with_times_as_written(flat)
        );
    }

    #[test]
    fn a_key_that_joined_nothing_has_no_group() {
        let view: KeyView = serde_json::from_value(with(json!({}))).unwrap();

        assert_eq!(
            (
                view.team,
                view.team_member,
                view.organization,
                view.project.alias
            ),
            (None, None, None, None)
        );
    }

    #[test]
    fn each_column_lands_in_the_group_of_the_object_it_was_read_from() {
        let view: KeyView = serde_json::from_value(joined_everything()).unwrap();
        let raw = |value: Value| Some(serde_json::from_value(value).unwrap());

        assert_eq!(
            (
                view.key.team_id.as_deref(),
                view.team,
                view.team_member,
                view.organization,
                view.project.alias.as_deref(),
            ),
            (
                Some("team-1"),
                Some(TeamColumns {
                    spend: 1.5,
                    alias: Some("Team".to_owned()),
                    tpm_limit: Some(10),
                    rpm_limit: Some(20),
                    tpd_limit: Some(30),
                    max_budget: Some(40.5),
                    soft_budget: Some(4.5),
                    model_max_budget: raw(json!({"gpt-4o": 1})),
                    models: vec!["gpt-4o".to_owned(), "all-proxy-models".to_owned()],
                    blocked: true,
                    model_aliases: raw(json!({"fast": "gpt-4o"})),
                    member: raw(json!({"role": "admin", "user_id": "user-1"})),
                    metadata: raw(json!({"tier": "gold"})),
                    object_permission_id: Some("permission-1".to_owned()),
                }),
                Some(TeamMemberColumns {
                    spend: 0.25,
                    tpm_limit: Some(5),
                    rpm_limit: Some(6),
                }),
                Some(OrganizationColumns {
                    alias: "Org".to_owned(),
                    max_budget: Some(100.0),
                    tpm_limit: Some(7),
                    rpm_limit: Some(8),
                    metadata: raw(json!({"plan": "pro"})),
                }),
                Some("Project"),
            )
        );
    }

    #[rstest]
    #[case::a_blocked_team(json!({"team_blocked": true}), "team")]
    #[case::team_models(json!({"team_models": ["gpt-4o"]}), "team")]
    #[case::a_team_alias(json!({"team_alias": "Team"}), "team")]
    #[case::a_roster_entry(json!({"team_member": {"user_id": "user-1"}}), "team")]
    #[case::a_member_limit(json!({"team_member_rpm_limit": 1}), "team member")]
    #[case::an_organization_limit(json!({"organization_max_budget": 1.0}), "organization")]
    fn a_group_column_without_its_marker_is_refused(#[case] change: Value, #[case] group: &str) {
        let error = serde_json::from_value::<KeyView>(with(change)).unwrap_err();

        assert!(
            error
                .to_string()
                .starts_with(&format!("the key row holds {group} columns")),
            "{error}"
        );
    }

    #[rstest]
    #[case::a_field_the_row_does_not_name(json!({"future_limit": 1}))]
    #[case::a_group_name(json!({"team": {"spend": 1.0}}))]
    #[case::a_column_of_another_type(json!({"spend": "free"}))]
    #[case::a_time_it_cannot_read(json!({"expires": "tomorrow"}))]
    fn a_flat_object_that_is_not_the_row_is_refused(#[case] change: Value) {
        assert!(serde_json::from_value::<KeyView>(with(change)).is_err());
    }
}
