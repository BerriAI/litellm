//! A key's team as the auth cache holds it, read into the auth core's [`TeamRecord`]. Each field
//! is taken apart by name, so a field added to the cached team does not compile here until it is
//! honored or named as one the native app does not serve.

use litellm_gateway_auth_core::record::{Member, MemberRole, TeamRecord};
use litellm_python_compat::{Value, json};
use litellm_redis_store::auth_cache::{self, Bag, CachedMember, CachedTeam};
use serde::de::DeserializeOwned;

use super::key::Carried;

type Object = serde_json::Map<String, serde_json::Value>;

/// The native app honors a team's `models` list and nothing else of it. `Err` names the first
/// field that holds something Python enforces and the native app does not: the block, a budget, a
/// limit, an alias table, an access group, a permission, the organization above it or anything in
/// its metadata. The roster and the member defaults only reach a key with an owner, which the
/// owner's own step declines, and whether a request may change guardrails is asked of a body
/// `admit` never admits. The rest only describes the team or what it spent.
fn unserved(team: &CachedTeam) -> Option<&'static str> {
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
    unread_models.then_some("models").or_else(|| {
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
    })
}

/// `value` as the JSON it is written as, or the name of the `column` that holds something JSON
/// cannot say.
fn plain<T: DeserializeOwned>(column: &'static str, value: &Value) -> Result<T, &'static str> {
    let text = json::dumps(value).map_err(|_| column)?;
    serde_json::from_str(&text).map_err(|_| column)
}

fn object(column: &'static str, bag: Option<&Bag>) -> Result<Option<Object>, &'static str> {
    bag.map(|bag| plain(column, &Value::Dict(bag.entries().to_vec())))
        .transpose()
}

fn member(member: &CachedMember) -> Member {
    Member {
        user_id: member.user_id.clone(),
        user_email: member.user_email.clone(),
        role: match member.role {
            auth_cache::MemberRole::Admin => MemberRole::Admin,
            auth_cache::MemberRole::User => MemberRole::User,
        },
    }
}

/// The team's row as the auth core reads it, whatever it holds. `Err` names a column whose value
/// is no JSON.
fn record(team: &CachedTeam) -> Result<TeamRecord, &'static str> {
    let aliases = team
        .litellm_model_table
        .as_ref()
        .and_then(|table| table.model_aliases.as_ref());
    Ok(TeamRecord {
        team_alias: team.team_alias.clone(),
        models: team.models.clone(),
        blocked: team.blocked,
        tpm_limit: team.tpm_limit,
        rpm_limit: team.rpm_limit,
        tpd_limit: team.tpd_limit,
        max_budget: team.max_budget,
        soft_budget: team.soft_budget,
        model_max_budget: object("model_max_budget", team.model_max_budget.as_ref())?,
        spend: team.spend,
        metadata: object("metadata", team.metadata.as_ref())?,
        model_aliases: aliases
            .map(|aliases| plain("litellm_model_table", aliases))
            .transpose()?,
        object_permission_id: team.object_permission_id.clone(),
        members_with_roles: team.members_with_roles.iter().map(member).collect(),
    })
}

/// `team` as the core takes it, or the first field of it the native app does not serve.
pub(super) fn read(team: &CachedTeam) -> Result<TeamRecord, &'static str> {
    match unserved(team) {
        Some(field) => Err(field),
        None => record(team),
    }
}

#[cfg(test)]
mod tests {
    use litellm_gateway_auth_core::projection::{self, TeamRowGrants};
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

    /// `LiteLLM_TeamTableCachedObj.model_dump(mode="json")` of a team that holds something in
    /// every column the auth core's record reads, with its alias table as the text a writer
    /// stored.
    const FULL: &str = r#"{
        "team_alias": "Full", "organization_id": "org",
        "members_with_roles": [
            {"user_id": "admin", "user_email": null, "role": "admin"},
            {"user_id": "member", "user_email": "first@example.com", "role": "user"},
            {"user_id": null, "user_email": "m@example.com", "role": "user"},
            {"user_id": "member", "user_email": "last@example.com", "role": "admin"}
        ],
        "metadata": {"guardrails": ["pii"]}, "tpm_limit": 1000, "rpm_limit": 10,
        "tpd_limit": 100000, "max_budget": 10.5, "soft_budget": 5.25,
        "models": ["claude", "all-proxy-models"], "blocked": true, "spend": 1.5,
        "model_max_budget": {"claude": {"budget_limit": 2.0}}, "model_id": 7,
        "litellm_model_table": {
            "id": 7, "model_aliases": "{\"gpt\": \"claude\"}", "created_by": "t",
            "updated_by": "t", "team": null
        },
        "object_permission_id": "permission"
    }"#;

    /// `_team_row_grants(LiteLLM_TeamTableCachedObj(**row), "member")` on BerriAI/litellm#53731,
    /// for the row [`FULL`] sets.
    #[rstest]
    fn a_cached_teams_record_sets_what_the_proxy_stamps_from_that_team() {
        let team = record(&cached(serde_json::from_str(FULL).unwrap())).unwrap();
        let model_max_budget = json!({"claude": {"budget_limit": 2.0}});
        let metadata = json!({"guardrails": ["pii"]});
        let models = ["claude".to_owned(), "all-proxy-models".to_owned()];
        let member = Member {
            user_id: Some("member".to_owned()),
            user_email: Some("last@example.com".to_owned()),
            role: MemberRole::Admin,
        };

        assert_eq!(
            projection::team_row(&team, Some("member")),
            TeamRowGrants {
                team_alias: Some("Full"),
                team_tpm_limit: Some(1000),
                team_rpm_limit: Some(10),
                team_tpd_limit: Some(100_000),
                team_max_budget: Some(10.5),
                team_soft_budget: Some(5.25),
                team_model_max_budget: model_max_budget.as_object(),
                team_spend: Some(1.5),
                team_models: &models,
                team_blocked: true,
                team_metadata: metadata.as_object(),
                team_model_aliases: Some([("gpt".to_owned(), "claude".to_owned())].into()),
                team_object_permission_id: Some("permission"),
                team_member: Some(&member),
            }
        );
    }

    #[rstest]
    fn a_team_python_cached_with_a_model_list_is_read_with_that_list() {
        let team = read(&cached(json!({}))).unwrap();

        assert_eq!(team.models, ["claude".to_owned()]);
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
