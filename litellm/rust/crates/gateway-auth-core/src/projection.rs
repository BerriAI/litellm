//! What a row a caller belongs to sets on the caller, the counterpart of the proxy's
//! `team_grants.py` and `grants.py`. Each function reads one record and returns the values the
//! proxy stamps on `UserAPIKeyAuth`, under the names it stamps them, so every host and every path
//! takes the same values from the same row. Nothing here reads a host, a counter or a request.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

use crate::record::{Member, TeamRecord};

/// The proxy's `TeamRowGrants`: what a team's own row sets on a caller in that team.
// TODO: `team_object_permission` once the object permission is a core record
#[derive(Clone, Debug, PartialEq)]
pub struct TeamRowGrants<'a> {
    pub team_alias: Option<&'a str>,
    pub team_tpm_limit: Option<i64>,
    pub team_rpm_limit: Option<i64>,
    pub team_tpd_limit: Option<i64>,
    pub team_max_budget: Option<f64>,
    pub team_soft_budget: Option<f64>,
    pub team_model_max_budget: Option<&'a Map<String, Value>>,
    pub team_spend: Option<f64>,
    pub team_models: &'a [String],
    pub team_blocked: bool,
    pub team_metadata: Option<&'a Map<String, Value>>,
    pub team_model_aliases: Option<BTreeMap<String, String>>,
    pub team_object_permission_id: Option<&'a str>,
    pub team_member: Option<&'a Member>,
}

/// `team` as a caller in it carries it, with `user_id`'s entry in its roster.
pub fn team_row<'a>(team: &'a TeamRecord, user_id: Option<&str>) -> TeamRowGrants<'a> {
    TeamRowGrants {
        team_alias: team.team_alias.as_deref(),
        team_tpm_limit: team.tpm_limit,
        team_rpm_limit: team.rpm_limit,
        team_tpd_limit: team.tpd_limit,
        team_max_budget: team.max_budget,
        team_soft_budget: team.soft_budget,
        team_model_max_budget: team.model_max_budget.as_ref(),
        team_spend: team.spend,
        team_models: &team.models,
        team_blocked: team.blocked,
        team_metadata: team.metadata.as_ref(),
        team_model_aliases: team.model_aliases.as_ref().and_then(model_aliases),
        team_object_permission_id: team.object_permission_id.as_deref(),
        team_member: user_id.and_then(|user_id| member(&team.members_with_roles, user_id)),
    }
}

/// A user listed twice gets the last entry, which is the one the key read's SQL join picks.
fn member<'a>(roster: &'a [Member], user_id: &str) -> Option<&'a Member> {
    roster
        .iter()
        .rfind(|member| member.user_id.as_deref() == Some(user_id))
}

/// The alias table as a map from a name to the model it stands for. A column that is no such
/// map, one entry that names no model included, aliases nothing.
fn model_aliases(column: &Value) -> Option<BTreeMap<String, String>> {
    match column {
        Value::String(text) => serde_json::from_str(text).ok(),
        Value::Object(entries) => entries
            .iter()
            .map(|(name, model)| Some((name.clone(), model.as_str()?.to_owned())))
            .collect(),
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::Array(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::record::MemberRole;

    fn member_of(user_id: Option<&str>, user_email: Option<&str>, role: MemberRole) -> Member {
        Member {
            user_id: user_id.map(str::to_owned),
            user_email: user_email.map(str::to_owned),
            role,
        }
    }

    fn object(value: Value) -> Map<String, Value> {
        value.as_object().unwrap().clone()
    }

    /// A team that holds something different in every column the projection reads.
    fn full() -> TeamRecord {
        TeamRecord {
            team_alias: Some("Full".to_owned()),
            models: vec!["claude".to_owned(), "all-proxy-models".to_owned()],
            blocked: true,
            tpm_limit: Some(1000),
            rpm_limit: Some(10),
            tpd_limit: Some(100_000),
            max_budget: Some(10.5),
            soft_budget: Some(5.25),
            model_max_budget: Some(object(json!({"claude": {"budget_limit": 2.0}}))),
            spend: Some(1.5),
            metadata: Some(object(json!({"guardrails": ["pii"]}))),
            model_aliases: Some(json!({"gpt": "claude"})),
            object_permission_id: Some("permission".to_owned()),
            members_with_roles: vec![
                member_of(Some("admin"), None, MemberRole::Admin),
                member_of(Some("member"), Some("first@example.com"), MemberRole::User),
                member_of(None, Some("m@example.com"), MemberRole::User),
                member_of(Some("member"), Some("last@example.com"), MemberRole::Admin),
            ],
        }
    }

    /// `_team_row_grants(LiteLLM_TeamTable(**row), "member")` on BerriAI/litellm#53731, for the
    /// row [`full`] holds.
    #[rstest]
    fn a_teams_row_sets_what_the_proxy_stamps_from_it() {
        let team = full();
        let model_max_budget = object(json!({"claude": {"budget_limit": 2.0}}));
        let metadata = object(json!({"guardrails": ["pii"]}));
        let models = ["claude".to_owned(), "all-proxy-models".to_owned()];
        let member = member_of(Some("member"), Some("last@example.com"), MemberRole::Admin);

        assert_eq!(
            team_row(&team, Some("member")),
            TeamRowGrants {
                team_alias: Some("Full"),
                team_tpm_limit: Some(1000),
                team_rpm_limit: Some(10),
                team_tpd_limit: Some(100_000),
                team_max_budget: Some(10.5),
                team_soft_budget: Some(5.25),
                team_model_max_budget: Some(&model_max_budget),
                team_spend: Some(1.5),
                team_models: &models,
                team_blocked: true,
                team_metadata: Some(&metadata),
                team_model_aliases: Some([("gpt".to_owned(), "claude".to_owned())].into()),
                team_object_permission_id: Some("permission"),
                team_member: Some(&member),
            }
        );
    }

    #[rstest]
    #[case::no_user(None)]
    #[case::a_user_the_roster_does_not_list(Some("stranger"))]
    fn a_caller_the_roster_does_not_name_carries_no_entry(#[case] user_id: Option<&str>) {
        let team = full();

        assert_eq!(team_row(&team, user_id).team_member, None);
    }

    #[rstest]
    #[case::the_text_of_an_object(json!(r#"{"gpt": "claude"}"#), Some(vec![("gpt", "claude")]))]
    #[case::an_empty_object(json!({}), Some(vec![]))]
    #[case::text_that_is_cut_short(json!(r#"{"gpt": "#), None)]
    #[case::text_of_an_entry_that_names_no_model(json!(r#"{"gpt": 1}"#), None)]
    #[case::an_entry_that_names_no_model_beside_one_that_does(
        json!({"gpt": 1, "ok": "claude"}),
        None
    )]
    #[case::a_list(json!(["gpt"]), None)]
    #[case::null(json!(null), None)]
    fn an_alias_column_is_a_map_of_names_or_aliases_nothing(
        #[case] column: Value,
        #[case] expected: Option<Vec<(&str, &str)>>,
    ) {
        let team = TeamRecord {
            model_aliases: Some(column),
            ..TeamRecord::default()
        };

        let expected = expected.map(|entries| {
            entries
                .into_iter()
                .map(|(name, model)| (name.to_owned(), model.to_owned()))
                .collect()
        });
        assert_eq!(team_row(&team, None).team_model_aliases, expected);
    }
}
