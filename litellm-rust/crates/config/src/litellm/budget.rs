use serde::Deserialize;
use std::collections::BTreeMap;

use litellm_auth_types::SecretValue;

use crate::{Raw, Spelled};

#[macro_rules_attribute::apply(crate::config_section)]
pub struct BudgetSettings {
    pub max_budget: Option<Spelled<f64>>,
    pub max_internal_user_budget: Option<Spelled<f64>>,
    pub default_max_internal_user_budget: Option<Spelled<f64>>,
    pub default_internal_user_params: Option<Raw>,
    pub max_ui_session_budget: Option<Spelled<f64>>,
    pub default_team_settings: Option<Box<[TeamDefaultSettings]>>,
    pub upperbound_key_generate_params: Option<UpperboundKeyGenerateParams>,
    pub key_alias_pattern: Option<String>,
    pub budget_reset_time: Option<String>,
    pub disable_end_user_cost_tracking: Option<Spelled<bool>>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TeamDefaultSettings {
    pub team_id: String,
    #[serde(default)]
    pub success_callback: Option<Box<[String]>>,
    #[serde(default)]
    pub failure_callback: Option<Box<[String]>>,
    #[serde(default)]
    pub callback_vars: Option<BTreeMap<String, CallbackVar>>,
    #[serde(flatten)]
    pub inline_callback_vars: BTreeMap<String, CallbackVar>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum CallbackVar {
    Bool(bool),
    Number(serde_yaml_ng::Number),
    Text(SecretValue),
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct UpperboundKeyGenerateParams {
    pub max_budget: Option<Spelled<f64>>,
    pub budget_duration: Option<String>,
    pub duration: Option<String>,
    pub max_parallel_requests: Option<Spelled<i64>>,
    pub tpm_limit: Option<Spelled<i64>>,
    pub rpm_limit: Option<Spelled<i64>>,
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use litellm_auth_types::SecretValue;
    use rstest::rstest;

    use super::{CallbackVar, TeamDefaultSettings};
    use crate::Config;

    fn teams(yaml: &str) -> Result<Box<[TeamDefaultSettings]>, crate::Error> {
        Config::from_yaml(&format!(
            "litellm_settings:\n  default_team_settings: {yaml}"
        ))
        .map(|config| {
            config
                .litellm_settings
                .budget
                .default_team_settings
                .unwrap()
        })
    }

    fn team(yaml: &str) -> TeamDefaultSettings {
        teams(yaml).unwrap()[0].clone()
    }

    #[rstest]
    fn named_keys_stay_out_of_inline_callback_vars() {
        let team = team(
            "[{team_id: t, success_callback: [langfuse], failure_callback: [otel], callback_vars: {k: v}, pk: x}]",
        );
        assert_eq!(team.team_id, "t");
        assert_eq!(
            team.success_callback.as_deref(),
            Some(&["langfuse".to_owned()][..])
        );
        assert_eq!(
            team.failure_callback.as_deref(),
            Some(&["otel".to_owned()][..])
        );
        assert_eq!(
            team.callback_vars,
            Some(BTreeMap::from([(
                "k".to_owned(),
                CallbackVar::Text(SecretValue::new("v"))
            )]))
        );
        assert_eq!(team.inline_callback_vars.keys().collect::<Vec<_>>(), ["pk"]);
    }

    #[rstest]
    #[case::text("pk", CallbackVar::Text(SecretValue::new("pk")))]
    #[case::reference("os.environ/PK", CallbackVar::Text(SecretValue::new("os.environ/PK")))]
    #[case::quoted_number("'5'", CallbackVar::Text(SecretValue::new("5")))]
    #[case::integer("5", CallbackVar::Number(5.into()))]
    #[case::float("0.5", CallbackVar::Number(0.5.into()))]
    #[case::bool("true", CallbackVar::Bool(true))]
    fn callback_vars_accept_scalars(#[case] yaml: &str, #[case] expected: CallbackVar) {
        let inline = team(&format!("[{{team_id: t, var: {yaml}}}]"));
        let nested = team(&format!("[{{team_id: t, callback_vars: {{var: {yaml}}}}}]"));
        assert_eq!(inline.inline_callback_vars.get("var"), Some(&expected));
        assert_eq!(nested.callback_vars.unwrap().get("var"), Some(&expected));
    }

    #[rstest]
    #[case::missing_team_id("[{success_callback: [langfuse]}]")]
    #[case::inline_list("[{team_id: t, var: [a]}]")]
    #[case::inline_mapping("[{team_id: t, var: {a: 1}}]")]
    #[case::inline_null("[{team_id: t, var: null}]")]
    #[case::nested_mapping("[{team_id: t, callback_vars: {var: {a: 1}}}]")]
    #[case::single_callback("[{team_id: t, success_callback: langfuse}]")]
    fn invalid_team_settings_fail_to_parse(#[case] yaml: &str) {
        assert!(teams(yaml).is_err());
    }
}
