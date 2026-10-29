mod deployment;

use std::collections::HashMap;

use litellm_config::{Config, Spelled};
use litellm_core_utils::{serde_compat::parse_pydantic_bool, settings::Lookup};
use litellm_inference_messages::{MessagesSettings, MessagesShaping};

pub use deployment::{Deployment, REQUEST_SETTINGS};

#[derive(Clone, Debug, Default)]
pub struct Router(HashMap<String, Deployment>);

fn resolve(text: &str, env: &impl Lookup) -> Option<String> {
    match text.strip_prefix("os.environ/") {
        Some(name) => env.get(name),
        None => Some(text.to_owned()),
    }
}

fn is_on(flag: &Spelled<bool>, env: &impl Lookup) -> bool {
    match flag {
        Spelled::Value(on) => *on,
        Spelled::Text(text) => {
            resolve(text, env).is_some_and(|value| parse_pydantic_bool(&value) == Some(true))
        }
    }
}

fn seconds(timeout: &Spelled<f64>, env: &impl Lookup) -> Option<f64> {
    match timeout {
        Spelled::Value(seconds) => Some(*seconds),
        Spelled::Text(text) => resolve(text, env)?.trim().parse().ok(),
    }
}

fn drops_params_globally(config: &Config, env: &impl Lookup) -> bool {
    match &config.litellm_settings.drop_params {
        Some(flag) => is_on(flag, env),
        None => env
            .get("LITELLM_DROP_PARAMS")
            .filter(|value| !value.trim().is_empty())
            .is_some_and(|value| parse_pydantic_bool(&value) != Some(false)),
    }
}

fn summarizes_reasoning(config: &Config, env: &impl Lookup) -> bool {
    config
        .litellm_settings
        .reasoning_auto_summary
        .as_ref()
        .is_some_and(|flag| is_on(flag, env))
        || env
            .get("LITELLM_REASONING_AUTO_SUMMARY")
            .is_some_and(|value| value.eq_ignore_ascii_case("true"))
}

fn request_timeout(config: &Config, env: &impl Lookup) -> Option<f64> {
    match &config.litellm_settings.request_timeout {
        Some(timeout) => seconds(timeout, env),
        None => env.parsed("REQUEST_TIMEOUT"),
    }
}

impl Router {
    pub fn from_config(config: &Config, env: &impl Lookup) -> Self {
        let drops_params_globally = drops_params_globally(config, env);
        let reasoning_auto_summary = summarizes_reasoning(config, env);
        let request_timeout = request_timeout(config, env);
        config
            .model_list
            .iter()
            .map(|model| {
                (
                    model.model_name.clone(),
                    Deployment {
                        model: model.litellm_params.model.clone(),
                        api_key: model
                            .litellm_params
                            .api_key
                            .as_ref()
                            .map(|value| value.expose().to_string()),
                        api_base: model.litellm_params.api_base.clone(),
                        custom_llm_provider: model.litellm_params.custom_llm_provider.clone(),
                        litellm_params: model.litellm_params.clone(),
                        drops_params_globally,
                        shaping: MessagesShaping {
                            settings: MessagesSettings {
                                drop_params: drops_params_globally
                                    || model
                                        .litellm_params
                                        .drop_params
                                        .as_ref()
                                        .is_some_and(|flag| is_on(flag, env)),
                                reasoning_auto_summary,
                                additional_drop_params: model
                                    .litellm_params
                                    .extra
                                    .get("additional_drop_params")
                                    .map(deployment::drop_paths)
                                    .unwrap_or_default(),
                                request_timeout,
                            },
                            ..MessagesShaping::default()
                        },
                        ..Deployment::default()
                    },
                )
            })
            .collect()
    }

    pub fn get(&self, model_name: &str) -> Option<&Deployment> {
        self.0.get(model_name)
    }
}

impl FromIterator<(String, Deployment)> for Router {
    fn from_iter<I: IntoIterator<Item = (String, Deployment)>>(entries: I) -> Self {
        Self(entries.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use litellm_config::Config;
    use rstest::rstest;
    use serde_json::{Map, Value, json};

    use super::{Deployment, Router};

    fn configured(settings: &str, params: &str, env: &[(&str, &str)]) -> Deployment {
        let config = Config::from_yaml(&format!(
            "{settings}model_list:\n  - model_name: public-model\n    litellm_params:\n      model: provider/model{params}"
        ))
        .unwrap();
        Router::from_config(&config, &|name: &str| {
            env.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| (*value).to_owned())
        })
        .get("public-model")
        .unwrap()
        .clone()
    }

    fn request(body: Value) -> Map<String, Value> {
        body.as_object().unwrap().clone()
    }

    #[rstest]
    #[case::unset("", "", false)]
    #[case::deployment_flag("", "\n      drop_params: true", true)]
    #[case::deployment_word("", "\n      drop_params: \" True \"", true)]
    #[case::deployment_off("", "\n      drop_params: false", false)]
    #[case::deployment_unknown_word("", "\n      drop_params: maybe", false)]
    #[case::deployment_yes("", "\n      drop_params: yes", true)]
    #[case::deployment_no("", "\n      drop_params: \"no\"", false)]
    #[case::deployment_env_reference("", "\n      drop_params: os.environ/DROP_PARAMS_FLAG", true)]
    #[case::deployment_unset_env_reference(
        "",
        "\n      drop_params: os.environ/MISSING_FLAG",
        false
    )]
    #[case::global_env_reference(
        "litellm_settings:\n  drop_params: os.environ/DROP_PARAMS_FLAG\n",
        "",
        true
    )]
    #[case::global_flag("litellm_settings:\n  drop_params: true\n", "", true)]
    #[case::global_word("litellm_settings:\n  drop_params: \"true\"\n", "", true)]
    #[case::global_off("litellm_settings:\n  drop_params: false\n", "", false)]
    #[case::global_over_deployment_off(
        "litellm_settings:\n  drop_params: true\n",
        "\n      drop_params: false",
        true
    )]
    fn configured_drop_params_reach_the_deployment(
        #[case] settings: &str,
        #[case] deployment: &str,
        #[case] expected: bool,
    ) {
        assert_eq!(
            configured(settings, deployment, &[("DROP_PARAMS_FLAG", " Yes ")])
                .shaping
                .settings
                .drop_params,
            expected
        );
    }

    #[rstest]
    #[case::unset("", None, false)]
    #[case::blank("", Some("  "), false)]
    #[case::on("", Some("true"), true)]
    #[case::off_word("", Some(" No "), false)]
    #[case::unknown_word_is_on("", Some("maybe"), true)]
    #[case::settings_off_over_env_on(
        "litellm_settings:\n  drop_params: false\n",
        Some("true"),
        false
    )]
    #[case::settings_on_over_env_off(
        "litellm_settings:\n  drop_params: true\n",
        Some("false"),
        true
    )]
    fn drop_params_env_flag_applies_when_the_config_leaves_it_unset(
        #[case] settings: &str,
        #[case] flag: Option<&str>,
        #[case] expected: bool,
    ) {
        let env: Vec<(&str, &str)> = flag
            .map(|value| ("LITELLM_DROP_PARAMS", value))
            .into_iter()
            .collect();
        let deployment = configured(settings, "", &env);

        assert_eq!(deployment.shaping.settings.drop_params, expected);
        assert_eq!(
            deployment
                .shaping_for(&request(json!({"drop_params": false})))
                .settings
                .drop_params,
            expected
        );
    }

    #[rstest]
    #[case::unset("", None, false)]
    #[case::settings_flag("litellm_settings:\n  reasoning_auto_summary: true\n", None, true)]
    #[case::settings_env_reference(
        "litellm_settings:\n  reasoning_auto_summary: os.environ/SUMMARY_FLAG\n",
        None,
        true
    )]
    #[case::settings_off("litellm_settings:\n  reasoning_auto_summary: false\n", None, false)]
    #[case::env_flag("", Some("TRUE"), true)]
    #[case::env_off("", Some("false"), false)]
    #[case::env_on_over_settings_off(
        "litellm_settings:\n  reasoning_auto_summary: false\n",
        Some("true"),
        true
    )]
    fn reasoning_auto_summary_comes_from_settings_or_env(
        #[case] settings: &str,
        #[case] flag: Option<&str>,
        #[case] expected: bool,
    ) {
        let env: Vec<(&str, &str)> = [("SUMMARY_FLAG", "yes")]
            .into_iter()
            .chain(flag.map(|value| ("LITELLM_REASONING_AUTO_SUMMARY", value)))
            .collect();

        assert_eq!(
            configured(settings, "", &env)
                .shaping
                .settings
                .reasoning_auto_summary,
            expected
        );
    }

    #[rstest]
    #[case::unset("", None, None)]
    #[case::settings_number("litellm_settings:\n  request_timeout: 2\n", None, Some(2.0))]
    #[case::settings_text("litellm_settings:\n  request_timeout: \" 2.5 \"\n", None, Some(2.5))]
    #[case::settings_env_reference(
        "litellm_settings:\n  request_timeout: os.environ/TIMEOUT_SECONDS\n",
        None,
        Some(45.0)
    )]
    #[case::env("", Some("30"), Some(30.0))]
    #[case::unparsable_env("", Some("soon"), None)]
    #[case::settings_over_env("litellm_settings:\n  request_timeout: 2\n", Some("30"), Some(2.0))]
    fn request_timeout_comes_from_settings_or_env(
        #[case] settings: &str,
        #[case] seconds: Option<&str>,
        #[case] expected: Option<f64>,
    ) {
        let env: Vec<(&str, &str)> = [("TIMEOUT_SECONDS", "45")]
            .into_iter()
            .chain(seconds.map(|value| ("REQUEST_TIMEOUT", value)))
            .collect();

        assert_eq!(
            configured(settings, "", &env)
                .shaping
                .settings
                .request_timeout,
            expected
        );
    }

    #[rstest]
    #[case::unset("", &[])]
    #[case::paths(
        "\n      additional_drop_params: [\"temperature\", 3, \"tools[*].input_examples\"]",
        &["temperature", "tools[*].input_examples"]
    )]
    #[case::not_a_list("\n      additional_drop_params: temperature", &[])]
    fn deployment_additional_drop_params_keep_their_string_paths(
        #[case] params: &str,
        #[case] expected: &[&str],
    ) {
        assert_eq!(
            configured("", params, &[])
                .shaping
                .settings
                .additional_drop_params,
            expected
        );
    }

    #[rstest]
    #[case::not_requested("", "", json!({}), false)]
    #[case::requested("", "", json!({"drop_params": true}), true)]
    #[case::requested_word("", "", json!({"drop_params": " yes "}), true)]
    #[case::requested_one("", "", json!({"drop_params": 1}), true)]
    #[case::requested_two_is_not_a_flag("", "", json!({"drop_params": 2}), false)]
    #[case::requested_unknown_word("", "", json!({"drop_params": "maybe"}), false)]
    #[case::deployment_kept("", "\n      drop_params: true", json!({}), true)]
    #[case::request_off_over_deployment_on("", "\n      drop_params: true", json!({"drop_params": false}), false)]
    #[case::request_null_over_deployment_on("", "\n      drop_params: true", json!({"drop_params": null}), false)]
    #[case::global_on_over_request_off(
        "litellm_settings:\n  drop_params: true\n",
        "",
        json!({"drop_params": false}),
        true
    )]
    fn requested_drop_params_override_the_deployment_but_not_the_global_setting(
        #[case] settings: &str,
        #[case] params: &str,
        #[case] body: Value,
        #[case] expected: bool,
    ) {
        assert_eq!(
            configured(settings, params, &[])
                .shaping_for(&request(body))
                .settings
                .drop_params,
            expected
        );
    }

    #[rstest]
    #[case::deployment_kept(json!({}), &["temperature"])]
    #[case::request_replaces_deployment(json!({"additional_drop_params": ["top_k", false]}), &["top_k"])]
    #[case::request_not_a_list(json!({"additional_drop_params": "top_k"}), &[])]
    fn requested_additional_drop_params_replace_the_deployment_paths(
        #[case] body: Value,
        #[case] expected: &[&str],
    ) {
        let deployment = configured(
            "litellm_settings:\n  request_timeout: 2\n",
            "\n      additional_drop_params: [\"temperature\"]",
            &[],
        );
        let shaping = deployment.shaping_for(&request(body));

        assert_eq!(shaping.settings.additional_drop_params, expected);
        assert_eq!(shaping.settings.request_timeout, Some(2.0));
    }
}
