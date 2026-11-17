mod deployment;

use std::{collections::HashMap, time::Duration};

use litellm_config::{Config, Spelled, seconds};
use litellm_env::Lookup;
use litellm_inference_messages::{MessagesSettings, MessagesShaping};
use litellm_llms::anthropic::model_capabilities::anthropic_model_capabilities;
use litellm_model_catalog::Catalog;
use litellm_python_compat::flags::lax_bool;
use litellm_router_types::ExtraParams;

pub use deployment::{Deployment, MessagesTimeouts, REQUEST_SETTINGS};

#[derive(Clone, Debug, Default)]
pub struct Router(HashMap<String, Deployment>);

fn is_on(flag: Option<&Spelled<bool>>) -> bool {
    match flag {
        Some(Spelled::Value(on)) => *on,
        Some(Spelled::Text(text)) => lax_bool(&text.as_str().into()) == Some(true),
        None => false,
    }
}

fn duration(timeout: Option<&Spelled<f64>>) -> Option<Duration> {
    Duration::try_from_secs_f64(seconds(timeout?)?)
        .ok()
        .filter(|duration| !duration.is_zero())
}

fn additional_drop_params(extra: &ExtraParams) -> Vec<String> {
    extra
        .get("additional_drop_params")
        .map(deployment::drop_paths)
        .unwrap_or_default()
}

impl Router {
    pub fn from_config(config: &Config, catalog: &Catalog, env: &impl Lookup) -> Self {
        let settings = &config.litellm_settings;
        let drops_params_globally = settings.drops_params(env);
        let reasoning_auto_summary = settings.summarizes_reasoning(env);
        let request_timeout = settings.request_timeout_seconds(env);
        config
            .model_list
            .iter()
            .map(|model| {
                let drop_params =
                    drops_params_globally || is_on(model.litellm_params.drop_params.as_ref());
                let litellm_params = model.litellm_params.clone();
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
                        messages_timeouts: MessagesTimeouts {
                            timeout: duration(litellm_params.timeout.as_ref()),
                            stream_timeout: duration(litellm_params.stream_timeout.as_ref()),
                        },
                        drops_params_globally,
                        shaping: MessagesShaping {
                            settings: MessagesSettings {
                                drop_params,
                                drops_unsupported_params: drop_params,
                                reasoning_auto_summary,
                                additional_drop_params: additional_drop_params(
                                    &litellm_params.extra,
                                ),
                                request_timeout,
                                base_model: litellm_params
                                    .extra
                                    .get("base_model")
                                    .and_then(|value| value.as_str())
                                    .map(str::to_owned),
                                ..MessagesSettings::default()
                            },
                            capabilities: anthropic_model_capabilities(
                                catalog,
                                &model.litellm_params.model,
                                model.litellm_params.custom_llm_provider.as_deref(),
                                litellm_params
                                    .extra
                                    .get("base_model")
                                    .and_then(|value| value.as_str()),
                                &|capability| {
                                    model
                                        .model_info
                                        .get(capability)
                                        .and_then(|value| value.as_bool())
                                },
                            ),
                        },
                        litellm_params,
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
    use litellm_model_catalog::Catalog;
    use rstest::rstest;
    use serde_json::{Map, Value, json};

    use super::{Deployment, Router};

    fn catalog() -> Catalog {
        Catalog::parse(br#"{"provider-model": {"litellm_provider": "provider"}}"#)
            .unwrap()
            .value
    }

    fn configured(settings: &str, params: &str, env: &[(&str, &str)]) -> Deployment {
        let config = Config::from_yaml(&format!(
            "{settings}model_list:\n  - model_name: public-model\n    litellm_params:\n      model: provider/model{params}"
        ))
        .unwrap();
        Router::from_config(&config, &catalog(), &|name: &str| {
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
        let settings = configured(settings, deployment, &[]).shaping.settings;

        assert_eq!(
            (settings.drop_params, settings.drops_unsupported_params),
            (expected, expected)
        );
    }

    #[rstest]
    #[case::unset("", None, false)]
    #[case::on("", Some("true"), true)]
    #[case::settings_off_over_env_on(
        "litellm_settings:\n  drop_params: false\n",
        Some("true"),
        false
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

        let requested_off = deployment
            .shaping_for(&request(json!({"drop_params": false})))
            .settings;

        assert_eq!(deployment.shaping.settings.drop_params, expected);
        assert_eq!(
            (
                requested_off.drop_params,
                requested_off.drops_unsupported_params
            ),
            (expected, expected)
        );
    }

    #[rstest]
    #[case::unset("", &[], false, None)]
    #[case::from_settings(
        "litellm_settings:\n  reasoning_auto_summary: true\n  request_timeout: 2\n",
        &[],
        true,
        Some(2.0)
    )]
    #[case::from_env(
        "",
        &[("LITELLM_REASONING_AUTO_SUMMARY", "true"), ("REQUEST_TIMEOUT", "30")],
        true,
        Some(30.0)
    )]
    fn reasoning_summary_and_request_timeout_reach_the_deployment(
        #[case] settings: &str,
        #[case] env: &[(&str, &str)],
        #[case] summarizes: bool,
        #[case] request_timeout: Option<f64>,
    ) {
        let settings = configured(settings, "", env).shaping.settings;

        assert_eq!(
            (settings.reasoning_auto_summary, settings.request_timeout),
            (summarizes, request_timeout)
        );
    }

    #[rstest]
    #[case::nothing_set("", "", false, None)]
    #[case::global_only("litellm_settings:\n  request_timeout: 2\n", "", false, Some(2.0))]
    #[case::deployment_over_global(
        "litellm_settings:\n  request_timeout: 2\n",
        "\n      timeout: 120",
        false,
        Some(120.0)
    )]
    #[case::deployment_text("", "\n      timeout: \" 2.5 \"", false, Some(2.5))]
    #[case::deployment_zero_is_unset(
        "litellm_settings:\n  request_timeout: 2\n",
        "\n      timeout: 0",
        false,
        Some(2.0)
    )]
    #[case::deployment_negative_is_unset("", "\n      timeout: -1", false, None)]
    #[case::stream_timeout_ignored_without_stream("", "\n      stream_timeout: 30", false, None)]
    #[case::stream_timeout_over_global(
        "litellm_settings:\n  request_timeout: 2\n",
        "\n      timeout: 120\n      stream_timeout: 30",
        true,
        Some(30.0)
    )]
    #[case::global_over_deployment_timeout_when_streaming(
        "litellm_settings:\n  request_timeout: 2\n",
        "\n      timeout: 120",
        true,
        Some(2.0)
    )]
    #[case::deployment_timeout_when_streaming_with_nothing_else(
        "",
        "\n      timeout: 120",
        true,
        Some(120.0)
    )]
    fn messages_timeout_prefers_the_deployment_like_the_python_router(
        #[case] settings: &str,
        #[case] params: &str,
        #[case] stream: bool,
        #[case] expected: Option<f64>,
    ) {
        let deployment = configured(settings, params, &[]);

        assert_eq!(
            deployment
                .messages_timeouts
                .for_call(stream, deployment.shaping.settings.request_timeout()),
            expected.map(std::time::Duration::from_secs_f64)
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
        let shaped = configured(settings, params, &[])
            .shaping_for(&request(body))
            .settings;

        assert_eq!(
            (shaped.drop_params, shaped.drops_unsupported_params),
            (expected, expected)
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

    #[rstest]
    #[case::declared("\n    model_info:\n      supports_reasoning: true", true)]
    #[case::declared_off("\n    model_info:\n      supports_reasoning: false", false)]
    #[case::undeclared("", false)]
    fn model_info_flags_reach_the_deployment(#[case] model_info: &str, #[case] expected: bool) {
        let config = Config::from_yaml(&format!(
            "model_list:\n  - model_name: public-model\n    litellm_params:\n      model: anthropic/claude-unlisted-9{model_info}"
        ))
        .unwrap();

        assert_eq!(
            Router::from_config(&config, &catalog(), &|_: &str| None)
                .get("public-model")
                .unwrap()
                .shaping
                .capabilities
                .supports_reasoning,
            expected
        );
    }
}
