mod deployment;

use std::collections::HashMap;

use litellm_config::{Config, Spelled};
use litellm_core_utils::{
    serde_compat::{parse_pydantic_bool, parse_str_bool},
    settings::Lookup,
};
use litellm_inference_messages::{MessagesSettings, MessagesShaping};
use litellm_llms::anthropic::model_capabilities::anthropic_model_capabilities;
use litellm_model_catalog::Catalog;
use litellm_router_types::{ExtraParams, LitellmParams};
use serde_json::{Map, Value};

pub use deployment::Deployment;

#[derive(Clone, Debug, Default)]
pub struct Router(HashMap<String, Deployment>);

fn is_on(flag: Option<&Spelled<bool>>, env: &impl Lookup) -> bool {
    match flag {
        Some(Spelled::Value(on)) => *on,
        Some(Spelled::Text(text)) => match text.strip_prefix("os.environ/") {
            Some(name) => env
                .get(name)
                .is_some_and(|value| parse_pydantic_bool(&value) == Some(true)),
            None => parse_pydantic_bool(text) == Some(true),
        },
        None => false,
    }
}

fn environment_value(name: &str, env: &impl Lookup) -> Value {
    env.get(name).map_or(Value::Null, |value| {
        parse_str_bool(&value).map_or(Value::String(value), Value::Bool)
    })
}

fn with_environment(value: &Value, env: &impl Lookup) -> Value {
    match value {
        Value::String(text) => text
            .strip_prefix("os.environ/")
            .map_or_else(|| value.clone(), |name| environment_value(name, env)),
        Value::Object(fields) => Value::Object(fields_with_environment(fields, env)),
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|item| match item {
                    Value::Object(fields) => Value::Object(fields_with_environment(fields, env)),
                    _ => item.clone(),
                })
                .collect(),
        ),
        _ => value.clone(),
    }
}

fn fields_with_environment(fields: &Map<String, Value>, env: &impl Lookup) -> Map<String, Value> {
    fields
        .iter()
        .map(|(name, value)| (name.clone(), with_environment(value, env)))
        .collect()
}

fn additional_drop_params(extra: &ExtraParams) -> Vec<String> {
    extra
        .get("additional_drop_params")
        .and_then(Value::as_array)
        .map(|names| {
            names
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

impl Router {
    pub fn from_config(config: &Config, env: &impl Lookup, catalog: &Catalog) -> Self {
        let drops_params_globally = is_on(config.litellm_settings.drop_params.as_ref(), env);
        config
            .model_list
            .iter()
            .map(|model| {
                let drop_params =
                    drops_params_globally || is_on(model.litellm_params.drop_params.as_ref(), env);
                let litellm_params = LitellmParams {
                    extra: model
                        .litellm_params
                        .extra
                        .iter()
                        .map(|(name, value)| (name.clone(), with_environment(value, env)))
                        .collect(),
                    ..model.litellm_params.clone()
                };
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
                        shaping: MessagesShaping {
                            settings: MessagesSettings {
                                drop_params,
                                drops_unsupported_params: drop_params,
                                additional_drop_params: additional_drop_params(
                                    &litellm_params.extra,
                                ),
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
    use serde_json::{Value, json};

    use super::Router;

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
        let config = Config::from_yaml(&format!(
            "{settings}model_list:\n  - model_name: public-model\n    litellm_params:\n      model: provider/model{deployment}"
        ))
        .unwrap();

        let settings = Router::from_config(
            &config,
            &|name: &str| (name == "DROP_PARAMS_FLAG").then(|| " Yes ".to_owned()),
            &Catalog::parse(br#"{"provider-model": {"litellm_provider": "provider"}}"#)
                .unwrap()
                .value,
        )
        .get("public-model")
        .unwrap()
        .shaping
        .settings
        .clone();

        assert_eq!(
            (settings.drop_params, settings.drops_unsupported_params),
            (expected, expected)
        );
    }

    #[rstest]
    #[case::text(json!("os.environ/PROMPT"), json!("Answer in French."))]
    #[case::true_word(json!("os.environ/ON_WORD"), json!(true))]
    #[case::false_word(json!("os.environ/OFF_WORD"), json!(false))]
    #[case::unset(json!("os.environ/UNSET"), Value::Null)]
    #[case::plain(json!("Answer in French."), json!("Answer in French."))]
    #[case::nested_object(json!({"inner": "os.environ/PROMPT"}), json!({"inner": "Answer in French."}))]
    #[case::object_in_a_list(
        json!([{"authorization_token": "os.environ/PROMPT"}]),
        json!([{"authorization_token": "Answer in French."}])
    )]
    #[case::text_in_a_list(json!(["os.environ/PROMPT"]), json!(["os.environ/PROMPT"]))]
    fn deployment_params_resolve_os_environ_references_as_python_does(
        #[case] configured: Value,
        #[case] expected: Value,
    ) {
        let config = Config::from_yaml(&format!(
            "model_list:\n  - model_name: public-model\n    litellm_params:\n      model: provider/model\n      system: {configured}"
        ))
        .unwrap();
        let environment = |name: &str| match name {
            "PROMPT" => Some("Answer in French.".to_owned()),
            "ON_WORD" => Some(" True ".to_owned()),
            "OFF_WORD" => Some("false".to_owned()),
            _ => None,
        };

        let router = Router::from_config(
            &config,
            &environment,
            &Catalog::parse(br#"{"provider-model": {"litellm_provider": "provider"}}"#)
                .unwrap()
                .value,
        );

        assert_eq!(
            router.get("public-model").unwrap().litellm_params.extra["system"],
            expected
        );
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
            Router::from_config(
                &config,
                &|_: &str| None,
                &Catalog::parse(br#"{"provider-model": {"litellm_provider": "provider"}}"#)
                    .unwrap()
                    .value
            )
            .get("public-model")
            .unwrap()
            .shaping
            .capabilities
            .supports_reasoning,
            expected
        );
    }
}
