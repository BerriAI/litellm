mod deployment;

use std::collections::HashMap;

use litellm_config::{Config, Spelled};
use litellm_inference_messages::{MessagesSettings, MessagesShaping};
use litellm_llms::anthropic::model_capabilities::anthropic_model_capabilities;
use litellm_model_catalog::Catalog;
use litellm_python_compat::flags::lax_bool;
use litellm_router_types::ExtraParams;
use serde_json::Value;

pub use deployment::Deployment;

#[derive(Clone, Debug, Default)]
pub struct Router(HashMap<String, Deployment>);

fn is_on(flag: Option<&Spelled<bool>>) -> bool {
    match flag {
        Some(Spelled::Value(on)) => *on,
        Some(Spelled::Text(text)) => lax_bool(&text.as_str().into()) == Some(true),
        None => false,
    }
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
    pub fn from_config(config: &Config, catalog: &Catalog) -> Self {
        let drops_params_globally = is_on(config.litellm_settings.drop_params.as_ref());
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
                        shaping: MessagesShaping {
                            settings: MessagesSettings {
                                drop_params,
                                drops_unsupported_params: drop_params,
                                additional_drop_params: additional_drop_params(
                                    &litellm_params.extra,
                                ),
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

    use super::Router;

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
        let config = Config::from_yaml(&format!(
            "{settings}model_list:\n  - model_name: public-model\n    litellm_params:\n      model: provider/model{deployment}"
        ))
        .unwrap();

        let settings = Router::from_config(
            &config,
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
