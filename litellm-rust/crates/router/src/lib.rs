mod deployment;

use std::collections::HashMap;

use litellm_config::{Config, Spelled};
use litellm_core_utils::{serde_compat::parse_pydantic_bool, settings::Lookup};
use litellm_inference_messages::{MessagesSettings, MessagesShaping};

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

impl Router {
    pub fn from_config(config: &Config, env: &impl Lookup) -> Self {
        let drops_params_globally = is_on(config.litellm_settings.drop_params.as_ref(), env);
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
                        shaping: MessagesShaping {
                            settings: MessagesSettings {
                                drop_params: drops_params_globally
                                    || is_on(model.litellm_params.drop_params.as_ref(), env),
                                ..MessagesSettings::default()
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

        assert_eq!(
            Router::from_config(&config, &|name: &str| (name == "DROP_PARAMS_FLAG")
                .then(|| " Yes ".to_owned()))
            .get("public-model")
            .unwrap()
            .shaping
            .settings
            .drop_params,
            expected
        );
    }
}
