mod deployment;

use std::collections::HashMap;

use litellm_config::{Config, Spelled};
use litellm_core_utils::serde_compat::parse_str_bool;
use litellm_inference_messages::{MessagesSettings, MessagesShaping};

pub use deployment::Deployment;

#[derive(Clone, Debug, Default)]
pub struct Router(HashMap<String, Deployment>);

fn is_on(flag: Option<&Spelled<bool>>) -> bool {
    match flag {
        Some(Spelled::Value(on)) => *on,
        Some(Spelled::Text(text)) => parse_str_bool(text) == Some(true),
        None => false,
    }
}

impl Router {
    pub fn from_config(config: &Config) -> Self {
        let drops_params_globally = is_on(config.litellm_settings.drop_params.as_ref());
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
                                    || is_on(model.litellm_params.drop_params.as_ref()),
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
