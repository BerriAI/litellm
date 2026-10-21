mod deployment;

use std::collections::HashMap;

use litellm_config::Model;

pub use deployment::Deployment;

#[derive(Clone, Debug, Default)]
pub struct Router(HashMap<String, Deployment>);

impl Router {
    pub fn from_model_list(model_list: &[Model]) -> Self {
        model_list
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
