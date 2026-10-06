mod deployment;
mod hooks;

use std::{collections::HashMap, sync::Arc};

pub use deployment::Deployment;
pub use hooks::RouterHooks;
use litellm_config::Model;

#[derive(Clone)]
pub struct Router {
    deployments: HashMap<String, Deployment>,
    hooks: Arc<dyn RouterHooks>,
}

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
        self.deployments.get(model_name)
    }

    pub fn with_hooks(self, hooks: Arc<dyn RouterHooks>) -> Self {
        Self { hooks, ..self }
    }

    pub async fn select(&self, model_name: &str) -> Option<&Deployment> {
        let candidates = self.deployments.get(model_name).into_iter().collect();
        self.hooks
            .filter_deployments(model_name, candidates)
            .await
            .into_iter()
            .next()
    }
}

impl Default for Router {
    fn default() -> Self {
        Self::from_iter([])
    }
}

impl FromIterator<(String, Deployment)> for Router {
    fn from_iter<I: IntoIterator<Item = (String, Deployment)>>(entries: I) -> Self {
        Self {
            deployments: entries.into_iter().collect(),
            hooks: Arc::new(()),
        }
    }
}
