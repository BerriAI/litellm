use std::time::Duration;

use litellm_core::messages::MessagesShaping;

#[derive(Clone, Debug, Default)]
pub struct Deployment {
    pub model: String,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub custom_llm_provider: Option<String>,
    pub timeout: Option<Duration>,
    pub shaping: MessagesShaping,
}

#[derive(Clone, Debug)]
pub(crate) struct CatalogEntry {
    pub id: String,
    pub model_name: String,
    pub deployment: Deployment,
}

impl CatalogEntry {
    pub fn candidate(&self) -> crate::selection::Candidate {
        crate::selection::Candidate {
            deployment_id: self.id.clone(),
            model_name: self.model_name.clone(),
            model: self.deployment.model.clone(),
        }
    }
}
