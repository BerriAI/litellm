use std::fmt;

use litellm_router_types::LitellmParams;
use serde::Deserialize;

use crate::{AdditionalFields, Object};

#[derive(Clone, Deserialize)]
pub struct Model {
    pub model_name: String,
    pub litellm_params: LitellmParams,
    #[serde(default)]
    pub model_info: Object,
    pub blocked: Option<bool>,
    #[serde(flatten)]
    pub additional_fields: AdditionalFields,
}

impl fmt::Debug for Model {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Model")
            .field("model_name", &self.model_name)
            .field("litellm_params", &self.litellm_params)
            .field("model_info", &self.model_info)
            .field("blocked", &self.blocked)
            .field("additional_fields", &self.additional_fields.keys())
            .finish()
    }
}
