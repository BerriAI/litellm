use litellm_router_types::LitellmParams;
use serde::Deserialize;

use crate::{AdditionalFields, Object};

#[derive(Clone, Debug, Deserialize)]
pub struct Model {
    pub model_name: String,
    pub litellm_params: LitellmParams,
    #[serde(default)]
    pub model_info: Object,
    pub blocked: Option<bool>,
    #[serde(flatten)]
    pub additional_fields: AdditionalFields,
}
