use litellm_router_types::LitellmParams;
use serde::Deserialize;
use veil::Redact;

use crate::{AdditionalFields, Object, Value};

/// The written `litellm_params` can carry credentials inline, so only whether they are set is
/// printed.
#[derive(Clone, Deserialize, Redact)]
#[serde(try_from = "Written")]
pub struct Model {
    pub model_name: String,
    pub litellm_params: LitellmParams,
    #[redact(fixed = 3)]
    pub written_params: Value,
    pub model_info: Object,
    pub blocked: Option<bool>,
    pub additional_fields: AdditionalFields,
}

#[derive(Deserialize)]
struct Written {
    model_name: String,
    litellm_params: Value,
    #[serde(default)]
    model_info: Object,
    blocked: Option<bool>,
    #[serde(flatten)]
    additional_fields: AdditionalFields,
}

impl TryFrom<Written> for Model {
    type Error = serde_yaml_ng::Error;

    fn try_from(written: Written) -> Result<Self, Self::Error> {
        Ok(Self {
            model_name: written.model_name,
            litellm_params: serde_yaml_ng::from_value(written.litellm_params.clone())?,
            written_params: written.litellm_params,
            model_info: written.model_info,
            blocked: written.blocked,
            additional_fields: written.additional_fields,
        })
    }
}
