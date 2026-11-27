use serde_json::{Map, Value};

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ProviderSpecificHeader {
    #[serde(default)]
    pub custom_llm_provider: String,
    #[serde(default)]
    pub extra_headers: Map<String, Value>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub user_profile_id_models_only: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub non_user_profile_id_models_only: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub yields_to_configured: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub configured_credential_headers: Vec<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub configured_credential_prefix: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub caller_credentials: Vec<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub without_configured_credential: bool,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(untagged)]
pub enum ProviderSpecificHeaders {
    One(ProviderSpecificHeader),
    Many(Vec<ProviderSpecificHeader>),
}
