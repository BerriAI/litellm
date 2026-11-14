use serde_json::{Map, Value};

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ProviderSpecificHeader {
    #[serde(default)]
    pub custom_llm_provider: String,
    #[serde(default)]
    pub extra_headers: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(untagged)]
pub enum ProviderSpecificHeaders {
    One(ProviderSpecificHeader),
    Many(Vec<ProviderSpecificHeader>),
}
