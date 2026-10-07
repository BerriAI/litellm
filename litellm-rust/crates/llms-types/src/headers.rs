use std::collections::BTreeMap;

use crate::recognized::Recognized;

#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ProviderSpecificHeader {
    #[serde(default)]
    pub custom_llm_provider: String,
    #[serde(default)]
    pub extra_headers: BTreeMap<String, Recognized<String>>,
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(untagged)]
pub enum ProviderSpecificHeaders {
    One(ProviderSpecificHeader),
    Many(Vec<ProviderSpecificHeader>),
}
