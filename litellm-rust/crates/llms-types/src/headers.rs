use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ProviderSpecificHeader {
    #[serde(default)]
    pub custom_llm_provider: String,
    #[serde(default)]
    pub extra_headers: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ProviderSpecificHeaders {
    One(ProviderSpecificHeader),
    Many(Vec<ProviderSpecificHeader>),
}
