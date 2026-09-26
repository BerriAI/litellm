use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// One regex rule generalizing unknown model ids to known families.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct FallbackRule {
    pub name: String,
    pub pattern: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Regex rules that generalize unknown model ids to known families; not a model entry.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct FallbackGeneralizations {
    pub rules: Vec<FallbackRule>,
}
