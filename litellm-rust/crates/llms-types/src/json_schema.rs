use serde_json::{Map, Value};
use std::collections::BTreeMap;

#[macro_rules_attribute::apply(wire_type)]
#[serde(untagged)]
pub enum JsonSchema {
    Boolean(bool),
    Object(Box<JsonSchemaObject>),
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(untagged)]
pub enum JsonSchemaType {
    Name(String),
    Names(Vec<String>),
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct JsonSchemaObject {
    #[serde(rename = "type")]
    pub schema_type: Option<JsonSchemaType>,
    pub properties: Option<BTreeMap<String, JsonSchema>>,
    pub required: Option<Vec<String>>,
    #[serde(rename = "additionalProperties")]
    pub additional_properties: Option<Box<JsonSchema>>,
    pub items: Option<Box<JsonSchema>>,
    #[serde(rename = "$defs")]
    pub defs: Option<BTreeMap<String, JsonSchema>>,
    #[serde(rename = "$ref")]
    pub reference: Option<String>,
    #[serde(rename = "anyOf")]
    pub any_of: Option<Vec<JsonSchema>>,
    #[serde(rename = "allOf")]
    pub all_of: Option<Vec<JsonSchema>>,
    #[serde(rename = "oneOf")]
    pub one_of: Option<Vec<JsonSchema>>,
    pub strict: Option<bool>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}
