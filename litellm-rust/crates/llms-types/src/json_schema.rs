use serde_json::{Map, Value};
use std::collections::BTreeMap;

use crate::recognized::Recognized;
use crate::serde_compat::deserialize_present;

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
    #[serde(default, deserialize_with = "deserialize_present")]
    #[serde(rename = "type")]
    pub schema_type: Option<Recognized<JsonSchemaType>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub properties: Option<Recognized<BTreeMap<String, Recognized<JsonSchema>>>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub required: Option<Recognized<Vec<String>>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    #[serde(rename = "additionalProperties")]
    pub additional_properties: Option<Recognized<Box<JsonSchema>>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub items: Option<Recognized<Box<JsonSchema>>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    #[serde(rename = "$defs")]
    pub defs: Option<Recognized<BTreeMap<String, Recognized<JsonSchema>>>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    #[serde(rename = "$ref")]
    pub reference: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    #[serde(rename = "anyOf")]
    pub any_of: Option<Recognized<Vec<Recognized<JsonSchema>>>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    #[serde(rename = "allOf")]
    pub all_of: Option<Recognized<Vec<Recognized<JsonSchema>>>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    #[serde(rename = "oneOf")]
    pub one_of: Option<Recognized<Vec<Recognized<JsonSchema>>>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub strict: Option<Recognized<bool>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}
