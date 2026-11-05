use serde_json::{Map, Value};

#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ResponsesApiResponse {
    pub id: String,
    pub model: String,
    pub output: Vec<Value>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}
