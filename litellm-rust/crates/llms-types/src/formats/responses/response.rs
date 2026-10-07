use super::ResponsesOutputItem;
use crate::recognized::Recognized;
use serde_json::{Map, Value};

#[macro_rules_attribute::apply(wire_type)]
pub struct ResponsesApiResponse {
    pub id: String,
    pub model: String,
    pub output: Vec<Recognized<ResponsesOutputItem>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}
