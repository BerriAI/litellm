use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::ResponsesItem;
use crate::recognized::Recognized;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<Recognized<ResponsesItem>>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponsesRequest {
    pub model: String,
    pub input: ResponsesInput,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}
