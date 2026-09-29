use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::ResponsesItem;
use crate::recognized::Recognized;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponsesApiResponse {
    pub id: String,
    pub model: String,
    pub output: Vec<Recognized<ResponsesItem>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}
