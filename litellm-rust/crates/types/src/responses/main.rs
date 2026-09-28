use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponsesApiResponse {
    pub id: String,
    pub model: String,
    pub output: Vec<Value>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}
