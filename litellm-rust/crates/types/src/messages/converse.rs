use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConverseRequest {
    pub messages: Vec<ConverseMessage>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConverseMessage {
    pub role: String,
    pub content: Vec<Value>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}
