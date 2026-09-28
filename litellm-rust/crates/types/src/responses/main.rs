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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<Value>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponsesApiRequest {
    pub model: String,
    pub input: ResponsesInput,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_stream"
    )]
    pub stream: Option<bool>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

fn deserialize_stream<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<bool>, D::Error> {
    bool::deserialize(deserializer).map(Some)
}
