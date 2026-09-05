use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<ResponsesInputItem>),
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct ResponsesInputItem {
    pub role: String,
    pub content: ResponsesInputContent,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ResponsesInputContent {
    Text(String),
    Parts(Vec<ResponsesInputTextPart>),
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct ResponsesInputTextPart {
    #[serde(rename = "type")]
    pub part_type: String,
    pub text: String,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

pub struct ResponsesRequest<'a> {
    pub model: &'a str,
    pub input: Value,
    pub optional_params: Map<String, Value>,
    pub api_key: Option<&'a str>,
    pub api_base: Option<&'a str>,
    pub custom_llm_provider: Option<&'a str>,
    pub extra_headers: Option<Map<String, Value>>,
    pub timeout: Option<Duration>,
    pub use_chat_completions_api: bool,
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct ResponsesApiResponse {
    pub id: String,
    pub created_at: u64,
    pub output: Vec<ResponsesOutputItem>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ResponsesOutputItem {
    #[serde(rename = "message")]
    Message {
        id: String,
        status: String,
        role: String,
        content: Vec<ResponsesOutputContent>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ResponsesOutputContent {
    #[serde(rename = "output_text")]
    OutputText {
        text: String,
        annotations: Vec<Value>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}
