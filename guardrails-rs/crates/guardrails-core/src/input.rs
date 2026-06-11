use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GuardrailInput {
    #[serde(default)]
    pub texts: Vec<String>,
    #[serde(default)]
    pub images: Vec<String>,
    #[serde(default)]
    pub structured_messages: Vec<Message>,
    #[serde(default)]
    pub tools: Vec<serde_json::Value>,
    #[serde(default)]
    pub tool_calls: Vec<serde_json::Value>,
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: MessageContent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<serde_json::Value>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputType {
    Request,
    Response,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RequestContext {
    #[serde(default)]
    pub litellm_call_id: Option<String>,
    #[serde(default)]
    pub litellm_trace_id: Option<String>,
    #[serde(default)]
    pub user_api_key_metadata: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    pub request_headers: Option<HashMap<String, String>>,
    #[serde(default)]
    pub dynamic_params: serde_json::Map<String, serde_json::Value>,
}
