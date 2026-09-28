use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::llms::openai::ChatMessage;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatCompletionsRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}
