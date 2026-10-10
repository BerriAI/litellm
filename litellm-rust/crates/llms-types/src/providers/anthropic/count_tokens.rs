use serde_json::Value;

use crate::formats::messages::{Message, SystemPrompt};

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct AnthropicCountTokensRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub tools: Option<Vec<Value>>,
    pub system: Option<SystemPrompt>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Eq)]
pub struct AnthropicCountTokensResponse {
    pub input_tokens: u64,
}
