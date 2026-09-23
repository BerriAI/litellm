use std::time::Duration;

use litellm_llms::anthropic::experimental_pass_through::messages::transformation::MessagesFeatures;
use litellm_llms::base_llm::anthropic_messages::transformation::BaseAnthropicMessagesConfig;
use serde_json::{Map, Value};

pub struct MessagesRequest<'a> {
    pub model: &'a str,
    pub body: Value,
    pub api_key: Option<&'a str>,
    pub api_base: Option<&'a str>,
    pub custom_llm_provider: Option<&'a str>,
    pub extra_headers: Option<Map<String, Value>>,
    pub timeout: Option<Duration>,
    pub messages_features: &'a MessagesFeatures,
}

pub struct ProviderMessagesRequest {
    pub provider: String,
    pub model: String,
    pub config: &'static dyn BaseAnthropicMessagesConfig,
    pub url: String,
    pub body: Value,
    pub upstream_headers: Vec<(String, String)>,
    pub timeout: Option<Duration>,
}
