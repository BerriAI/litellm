use std::time::Duration;

use litellm_core::messages::transformation::AnthropicMessagesProviderConfig;
use serde_json::{Map, Value};

pub struct MessagesRequest<'a> {
    pub model: &'a str,
    pub body: Value,
    pub api_key: Option<&'a str>,
    pub api_base: Option<&'a str>,
    pub custom_llm_provider: Option<&'a str>,
    pub extra_headers: Option<Map<String, Value>>,
    pub timeout: Option<Duration>,
}

pub(crate) struct ProviderMessagesRequest {
    pub(crate) model: String,
    pub(crate) config: &'static dyn AnthropicMessagesProviderConfig,
    pub(crate) url: String,
    pub(crate) body: Value,
    pub(crate) upstream_headers: Vec<(String, String)>,
    pub(crate) timeout: Option<Duration>,
}
