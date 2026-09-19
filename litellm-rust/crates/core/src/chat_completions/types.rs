use std::time::Duration;

use litellm_llms::base_llm::chat::transformation::{BaseConfig, RequestAuth};
use litellm_types::llms::openai::ChatMessage;
use serde_json::{Map, Value};

/// A `/chat/completions` call as it crosses into the core.
///
/// `optional_params` arrives already mapped to the provider's own parameter
/// names by the host, exactly as the messages route receives an already
/// Anthropic-shaped body. The core owns the conversation translation, the
/// provider call, and the response normalization.
pub struct ChatCompletionsRequest<'a> {
    pub model: &'a str,
    pub messages: Value,
    pub optional_params: Map<String, Value>,
    pub api_key: Option<&'a str>,
    pub api_base: Option<&'a str>,
    pub custom_llm_provider: Option<&'a str>,
    pub extra_headers: Option<Map<String, Value>>,
    pub timeout: Option<Duration>,
}

pub struct ResolvedChatCompletionsRequest<'a> {
    pub model: String,
    pub config: &'static dyn BaseConfig,
    pub messages: Vec<ChatMessage>,
    pub optional_params: Map<String, Value>,
    pub api_key: Option<&'a str>,
    pub api_base: Option<&'a str>,
    pub extra_headers: Option<Map<String, Value>>,
    pub timeout: Option<Duration>,
}

pub struct ProviderChatCompletionsRequest {
    pub model: String,
    pub config: &'static dyn BaseConfig,
    pub url: String,
    pub body: Value,
    pub upstream_headers: Vec<(String, String)>,
    pub auth: RequestAuth,
    pub optional_params: Map<String, Value>,
    pub timeout: Option<Duration>,
}
