use std::time::Duration;

use litellm_llms_types::formats::chat_completions::ChatMessage;
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

pub struct ChatCompletionsCall {
    pub model: String,
    pub messages: Value,
    pub optional_params: Map<String, Value>,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub custom_llm_provider: Option<String>,
    pub extra_headers: Option<Map<String, Value>>,
    pub timeout: Option<Duration>,
}

impl From<ChatCompletionsRequest<'_>> for ChatCompletionsCall {
    fn from(request: ChatCompletionsRequest<'_>) -> Self {
        Self {
            model: request.model.into(),
            messages: request.messages,
            optional_params: request.optional_params,
            api_key: request.api_key.map(str::to_owned),
            api_base: request.api_base.map(str::to_owned),
            custom_llm_provider: request.custom_llm_provider.map(str::to_owned),
            extra_headers: request.extra_headers,
            timeout: request.timeout,
        }
    }
}

pub struct ResolvedChatCompletionsRequest {
    pub messages: Vec<ChatMessage>,
    pub optional_params: Map<String, Value>,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub extra_headers: Option<Map<String, Value>>,
    pub timeout: Option<Duration>,
}
