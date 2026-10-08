use litellm_secrets::source::Secrets;
use std::time::Duration;

use litellm_auth::SecretValue;
use litellm_inference::Connection;
use litellm_llms::base_llm::{auth::ValidatedEnvironment, chat::transformation::BaseConfig};
use litellm_llms_types::formats::chat_completions::ChatMessage;
use serde_json::{Map, Value};

/// A `/chat/completions` call as it crosses into the core.
///
/// `optional_params` arrives already mapped to the provider's own parameter
/// names by the host, exactly as the messages route receives an already
/// Anthropic-shaped body. The core owns the conversation translation, the
/// provider call, and the response normalization.
pub struct ChatCompletionsCall {
    pub model: String,
    pub custom_llm_provider: Option<String>,
    pub messages: Value,
    pub optional_params: Map<String, Value>,
    pub connection: Connection,
}

pub struct ResolvedChatCompletionsRequest {
    pub model: String,
    pub custom_llm_provider: String,
    pub config: &'static dyn BaseConfig,
    pub messages: Vec<ChatMessage>,
    pub optional_params: Map<String, Value>,
    pub connection: Connection,
}

pub struct ProviderChatCompletionsRequest {
    pub model: String,
    pub custom_llm_provider: String,
    pub config: &'static dyn BaseConfig,
    pub url: String,
    pub body: Value,
    /// The route's parameters before the provider transformation, reported to the host
    /// beside the wire request.
    pub optional_params: Map<String, Value>,
    /// The forwarded and default headers plus how the call authenticates; the credential
    /// itself is applied when the request is sent.
    pub environment: ValidatedEnvironment,
    pub secrets: Secrets,
    pub timeout: Option<Duration>,
    pub api_key: Option<SecretValue>,
}
