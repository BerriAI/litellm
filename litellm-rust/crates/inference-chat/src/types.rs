use litellm_secrets::source::Secrets;
use std::{sync::Arc, time::Duration};

use litellm_auth::SecretValue;
use litellm_llms::base_llm::{
    auth::ValidatedEnvironment,
    chat::{streaming::StreamShape, transformation::BaseConfig},
    model_info::ModelInfo,
};
use litellm_llms_types::formats::chat_completions::ChatMessage;
use serde_json::{Map, Value};

/// A `/chat/completions` call as it crosses into the core.
///
/// `optional_params` are the OpenAI request's own params; the provider config maps them,
/// refusing what the provider cannot take unless `drop_params` says to drop it. The core
/// owns the conversation translation, the provider call, and the response normalization.
pub struct ChatCompletionsRequest<'a> {
    pub model: &'a str,
    pub messages: Value,
    pub optional_params: Map<String, Value>,
    pub api_key: Option<&'a str>,
    pub api_base: Option<&'a str>,
    pub custom_llm_provider: Option<&'a str>,
    pub extra_headers: Option<Map<String, Value>>,
    pub timeout: Option<Duration>,
    /// `drop_params` from the deployment or the proxy settings, `None` when neither says. A
    /// request's own `drop_params` rides in `optional_params`.
    pub drop_params: Option<bool>,
    /// The model cost map, for the param rules that read model capabilities.
    pub model_info: Option<Arc<dyn ModelInfo>>,
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
    /// `drop_params` from the deployment or the proxy settings, `None` when neither says. A
    /// request's own `drop_params` rides in `optional_params`.
    pub drop_params: Option<bool>,
    /// The model cost map, for the param rules that read model capabilities.
    pub model_info: Option<Arc<dyn ModelInfo>>,
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
            drop_params: request.drop_params,
            model_info: request.model_info,
        }
    }
}

pub struct ResolvedChatCompletionsRequest<'a> {
    pub model: String,
    pub custom_llm_provider: String,
    pub config: &'static dyn BaseConfig,
    pub messages: Vec<ChatMessage>,
    pub optional_params: Map<String, Value>,
    pub api_key: Option<&'a str>,
    pub api_base: Option<&'a str>,
    pub extra_headers: Option<Map<String, Value>>,
    pub timeout: Option<Duration>,
    /// `drop_params` from the deployment or the proxy settings, `None` when neither says. A
    /// request's own `drop_params` rides in `optional_params`.
    pub drop_params: Option<bool>,
    /// The model cost map, for the param rules that read model capabilities.
    pub model_info: Option<Arc<dyn ModelInfo>>,
}

pub struct ProviderChatCompletionsRequest {
    pub model: String,
    pub custom_llm_provider: String,
    pub config: &'static dyn BaseConfig,
    pub url: String,
    pub body: Value,
    /// How the config's stream parser reads a streamed answer to this request.
    pub stream_shape: StreamShape,
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
