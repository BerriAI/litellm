use litellm_core_utils::call_arguments::CallArguments;
use litellm_types::{
    llms::openai::{ChatMessage, ChatMessageContent},
    utils::ChatCompletionsResponse,
};
use serde_json::{Map, Value};

use crate::base_llm::translation::{
    ExtensionPolicy, ProviderIdentity, ProviderResponse, StreamLimits, StreamTranslation,
    TranslationError,
};

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("expected {expected}, got {actual}")]
    InvalidType {
        expected: &'static str,
        actual: &'static str,
    },
    #[error("missing required field: {0}")]
    MissingField(&'static str),
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("invalid response: {0}")]
    InvalidResponse(String),
    #[error("unsupported: {0}")]
    Unsupported(&'static str),
    #[error(transparent)]
    Auth(#[from] litellm_auth::Error),
}

/// The provider-shaped request body a config produces. Named rather than a bare
/// `Value` so the transform contract stays a typed one, mirroring
/// [`crate::base_llm::audio_transcription::transformation::AudioTranscriptionRequestData`].
pub struct ProviderChatRequestData {
    pub body: Value,
}

/// The raw provider response body handed back to a config for normalization.
pub struct ProviderChatResponseData {
    pub body: Value,
}

pub const STREAM_PARAM: &str = "stream";

/// Message fields that carry no meaning for the upstream body, so their
/// presence does not make a request untranslatable.
const IGNORABLE_MESSAGE_FIELDS: &[&str] = &["name"];

pub use litellm_auth::RequestAuth;

/// Why a request cannot be served by the Rust path.
///
/// The core declines rather than guessing: the host turns this into a
/// transparent fallback to the Python implementation, which covers the full
/// surface. Acceptance is an allowlist, so a parameter or message shape the
/// core has never seen declines by construction instead of being translated
/// wrong.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Unsupported(pub &'static str);

pub trait ChatExecutionConfig: Sync {
    /// Supported OpenAI parameter names paired with their provider names.
    fn supported_openai_param_mappings(&self) -> &'static [(&'static str, &'static str)];

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error>;

    fn transform_request(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        optional_params: Map<String, Value>,
    ) -> Result<ProviderChatRequestData, Error>;

    fn transform_response(
        &self,
        model: &str,
        response: ProviderChatResponseData,
    ) -> Result<ChatCompletionsResponse, Error>;

    fn auth(
        &self,
        api_key: Option<&str>,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<RequestAuth, Error>;

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        &[("content-type", "application/json")]
    }

    /// Whether an auth header the caller already supplied is the credential this
    /// request should authenticate with, so the resolved one is not applied.
    ///
    /// Defaults to false: the deployment's credential outranks anything
    /// forwarded, which is what every provider wants for its own auth header.
    /// A provider overrides this only for a scheme it hands off to entirely.
    fn defers_to_forwarded_auth(&self, _headers: &[(String, String)]) -> bool {
        false
    }

    /// Parameters consumed as call configuration (credentials, endpoints)
    /// rather than placed in the body. Accepted, never serialized.
    fn config_params(&self) -> &'static [&'static str] {
        &[]
    }

    fn unsupported_reason(
        &self,
        messages: &[ChatMessage],
        optional_params: &Map<String, Value>,
    ) -> Option<Unsupported> {
        unsupported_param(
            self.supported_openai_param_mappings(),
            self.config_params(),
            optional_params,
        )
        .or_else(|| messages.iter().find_map(unsupported_message))
    }
}

pub fn unsupported_param(
    supported: &'static [(&'static str, &'static str)],
    config: &'static [&'static str],
    optional_params: &Map<String, Value>,
) -> Option<Unsupported> {
    if optional_params
        .get(STREAM_PARAM)
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return Some(Unsupported("streaming"));
    }
    optional_params
        .keys()
        .any(|key| {
            key != STREAM_PARAM
                && !supported
                    .iter()
                    .any(|(_, provider_name)| *provider_name == key)
                && !config.contains(&key.as_str())
        })
        .then_some(Unsupported("unrecognized request parameter"))
}

/// Message shapes the core can translate faithfully: text content, either a
/// plain string or a non-empty list of parts that are all
/// `{"type": "text", "text": ...}`. Tool calls, tool results, and multimodal
/// parts decline so Python's fuller translation handles them.
pub fn unsupported_message(message: &ChatMessage) -> Option<Unsupported> {
    if message
        .extra
        .keys()
        .any(|key| !IGNORABLE_MESSAGE_FIELDS.contains(&key.as_str()))
    {
        return Some(Unsupported("unrecognized message field"));
    }
    if !matches!(message.role.as_str(), "system" | "user" | "assistant") {
        return Some(Unsupported("unrecognized message role"));
    }
    match &message.content {
        None => Some(Unsupported("message without content")),
        Some(ChatMessageContent::Text(_)) => None,
        Some(ChatMessageContent::Parts(parts)) if parts.is_empty() => {
            Some(Unsupported("message without content"))
        }
        Some(ChatMessageContent::Parts(parts)) => parts
            .iter()
            .any(|part| {
                part.get("type").and_then(Value::as_str) != Some("text")
                    || part.get("text").and_then(Value::as_str).is_none()
                    || part.as_object().is_some_and(|object| object.len() != 2)
            })
            .then_some(Unsupported("non-text message content")),
    }
}

pub trait BaseConfig: Send + Sync {
    type Params;
    type StreamInput;
    type CustomStream: futures_util::Stream<
            Item = Result<
                litellm_types::utils::ChatCompletionChunk,
                crate::base_llm::translation::TranslationError,
            >,
        >;
    type ParsedResponse;
    type Credentials;
    type Environment;
    type ProviderRequest: serde::Serialize;
    type Decoder: crate::base_llm::translation::SemanticDecoder<
            Event = litellm_types::utils::ChatCompletionChunk,
            Completion = ChatCompletionsResponse,
        >;

    fn get_config(&self) -> std::collections::BTreeMap<String, Value>;

    fn get_json_schema_from_pydantic_object(&self, _response_format: &Value) -> Option<Value>;

    fn is_thinking_enabled(&self, _params: &CallArguments) -> bool;

    fn is_max_tokens_in_request(&self, _params: &CallArguments) -> bool;

    fn update_optional_params_with_thinking_tokens(
        &self,
        _params: &CallArguments,
        _optional_params: &Self::Params,
    ) -> Result<Self::Params, TranslationError>;

    fn should_fake_stream(&self, _params: &CallArguments) -> bool;

    fn add_tools_to_optional_params(
        &self,
        _params: &CallArguments,
        _tools: &[Value],
    ) -> Result<Self::Params, TranslationError>;

    fn translate_developer_role_to_system_role(
        &self,
        _messages: &[ChatMessage],
    ) -> Result<Box<[ChatMessage]>, TranslationError>;

    fn should_retry_llm_api_inside_llm_translation_on_http_error(
        &self,
        _response: ProviderResponse<'_>,
        _arguments: &CallArguments,
    ) -> bool;

    fn transform_request_on_unprocessable_entity_error(
        &self,
        _response: ProviderResponse<'_>,
        _request: &Self::ProviderRequest,
    ) -> Result<Self::ProviderRequest, TranslationError>;

    fn max_retry_on_unprocessable_entity_error(&self) -> usize;

    fn get_supported_openai_params(&self, _provider: &ProviderIdentity) -> Box<[String]>;

    fn add_response_format_to_tools(
        &self,
        _params: &CallArguments,
        _response_format: &Value,
    ) -> Result<Self::Params, TranslationError>;

    fn map_openai_params(
        &self,
        _arguments: &CallArguments,
        _provider: &ProviderIdentity,
    ) -> Result<Self::Params, TranslationError>;

    fn validate_environment(
        &self,
        _credentials: &Self::Credentials,
        _arguments: &litellm_core_utils::call_arguments::CallArguments,
        _headers: &[(String, String)],
        _api_base: Option<&str>,
        _params: &Self::Params,
        _provider: &ProviderIdentity,
    ) -> Result<Self::Environment, TranslationError>;

    fn sign_request(
        &self,
        _request: reqwest::Request,
        _environment: &Self::Environment,
    ) -> Result<crate::base_llm::translation::SignedRequest, TranslationError>;

    fn get_complete_url(
        &self,
        _api_base: Option<&str>,
        _params: &Self::Params,
        _provider: &ProviderIdentity,
        _environment: &Self::Environment,
    ) -> Result<String, TranslationError>;

    fn transform_request(
        &self,
        _messages: &[ChatMessage],
        _params: &Self::Params,
        _provider: &ProviderIdentity,
        _environment: &Self::Environment,
    ) -> Result<Self::ProviderRequest, TranslationError>;

    fn async_transform_request(
        &self,
        _messages: &[ChatMessage],
        _params: &Self::Params,
        _provider: &ProviderIdentity,
        _environment: &Self::Environment,
    ) -> impl std::future::Future<Output = Result<Self::ProviderRequest, TranslationError>> + Send;

    fn transform_response(
        &self,
        _response: ProviderResponse<'_>,
        _provider: &ProviderIdentity,
    ) -> Result<ChatCompletionsResponse, TranslationError>;

    fn transform_parsed_response_dict(
        &self,
        _response: Self::ParsedResponse,
    ) -> Result<Self::ParsedResponse, TranslationError>;

    fn get_error_class(
        &self,
        _error_message: &str,
        _response: ProviderResponse<'_>,
    ) -> TranslationError;

    fn get_model_response_iterator(
        &self,
        _completion_stream: Self::StreamInput,
        _provider: &crate::base_llm::translation::ProviderIdentity,
        _params: &Self::Params,
        _request: &Self::ProviderRequest,
        _limits: crate::base_llm::translation::StreamLimits,
    ) -> Result<Self::Decoder, crate::base_llm::translation::TranslationError>;

    fn get_async_custom_stream_wrapper(
        &self,
        _completion_stream: Self::StreamInput,
        _provider: &crate::base_llm::translation::ProviderIdentity,
        _params: &Self::Params,
        _request: &Self::ProviderRequest,
        _limits: crate::base_llm::translation::StreamLimits,
    ) -> impl std::future::Future<
        Output = Result<Self::CustomStream, crate::base_llm::translation::TranslationError>,
    > + Send;

    fn get_sync_custom_stream_wrapper(
        &self,
        _completion_stream: Self::StreamInput,
        _provider: &crate::base_llm::translation::ProviderIdentity,
        _params: &Self::Params,
        _request: &Self::ProviderRequest,
        _limits: crate::base_llm::translation::StreamLimits,
    ) -> Result<Self::CustomStream, crate::base_llm::translation::TranslationError>;

    fn custom_llm_provider(&self) -> Option<&'static str>;

    fn has_custom_stream_wrapper(&self) -> bool;

    fn uses_async_transform_request(&self) -> bool;

    fn supports_stream_param_in_request_body(&self) -> bool;

    fn post_stream_processing(&self, _stream: Self::CustomStream) -> Self::CustomStream;

    fn apply_assembled_streaming_response_metadata(
        &self,
        _response: ChatCompletionsResponse,
    ) -> ChatCompletionsResponse;

    fn calculate_additional_costs(
        &self,
        _model: &str,
        _prompt_tokens: u64,
        _completion_tokens: u64,
    ) -> Option<Value>;

    fn extension_policy(&self) -> ExtensionPolicy;

    fn stream_decoder(
        &self,
        _provider: &ProviderIdentity,
        _limits: StreamLimits,
    ) -> Result<StreamTranslation<Self::Decoder>, TranslationError>;
}

macro_rules! scaffold_chat_translation_types {
    () => {
        pub enum Parameters {}
        pub enum Credentials {}
        pub enum Environment {}

        #[derive(serde::Serialize)]
        pub enum ProviderRequest {}

        pub enum ProviderResponse {}
        pub enum ProviderFrame {}
        pub enum StreamInput {}
        pub enum CustomStream {}

        impl futures_util::Stream for CustomStream {
            type Item = Result<
                litellm_types::utils::ChatCompletionChunk,
                crate::base_llm::translation::TranslationError,
            >;

            fn poll_next(
                self: std::pin::Pin<&mut Self>,
                _cx: &mut std::task::Context<'_>,
            ) -> std::task::Poll<Option<Self::Item>> {
                todo!()
            }
        }

        #[derive(Default)]
        pub struct Decoder;

        impl crate::base_llm::translation::SemanticDecoder for Decoder {
            type Frame = ProviderFrame;
            type Event = litellm_types::utils::ChatCompletionChunk;
            type Completion = litellm_types::utils::ChatCompletionsResponse;

            fn decode(
                &mut self,
                _frame: Self::Frame,
            ) -> Result<
                crate::base_llm::translation::StreamProgress<Self::Event, Self::Completion>,
                crate::base_llm::translation::TranslationError,
            > {
                todo!()
            }

            fn end_of_input(
                self,
            ) -> Result<
                crate::base_llm::base_model_iterator::StreamOutcome<Self::Completion>,
                crate::base_llm::translation::TranslationError,
            > {
                todo!()
            }
        }
    };
}

macro_rules! scaffold_chat_translation_config {
    ($config:ident) => {
        #[derive(Default)]
        pub struct $config;

        $crate::base_llm::chat::transformation::scaffold_chat_translation_config!(@existing $config);
    };
    (@existing $config:ident) => {
        impl crate::base_llm::chat::transformation::BaseConfig for $config {
            type Params = Parameters;
            type StreamInput = StreamInput;
            type CustomStream = CustomStream;
            type ParsedResponse = ProviderResponse;
            type Credentials = Credentials;
            type Environment = Environment;
            type ProviderRequest = ProviderRequest;
            type Decoder = Decoder;

    fn get_config(&self) -> std::collections::BTreeMap<String, serde_json::Value> {
        todo!()
    }

    fn get_json_schema_from_pydantic_object(&self, _response_format: &serde_json::Value) -> Option<serde_json::Value> {
        todo!()
    }

    fn is_thinking_enabled(&self, _params: &litellm_core_utils::call_arguments::CallArguments) -> bool {
        todo!()
    }

    fn is_max_tokens_in_request(&self, _params: &litellm_core_utils::call_arguments::CallArguments) -> bool {
        todo!()
    }

    fn update_optional_params_with_thinking_tokens(
        &self,
        _params: &litellm_core_utils::call_arguments::CallArguments,
        _optional_params: &Self::Params,
    ) -> Result<Self::Params, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn should_fake_stream(&self, _params: &litellm_core_utils::call_arguments::CallArguments) -> bool {
        todo!()
    }

    fn add_tools_to_optional_params(
        &self,
        _params: &litellm_core_utils::call_arguments::CallArguments,
        _tools: &[serde_json::Value],
    ) -> Result<Self::Params, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn translate_developer_role_to_system_role(
        &self,
        _messages: &[litellm_types::llms::openai::ChatMessage],
    ) -> Result<Box<[litellm_types::llms::openai::ChatMessage]>, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn should_retry_llm_api_inside_llm_translation_on_http_error(&self, _response: crate::base_llm::translation::ProviderResponse<'_>, _arguments: &litellm_core_utils::call_arguments::CallArguments) -> bool {
        todo!()
    }

    fn transform_request_on_unprocessable_entity_error(
        &self,
        _response: crate::base_llm::translation::ProviderResponse<'_>,
        _request: &Self::ProviderRequest,
    ) -> Result<Self::ProviderRequest, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn max_retry_on_unprocessable_entity_error(&self) -> usize {
        todo!()
    }

    fn get_supported_openai_params(&self, _provider: &crate::base_llm::translation::ProviderIdentity) -> Box<[String]> {
        todo!()
    }

    fn add_response_format_to_tools(
        &self,
        _params: &litellm_core_utils::call_arguments::CallArguments,
        _response_format: &serde_json::Value,
    ) -> Result<Self::Params, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn map_openai_params(
        &self,
        _arguments: &litellm_core_utils::call_arguments::CallArguments,
        _provider: &crate::base_llm::translation::ProviderIdentity,
    ) -> Result<Self::Params, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn validate_environment(
        &self,
        _credentials: &Self::Credentials,
        _arguments: &litellm_core_utils::call_arguments::CallArguments,
        _headers: &[(String, String)],
        _api_base: Option<&str>,
        _params: &Self::Params,
        _provider: &crate::base_llm::translation::ProviderIdentity,
    ) -> Result<Self::Environment, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn sign_request(
        &self,
        _request: reqwest::Request,
        _environment: &Self::Environment,
    ) -> Result<crate::base_llm::translation::SignedRequest, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn get_complete_url(
        &self,
        _api_base: Option<&str>,
        _params: &Self::Params,
        _provider: &crate::base_llm::translation::ProviderIdentity,
        _environment: &Self::Environment,
    ) -> Result<String, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn transform_request(
        &self,
        _messages: &[litellm_types::llms::openai::ChatMessage],
        _params: &Self::Params,
        _provider: &crate::base_llm::translation::ProviderIdentity,
        _environment: &Self::Environment,
    ) -> Result<Self::ProviderRequest, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn async_transform_request(
        &self,
        _messages: &[litellm_types::llms::openai::ChatMessage],
        _params: &Self::Params,
        _provider: &crate::base_llm::translation::ProviderIdentity,
        _environment: &Self::Environment,
    ) -> impl std::future::Future<Output = Result<Self::ProviderRequest, crate::base_llm::translation::TranslationError>> + Send
    {
        async { todo!() }
    }

    fn transform_response(
        &self,
        _response: crate::base_llm::translation::ProviderResponse<'_>,
        _provider: &crate::base_llm::translation::ProviderIdentity,
    ) -> Result<litellm_types::utils::ChatCompletionsResponse, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn transform_parsed_response_dict(&self, _response: Self::ParsedResponse) -> Result<Self::ParsedResponse, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn get_error_class(&self, _error_message: &str, _response: crate::base_llm::translation::ProviderResponse<'_>) -> crate::base_llm::translation::TranslationError {
        todo!()
    }

    fn get_model_response_iterator(
        &self,
        _completion_stream: Self::StreamInput,
        _provider: &crate::base_llm::translation::ProviderIdentity,
        _params: &Self::Params,
        _request: &Self::ProviderRequest,
        _limits: crate::base_llm::translation::StreamLimits,
    ) -> Result<Self::Decoder, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn get_async_custom_stream_wrapper(
        &self,
        _completion_stream: Self::StreamInput,
        _provider: &crate::base_llm::translation::ProviderIdentity,
        _params: &Self::Params,
        _request: &Self::ProviderRequest,
        _limits: crate::base_llm::translation::StreamLimits,
    ) -> impl std::future::Future<Output = Result<Self::CustomStream, crate::base_llm::translation::TranslationError>> + Send { async { todo!() } }

    fn get_sync_custom_stream_wrapper(
        &self,
        _completion_stream: Self::StreamInput,
        _provider: &crate::base_llm::translation::ProviderIdentity,
        _params: &Self::Params,
        _request: &Self::ProviderRequest,
        _limits: crate::base_llm::translation::StreamLimits,
    ) -> Result<Self::CustomStream, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn custom_llm_provider(&self) -> Option<&'static str> {
        todo!()
    }

    fn has_custom_stream_wrapper(&self) -> bool {
        todo!()
    }

    fn uses_async_transform_request(&self) -> bool {
        todo!()
    }

    fn supports_stream_param_in_request_body(&self) -> bool {
        todo!()
    }

    fn post_stream_processing(&self, _stream: Self::CustomStream) -> Self::CustomStream {
        todo!()
    }

    fn apply_assembled_streaming_response_metadata(
        &self,
        _response: litellm_types::utils::ChatCompletionsResponse,
    ) -> litellm_types::utils::ChatCompletionsResponse {
        todo!()
    }

    fn calculate_additional_costs(
        &self,
        _model: &str,
        _prompt_tokens: u64,
        _completion_tokens: u64,
    ) -> Option<serde_json::Value> {
        todo!()
    }

    fn extension_policy(&self) -> crate::base_llm::translation::ExtensionPolicy {
        todo!()
    }

    fn stream_decoder(
        &self,
        _provider: &crate::base_llm::translation::ProviderIdentity,
        _limits: crate::base_llm::translation::StreamLimits,
    ) -> Result<crate::base_llm::translation::StreamTranslation<Self::Decoder>, crate::base_llm::translation::TranslationError> {
        todo!()
    }
        }
    };
}

pub(crate) use scaffold_chat_translation_config;
pub(crate) use scaffold_chat_translation_types;
