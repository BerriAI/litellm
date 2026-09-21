use litellm_types::llms::anthropic_messages::{
    anthropic_request::AnthropicMessagesRequest, anthropic_response::AnthropicMessagesResponse,
};

use litellm_core_utils::call_arguments::CallArguments;

use crate::base_llm::chat::transformation::Error;
use crate::base_llm::translation::{
    ExtensionPolicy, ProviderIdentity, ProviderResponse, StreamLimits, StreamTranslation,
    TranslationError,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MessagesAuthStrategy {
    Bearer,
    Header(&'static str),
}

impl MessagesAuthStrategy {
    pub fn header_name(self) -> &'static str {
        match self {
            Self::Bearer => "authorization",
            Self::Header(header_name) => header_name,
        }
    }
}

pub trait MessagesExecutionConfig: Sync {
    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error>;

    fn transform_anthropic_messages_request(
        &self,
        request: AnthropicMessagesRequest,
    ) -> Result<AnthropicMessagesRequest, Error> {
        Ok(request)
    }

    fn transform_anthropic_messages_response(
        &self,
        _model: &str,
        response: AnthropicMessagesResponse,
    ) -> Result<AnthropicMessagesResponse, Error> {
        Ok(response)
    }

    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error>;

    fn auth_strategy(&self) -> MessagesAuthStrategy {
        MessagesAuthStrategy::Header("x-api-key")
    }

    fn accepts_bearer_auth(&self) -> bool {
        false
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        &[
            ("anthropic-version", "2023-06-01"),
            ("content-type", "application/json"),
        ]
    }
}

pub trait BaseAnthropicMessagesConfig: Send + Sync {
    type Params;
    type StreamInput;
    type CustomStream: futures_util::Stream<Item = Result<crate::anthropic::experimental_pass_through::messages::streaming_iterator::AnthropicMessagesStreamEvent, crate::base_llm::translation::TranslationError>>;
    type Credentials;
    type Environment;
    type ProviderRequest: serde::Serialize;
    type Decoder: crate::base_llm::translation::SemanticDecoder<
            Event = crate::anthropic::experimental_pass_through::messages::streaming_iterator::AnthropicMessagesStreamEvent,
            Completion = AnthropicMessagesResponse,
        >;

    fn validate_anthropic_messages_environment(
        &self,
        _credentials: &Self::Credentials,
        _arguments: &litellm_core_utils::call_arguments::CallArguments,
        _headers: &[(String, String)],
        _api_base: Option<&str>,
        _params: &Self::Params,
        _provider: &ProviderIdentity,
    ) -> Result<Self::Environment, TranslationError>;

    fn get_complete_url(
        &self,
        _api_base: Option<&str>,
        _params: &Self::Params,
        _provider: &ProviderIdentity,
        _environment: &Self::Environment,
    ) -> Result<String, TranslationError>;

    fn get_supported_anthropic_messages_params(
        &self,
        _provider: &ProviderIdentity,
    ) -> Box<[String]>;

    fn map_anthropic_messages_optional_params(
        &self,
        _arguments: &CallArguments,
        _provider: &ProviderIdentity,
    ) -> Result<Self::Params, TranslationError>;

    fn transform_anthropic_messages_request(
        &self,
        _request: &AnthropicMessagesRequest,
        _params: &Self::Params,
        _provider: &ProviderIdentity,
        _environment: &Self::Environment,
    ) -> Result<Self::ProviderRequest, TranslationError>;

    fn transform_anthropic_messages_response(
        &self,
        _response: ProviderResponse<'_>,
        _provider: &ProviderIdentity,
    ) -> Result<AnthropicMessagesResponse, TranslationError>;

    fn sign_request(
        &self,
        _request: reqwest::Request,
        _environment: &Self::Environment,
    ) -> Result<crate::base_llm::translation::SignedRequest, TranslationError>;

    fn should_filter_anthropic_beta_headers(&self) -> bool;

    fn handles_web_search_natively(&self) -> bool;

    fn get_async_streaming_response_iterator(
        &self,
        _completion_stream: Self::StreamInput,
        _provider: &crate::base_llm::translation::ProviderIdentity,
        _params: &Self::Params,
        _request: &Self::ProviderRequest,
        _limits: crate::base_llm::translation::StreamLimits,
    ) -> Result<Self::CustomStream, crate::base_llm::translation::TranslationError>;

    fn get_error_class(
        &self,
        _error_message: &str,
        _response: ProviderResponse<'_>,
    ) -> TranslationError;

    fn max_retry_on_anthropic_messages_http_error(&self) -> usize;

    fn should_retry_anthropic_messages_on_http_error(
        &self,
        _response: ProviderResponse<'_>,
        _arguments: &CallArguments,
    ) -> bool;

    fn transform_anthropic_messages_request_on_http_error(
        &self,
        _response: ProviderResponse<'_>,
        _request: &Self::ProviderRequest,
    ) -> Result<Self::ProviderRequest, TranslationError>;

    fn extension_policy(&self) -> ExtensionPolicy;

    fn stream_decoder(
        &self,
        _provider: &ProviderIdentity,
        _limits: StreamLimits,
    ) -> Result<StreamTranslation<Self::Decoder>, TranslationError>;
}

macro_rules! scaffold_messages_translation_types {
    () => {
        pub enum Parameters {}
        pub enum Credentials {}
        pub enum Environment {}

        #[derive(serde::Serialize)]
        pub enum ProviderRequest {}

        pub enum ProviderFrame {}
        pub enum StreamInput {}
        pub enum CustomStream {}

        impl futures_util::Stream for CustomStream {
            type Item = Result<crate::anthropic::experimental_pass_through::messages::streaming_iterator::AnthropicMessagesStreamEvent, crate::base_llm::translation::TranslationError>;

            fn poll_next(self: std::pin::Pin<&mut Self>, _cx: &mut std::task::Context<'_>) -> std::task::Poll<Option<Self::Item>> {
                todo!()
            }
        }

        #[derive(Default)]
        pub struct Decoder;

        impl crate::base_llm::translation::SemanticDecoder for Decoder {
            type Frame = ProviderFrame;
            type Event = crate::anthropic::experimental_pass_through::messages::streaming_iterator::AnthropicMessagesStreamEvent;
            type Completion = litellm_types::llms::anthropic_messages::anthropic_response::AnthropicMessagesResponse;

            fn decode(&mut self, _frame: Self::Frame) -> Result<crate::base_llm::translation::StreamProgress<Self::Event, Self::Completion>, crate::base_llm::translation::TranslationError> {
                todo!()
            }

            fn end_of_input(self) -> Result<crate::base_llm::base_model_iterator::StreamOutcome<Self::Completion>, crate::base_llm::translation::TranslationError> {
                todo!()
            }

        }
    };
}

macro_rules! scaffold_messages_translation_config {
    ($config:ident) => {
        #[derive(Default)]
        pub struct $config;

        $crate::base_llm::anthropic_messages::transformation::scaffold_messages_translation_config!(@existing $config);
    };
    (@existing $config:ident) => {
        impl crate::base_llm::anthropic_messages::transformation::BaseAnthropicMessagesConfig
            for $config
        {
            type Params = Parameters;
            type StreamInput = StreamInput;
            type CustomStream = CustomStream;
            type Credentials = Credentials;
            type Environment = Environment;
            type ProviderRequest = ProviderRequest;
            type Decoder = Decoder;

    fn validate_anthropic_messages_environment(
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

    fn get_complete_url(
        &self,
        _api_base: Option<&str>,
        _params: &Self::Params,
        _provider: &crate::base_llm::translation::ProviderIdentity,
        _environment: &Self::Environment,
    ) -> Result<String, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn get_supported_anthropic_messages_params(
        &self,
        _provider: &crate::base_llm::translation::ProviderIdentity,
    ) -> Box<[String]> {
        todo!()
    }

    fn map_anthropic_messages_optional_params(
        &self,
        _arguments: &litellm_core_utils::call_arguments::CallArguments,
        _provider: &crate::base_llm::translation::ProviderIdentity,
    ) -> Result<Self::Params, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn transform_anthropic_messages_request(
        &self,
        _request: &litellm_types::llms::anthropic_messages::anthropic_request::AnthropicMessagesRequest,
        _params: &Self::Params,
        _provider: &crate::base_llm::translation::ProviderIdentity,
        _environment: &Self::Environment,
    ) -> Result<Self::ProviderRequest, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn transform_anthropic_messages_response(
        &self,
        _response: crate::base_llm::translation::ProviderResponse<'_>,
        _provider: &crate::base_llm::translation::ProviderIdentity,
    ) -> Result<litellm_types::llms::anthropic_messages::anthropic_response::AnthropicMessagesResponse, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn sign_request(
        &self,
        _request: reqwest::Request,
        _environment: &Self::Environment,
    ) -> Result<crate::base_llm::translation::SignedRequest, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn should_filter_anthropic_beta_headers(&self) -> bool {
        todo!()
    }

    fn handles_web_search_natively(&self) -> bool {
        todo!()
    }

    fn get_async_streaming_response_iterator(
        &self,
        _completion_stream: Self::StreamInput,
        _provider: &crate::base_llm::translation::ProviderIdentity,
        _params: &Self::Params,
        _request: &Self::ProviderRequest,
        _limits: crate::base_llm::translation::StreamLimits,
    ) -> Result<Self::CustomStream, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn get_error_class(&self, _error_message: &str, _response: crate::base_llm::translation::ProviderResponse<'_>) -> crate::base_llm::translation::TranslationError {
        todo!()
    }

    fn max_retry_on_anthropic_messages_http_error(&self) -> usize {
        todo!()
    }

    fn should_retry_anthropic_messages_on_http_error(&self, _response: crate::base_llm::translation::ProviderResponse<'_>, _arguments: &litellm_core_utils::call_arguments::CallArguments) -> bool {
        todo!()
    }

    fn transform_anthropic_messages_request_on_http_error(
        &self,
        _response: crate::base_llm::translation::ProviderResponse<'_>,
        _request: &Self::ProviderRequest,
    ) -> Result<Self::ProviderRequest, crate::base_llm::translation::TranslationError> {
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

pub(crate) use scaffold_messages_translation_config;
pub(crate) use scaffold_messages_translation_types;
