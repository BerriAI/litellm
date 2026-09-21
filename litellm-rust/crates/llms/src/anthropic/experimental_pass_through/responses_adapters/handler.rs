use litellm_types::{
    llms::anthropic_messages::{
        anthropic_request::AnthropicMessagesRequest, anthropic_response::AnthropicMessagesResponse,
    },
    responses::streaming::{ResponsesRequest, ResponsesResponse},
};

use super::transformation::LiteLlmAnthropicToResponsesApiAdapter;
use crate::base_llm::translation::TranslationError;

pub trait ResponsesExecution: Send + Sync {
    type Stream: futures_util::Stream<
            Item = Result<
                litellm_types::responses::streaming::ResponsesEvent,
                crate::base_llm::base_model_iterator::StreamError,
            >,
        >;

    fn execute(
        &self,
        request: ResponsesRequest,
    ) -> impl std::future::Future<Output = Result<ResponsesDelivery<Self::Stream>, TranslationError>>
    + Send;
}

pub enum ResponsesDelivery<S> {
    Completed(Box<ResponsesResponse>),
    Streaming(S),
}

pub enum MessagesDelivery<S> {
    Completed(Box<AnthropicMessagesResponse>),
    Streaming(S),
}

pub struct LiteLlmMessagesToResponsesApiHandler<E> {
    pub execution: E,
    pub adapter: LiteLlmAnthropicToResponsesApiAdapter,
}

impl<E: ResponsesExecution> LiteLlmMessagesToResponsesApiHandler<E> {
    pub fn anthropic_messages_handler(
        &self,
        _request: AnthropicMessagesRequest,
    ) -> Result<
        MessagesDelivery<
            crate::base_llm::base_model_iterator::TransformedStream<
                E::Stream,
                super::streaming_iterator::AnthropicResponsesStreamWrapper,
            >,
        >,
        TranslationError,
    > {
        todo!()
    }

    pub async fn async_anthropic_messages_handler(
        &self,
        _request: AnthropicMessagesRequest,
    ) -> Result<
        MessagesDelivery<
            crate::base_llm::base_model_iterator::TransformedStream<
                E::Stream,
                super::streaming_iterator::AnthropicResponsesStreamWrapper,
            >,
        >,
        TranslationError,
    > {
        todo!()
    }
}
