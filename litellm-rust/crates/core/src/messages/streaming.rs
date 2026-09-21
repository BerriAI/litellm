use futures_util::Stream;
use litellm_host::machine::RouteMachine;
use litellm_llms::{
    anthropic::experimental_pass_through::responses_adapters::streaming_iterator::AnthropicResponsesStreamWrapper,
    base_llm::base_model_iterator::{
        AnthropicMessagesApi, StreamError, StreamPolicy, TransformedStream,
    },
};
use litellm_types::responses::streaming::ResponsesEvent;

use crate::streaming::{StreamingCall, StreamingRoute};

pub struct Streaming;

impl StreamingCall for Streaming {
    type Contract = AnthropicMessagesApi;
    type Request = super::route::MessagesCall;
    type Response =
        litellm_types::llms::anthropic_messages::anthropic_response::AnthropicMessagesResponse;
}

pub fn messages_stream_machine() -> RouteMachine<StreamingRoute<Streaming>> {
    todo!(
        "Select a typed native or translated stream and preserve existing provider acceptance gates until the streaming path is implemented"
    )
}

pub fn messages_from_responses<S>(
    _source: S,
    _adapter: AnthropicResponsesStreamWrapper,
    _policy: StreamPolicy,
) -> TransformedStream<S, AnthropicResponsesStreamWrapper>
where
    S: Stream<Item = Result<ResponsesEvent, StreamError>>,
{
    todo!("Compose canonical Responses events with the Messages bridge before typed host delivery")
}
