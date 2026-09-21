use crate::base_llm::translation::TranslationError;
use bytes::Bytes;
use futures_util::Stream;
use litellm_types::llms::anthropic_messages::anthropic_request::AnthropicMessagesRequest;
use std::{
    pin::Pin,
    task::{Context, Poll},
};

pub struct AgenticAnthropicStreamingIterator<S, H> {
    _completion_stream: S,
    _host: H,
    _request: AnthropicMessagesRequest,
    _hold_back: bool,
    _collected_bytes: Vec<Bytes>,
    _follow_up_iterator: Option<S>,
}

impl<S, H> AgenticAnthropicStreamingIterator<S, H> {
    pub fn new(
        _completion_stream: S,
        _host: H,
        _request: AnthropicMessagesRequest,
        _hold_back: bool,
    ) -> Self {
        todo!()
    }

    pub fn has_buffered_provider_output(&self) -> bool {
        todo!()
    }
}

impl<S: Stream<Item = Result<Bytes, TranslationError>>, H> Stream
    for AgenticAnthropicStreamingIterator<S, H>
{
    type Item = Result<Bytes, TranslationError>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        todo!()
    }
}
