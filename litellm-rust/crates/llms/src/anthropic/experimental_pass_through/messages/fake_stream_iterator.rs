use crate::base_llm::translation::TranslationError;
use bytes::Bytes;
use futures_util::Stream;
use litellm_types::llms::anthropic_messages::anthropic_response::AnthropicMessagesResponse;
use std::{
    pin::Pin,
    task::{Context, Poll},
};

pub struct FakeAnthropicMessagesStreamIterator {
    _response: AnthropicMessagesResponse,
    _pending: std::collections::VecDeque<Bytes>,
}

impl FakeAnthropicMessagesStreamIterator {
    pub fn new(_response: AnthropicMessagesResponse) -> Self {
        todo!()
    }
}

impl Stream for FakeAnthropicMessagesStreamIterator {
    type Item = Result<Bytes, TranslationError>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        todo!()
    }
}
