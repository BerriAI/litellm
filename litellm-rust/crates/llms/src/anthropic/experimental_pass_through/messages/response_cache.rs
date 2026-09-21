use crate::base_llm::translation::TranslationError;
use bytes::Bytes;
use futures_util::Stream;
use std::{
    pin::Pin,
    task::{Context, Poll},
};

pub trait MessagesStreamCache: Send + Sync {
    fn persist(
        &self,
        events: Box<[Bytes]>,
    ) -> impl std::future::Future<Output = Result<(), TranslationError>> + Send;
}

pub struct AnthropicMessagesStreamCacheWriter<S, C> {
    _stream: S,
    _caching_handler: C,
    _collected_chunks: Vec<Bytes>,
    _persisted: bool,
}

impl<S, C: MessagesStreamCache> AnthropicMessagesStreamCacheWriter<S, C> {
    pub fn new(_stream: S, _caching_handler: C) -> Self {
        todo!()
    }

    pub async fn persist(&mut self) -> Result<(), TranslationError> {
        todo!()
    }
}

impl<S: Stream<Item = Result<Bytes, TranslationError>>, C: MessagesStreamCache> Stream
    for AnthropicMessagesStreamCacheWriter<S, C>
{
    type Item = Result<Bytes, TranslationError>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        todo!()
    }
}

pub struct CachedAnthropicMessagesStreamIterator<H> {
    _events: Box<[Bytes]>,
    _current_index: usize,
    _host: H,
}

impl<H> CachedAnthropicMessagesStreamIterator<H> {
    pub fn new(_events: Vec<Bytes>, _host: H) -> Self {
        todo!()
    }
}

impl<H> Stream for CachedAnthropicMessagesStreamIterator<H> {
    type Item = Result<Bytes, TranslationError>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        todo!()
    }
}
