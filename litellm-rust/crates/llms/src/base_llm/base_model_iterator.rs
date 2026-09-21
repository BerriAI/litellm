use std::{
    collections::VecDeque,
    pin::Pin,
    task::{Context, Poll},
};

use futures_util::Stream;
use litellm_types::{responses::streaming::ResponsesEvent, utils::ChatCompletionChunk};

use crate::anthropic::experimental_pass_through::messages::streaming_iterator::AnthropicMessagesStreamEvent;

pub trait ServerStreamingContract {
    type Event;
}

pub struct ChatCompletions;
pub struct ResponsesApi;
pub struct AnthropicMessagesApi;

impl ServerStreamingContract for ChatCompletions {
    type Event = ChatCompletionChunk;
}

impl ServerStreamingContract for ResponsesApi {
    type Event = ResponsesEvent;
}

impl ServerStreamingContract for AnthropicMessagesApi {
    type Event = AnthropicMessagesStreamEvent;
}

pub trait StreamTransformation: Sized {
    type Caller: ServerStreamingContract;
    type Upstream: ServerStreamingContract;
    type Error;

    fn transform_event(
        &mut self,
        event: <Self::Upstream as ServerStreamingContract>::Event,
    ) -> Result<Vec<<Self::Caller as ServerStreamingContract>::Event>, Self::Error>;

    fn finish(self) -> Result<Vec<<Self::Caller as ServerStreamingContract>::Event>, Self::Error>;
}

pub struct TransformedStream<S, T: StreamTransformation> {
    upstream: S,
    transformation: Option<T>,
    pending: VecDeque<<T::Caller as ServerStreamingContract>::Event>,
    finished: bool,
}

impl<S, T: StreamTransformation> TransformedStream<S, T> {
    pub fn new(_upstream: S, _transformation: T) -> Self {
        todo!("Retain one call's conversion state and a bounded pending-event queue")
    }
}

impl<S, T> Stream for TransformedStream<S, T>
where
    T: StreamTransformation,
    S: Stream<Item = Result<<T::Upstream as ServerStreamingContract>::Event, T::Error>>,
{
    type Item = Result<<T::Caller as ServerStreamingContract>::Event, T::Error>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        todo!(
            "Drain pending events before polling upstream; preserve backpressure, finish once on EOF, and propagate cancellation and errors without synthesizing success"
        )
    }
}

pub trait StreamTransformer {
    type Input;
    type Output;
    type Error;

    fn transform(&mut self, input: Self::Input) -> Result<Vec<Self::Output>, Self::Error>;

    fn finish(&mut self) -> Result<Vec<Self::Output>, Self::Error>;
}
