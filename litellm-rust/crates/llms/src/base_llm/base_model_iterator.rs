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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApiDialect {
    ChatCompletions,
    Responses,
    AnthropicMessages,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferedSource {
    Cache,
    NonStreamingResponse,
    Mock,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeliveryMode {
    Live,
    Buffered(BufferedSource),
}

#[derive(Clone, Copy, Debug)]
pub struct StreamPolicy {
    pub max_buffer_bytes: usize,
    pub max_pending_events: usize,
    pub max_tool_argument_bytes: usize,
    pub deadline: Option<std::time::Instant>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StreamError {
    Upstream(String),
    InvalidEvent(String),
    UnexpectedEof,
    BufferLimitExceeded,
    Cancelled,
    DeadlineExceeded,
    Host(String),
}

pub enum StreamOutcome<Response> {
    Completed(Response),
    Incomplete(Response),
    Failed {
        error: StreamError,
        response: Option<Response>,
    },
    Cancelled {
        response: Option<Response>,
    },
}

pub struct MatchingDialectRelay {
    _dialect: ApiDialect,
}

impl MatchingDialectRelay {
    pub fn new(_caller: ApiDialect, _upstream: ApiDialect) -> Result<Self, StreamError> {
        todo!("Reject raw relay unless caller and upstream use the same semantic dialect")
    }
}

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
    policy: StreamPolicy,
}

impl<S, T: StreamTransformation> TransformedStream<S, T> {
    pub fn new(_upstream: S, _transformation: T, _policy: StreamPolicy) -> Self {
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
