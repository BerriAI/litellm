use litellm_llms::base_llm::base_model_iterator::{
    BufferedSource, StreamError, StreamOutcome, StreamPolicy,
};
use std::{
    pin::Pin,
    task::{Context, Poll},
};

use futures_util::Stream;
use litellm_types::responses::streaming::{ResponsesEvent, ResponsesResponse};

pub enum ResponsesStreamState {
    Pending,
    Active,
    Terminal(StreamOutcome<ResponsesResponse>),
    Failed(StreamError),
}

pub struct ResponsesApiStreamingIterator<S> {
    source: S,
    state: ResponsesStreamState,
    policy: StreamPolicy,
}

impl<S> ResponsesApiStreamingIterator<S> {
    pub fn new(_source: S, _policy: StreamPolicy) -> Self {
        todo!("Compose BaseResponsesAPIStreamingIterator lifecycle state with a typed event source")
    }

    pub fn process_chunk(
        &mut self,
        _event: ResponsesEvent,
    ) -> Result<Option<ResponsesEvent>, StreamError> {
        todo!(
            "Port base _process_chunk after provider decoding: response IDs, completed/failed/incomplete outcomes, usage, and error events"
        )
    }

    pub fn get_completed_response_object(&self) -> Option<&ResponsesResponse> {
        todo!("Expose the completed response retained by the shared lifecycle")
    }

    pub fn record_failed_response_usage(&mut self, _response: &ResponsesResponse) {
        todo!("Retain billable usage on failed and incomplete responses for host accounting")
    }
}

pub struct BufferedResponsesStream {
    _response: ResponsesResponse,
    _source: BufferedSource,
    _policy: StreamPolicy,
}

impl BufferedResponsesStream {
    pub fn new(
        _response: ResponsesResponse,
        _source: BufferedSource,
        _policy: StreamPolicy,
    ) -> Self {
        todo!(
            "Port cached/mock/simulated response event generation with explicit buffered delivery and stable item IDs"
        )
    }
}

impl Stream for BufferedResponsesStream {
    type Item = Result<ResponsesEvent, StreamError>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        todo!(
            "Emit bounded ordered lifecycle/content/tool events from the retained response, preserving its terminal status and usage"
        )
    }
}

impl<S> Stream for ResponsesApiStreamingIterator<S>
where
    S: Stream<Item = Result<ResponsesEvent, StreamError>>,
{
    type Item = Result<ResponsesEvent, StreamError>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        todo!(
            "Port __anext__: read typed events, enforce deadlines and terminal validation, and coordinate delivery, cache, logging and post-stream hooks through route-machine host operations"
        )
    }
}

pub struct NativeResponsesStream<S> {
    source: S,
}

impl<S> NativeResponsesStream<S> {
    pub fn new(_source: S) -> Self {
        todo!(
            "Accept canonical Responses events from provider decoding, independent of transport framing"
        )
    }
}

impl<S> Stream for NativeResponsesStream<S>
where
    S: Stream<Item = Result<ResponsesEvent, StreamError>>,
{
    type Item = Result<ResponsesEvent, StreamError>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        todo!(
            "Forward native typed events with backpressure and cancellation to the shared Responses driver"
        )
    }
}
