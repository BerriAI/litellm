use std::collections::VecDeque;
use std::marker::PhantomData;
use std::pin::Pin;
use std::time::Duration;

use crate::error::Error;
use futures_util::future::BoxFuture;
use futures_util::{Stream, StreamExt, stream};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub type EventStream<E> = Pin<Box<dyn Stream<Item = Result<E, Error>> + Send + 'static>>;
pub type ProviderChunkStream =
    Pin<Box<dyn Stream<Item = Result<ProviderStreamChunk, Error>> + Send + 'static>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamTransport {
    Http,
    WebSocket,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamProviderId {
    Anthropic,
    AzureAi,
    BedrockConverse,
    OpenAi,
}

impl TryFrom<&str> for StreamProviderId {
    type Error = Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "anthropic" => Ok(Self::Anthropic),
            "azure_ai" => Ok(Self::AzureAi),
            "bedrock" | "bedrock_converse" => Ok(Self::BedrockConverse),
            "openai" => Ok(Self::OpenAi),
            _ => Err(Error::InvalidProvider(value.to_string())),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct JsonObject(pub Map<String, Value>);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Header {
    pub name: String,
    pub value: String,
}

#[derive(Clone, Default, PartialEq)]
pub struct ProviderCredentials {
    api_key: Option<String>,
    aws_access_key_id: Option<String>,
    aws_secret_access_key: Option<String>,
    aws_session_token: Option<String>,
}

impl ProviderCredentials {
    pub fn new(
        api_key: Option<String>,
        aws_access_key_id: Option<String>,
        aws_secret_access_key: Option<String>,
        aws_session_token: Option<String>,
    ) -> Self {
        Self {
            api_key,
            aws_access_key_id,
            aws_secret_access_key,
            aws_session_token,
        }
    }

    pub fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
    }

    pub fn aws_access_key_id(&self) -> Option<&str> {
        self.aws_access_key_id.as_deref()
    }

    pub fn aws_secret_access_key(&self) -> Option<&str> {
        self.aws_secret_access_key.as_deref()
    }

    pub fn aws_session_token(&self) -> Option<&str> {
        self.aws_session_token.as_deref()
    }
}

/// ```compile_fail
/// fn assert_serialize<T: serde::Serialize>() {}
/// assert_serialize::<litellm_core::streaming::ProviderCredentials>();
/// assert_serialize::<litellm_core::streaming::StreamTarget>();
/// assert_serialize::<litellm_core::streaming::StreamTransportOptions>();
/// ```
///
/// ```compile_fail
/// use litellm_core::streaming::{JsonObject, ProviderCredentials, StreamProviderId, StreamTarget};
/// let mut target = StreamTarget::new(
///     StreamProviderId::OpenAi,
///     ProviderCredentials::default(),
///     None,
/// );
/// target.metadata = JsonObject::default();
/// ```
#[derive(Clone, PartialEq)]
pub struct StreamTarget {
    provider: StreamProviderId,
    credentials: ProviderCredentials,
    api_base: Option<String>,
}

impl StreamTarget {
    pub fn new(
        provider: StreamProviderId,
        credentials: ProviderCredentials,
        api_base: Option<String>,
    ) -> Self {
        Self {
            provider,
            credentials,
            api_base,
        }
    }

    pub fn provider(&self) -> StreamProviderId {
        self.provider
    }

    pub fn credentials(&self) -> &ProviderCredentials {
        &self.credentials
    }

    pub fn api_base(&self) -> Option<&str> {
        self.api_base.as_deref()
    }
}

#[derive(Clone, Default, PartialEq)]
pub struct StreamTransportOptions {
    forwarded_headers: Vec<Header>,
    timeout: Option<Duration>,
}

impl StreamTransportOptions {
    pub fn new(forwarded_headers: Vec<Header>, timeout: Option<Duration>) -> Self {
        Self {
            forwarded_headers,
            timeout,
        }
    }

    pub fn forwarded_headers(&self) -> &[Header] {
        &self.forwarded_headers
    }

    pub fn timeout(&self) -> Option<Duration> {
        self.timeout
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamMetadata {
    pub status_code: u16,
    pub provider: StreamProviderId,
    pub transport: StreamTransport,
    pub response_headers: Vec<Header>,
}

pub struct OpenedStream<E> {
    pub metadata: StreamMetadata,
    pub events: EventStream<E>,
}

pub struct OpenedWireStream {
    pub metadata: StreamMetadata,
    pub chunks: ProviderChunkStream,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderStreamChunk(Vec<u8>);

impl ProviderStreamChunk {
    pub fn new(bytes: impl Into<Vec<u8>>) -> Self {
        Self(bytes.into())
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

pub trait StreamDecoder: Send + 'static {
    type WireEvent: Send + 'static;

    fn push(&mut self, chunk: ProviderStreamChunk) -> Result<Vec<Self::WireEvent>, Error>;

    fn finish(&mut self) -> Result<Vec<Self::WireEvent>, Error> {
        Ok(Vec::new())
    }
}

pub trait StreamProvider<R, E>: Send + Sync + 'static {
    type PreparedRequest: Send + 'static;
    type WireEvent: Send + 'static;
    type Decoder: StreamDecoder<WireEvent = Self::WireEvent>;

    fn transform_request(&self, request: R) -> Result<Self::PreparedRequest, Error>;

    fn call(
        &'static self,
        request: Self::PreparedRequest,
    ) -> BoxFuture<'static, Result<OpenedWireStream, Error>>;

    fn decoder(&self) -> Self::Decoder;

    fn normalize(&self, event: Self::WireEvent) -> Result<Vec<E>, Error>;
}

struct PipelineState<P: 'static, D, E, R>
where
    D: StreamDecoder,
{
    provider: &'static P,
    decoder: D,
    chunks: ProviderChunkStream,
    pending: VecDeque<Result<E, Error>>,
    finished: bool,
    request: PhantomData<fn(R)>,
}

pub async fn open_provider_stream<P, R, E>(
    provider: &'static P,
    request: R,
) -> Result<OpenedStream<E>, Error>
where
    P: StreamProvider<R, E>,
    R: Send + 'static,
    E: Send + 'static,
{
    let prepared = provider.transform_request(request)?;
    let opened = provider.call(prepared).await?;
    let state = PipelineState {
        provider,
        decoder: provider.decoder(),
        chunks: opened.chunks,
        pending: VecDeque::new(),
        finished: false,
        request: PhantomData,
    };
    let events = stream::unfold(state, |mut state| async move {
        loop {
            if let Some(event) = state.pending.pop_front() {
                return Some((event, state));
            }
            if state.finished {
                return None;
            }
            match state.chunks.next().await {
                Some(Ok(chunk)) => match state.decoder.push(chunk) {
                    Ok(events) => queue_normalized(&mut state, events),
                    Err(error) => {
                        state.finished = true;
                        return Some((Err(error), state));
                    }
                },
                Some(Err(error)) => {
                    state.finished = true;
                    return Some((Err(error), state));
                }
                None => {
                    state.finished = true;
                    match state.decoder.finish() {
                        Ok(events) => queue_normalized(&mut state, events),
                        Err(error) => return Some((Err(error), state)),
                    }
                }
            }
        }
    });
    Ok(OpenedStream {
        metadata: opened.metadata,
        events: Box::pin(events),
    })
}

fn queue_normalized<P, D, E, R>(state: &mut PipelineState<P, D, E, R>, events: Vec<D::WireEvent>)
where
    P: StreamProvider<R, E, Decoder = D>,
    D: StreamDecoder<WireEvent = <P as StreamProvider<R, E>>::WireEvent>,
{
    for event in events {
        match state.provider.normalize(event) {
            Ok(normalized) => state.pending.extend(normalized.into_iter().map(Ok)),
            Err(error) => {
                state.pending.push_back(Err(error));
                state.finished = true;
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::pin::Pin;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::task::{Context, Poll};

    use futures_util::future::FutureExt;

    use super::*;

    struct FakeRequest;
    struct PreparedRequest;

    struct FakeProvider {
        calls: Arc<Mutex<Vec<&'static str>>>,
    }

    struct FakeDecoder {
        calls: Arc<Mutex<Vec<&'static str>>>,
        pending: String,
    }

    #[derive(Clone, Copy)]
    enum FailurePoint {
        Transform,
        Call,
        Chunk,
        Push,
        Finish,
        Normalize,
    }

    struct FailureProvider(FailurePoint);

    struct FailureDecoder(FailurePoint);

    impl StreamDecoder for FailureDecoder {
        type WireEvent = String;

        fn push(&mut self, chunk: ProviderStreamChunk) -> Result<Vec<Self::WireEvent>, Error> {
            if matches!(self.0, FailurePoint::Push) {
                return Err(Error::InvalidResponse("decoder push failed".to_string()));
            }
            Ok(vec![
                String::from_utf8(chunk.0).expect("test chunk should be UTF-8"),
            ])
        }

        fn finish(&mut self) -> Result<Vec<Self::WireEvent>, Error> {
            if matches!(self.0, FailurePoint::Finish) {
                return Err(Error::InvalidResponse("decoder finish failed".to_string()));
            }
            Ok(Vec::new())
        }
    }

    impl StreamProvider<FakeRequest, String> for FailureProvider {
        type PreparedRequest = PreparedRequest;
        type WireEvent = String;
        type Decoder = FailureDecoder;

        fn transform_request(&self, _request: FakeRequest) -> Result<Self::PreparedRequest, Error> {
            if matches!(self.0, FailurePoint::Transform) {
                return Err(Error::InvalidRequest("transform failed".to_string()));
            }
            Ok(PreparedRequest)
        }

        fn call(
            &'static self,
            _request: Self::PreparedRequest,
        ) -> BoxFuture<'static, Result<OpenedWireStream, Error>> {
            async move {
                if matches!(self.0, FailurePoint::Call) {
                    return Err(Error::Network("open failed".to_string()));
                }
                let chunks: ProviderChunkStream = match self.0 {
                    FailurePoint::Chunk => Box::pin(stream::iter([
                        Err(Error::Network("source failed".to_string())),
                        Ok(ProviderStreamChunk::new("ignored")),
                    ])),
                    FailurePoint::Finish => Box::pin(stream::empty()),
                    _ => Box::pin(stream::iter([Ok(ProviderStreamChunk::new("event"))])),
                };
                Ok(OpenedWireStream {
                    metadata: StreamMetadata {
                        status_code: 200,
                        provider: StreamProviderId::OpenAi,
                        transport: StreamTransport::Http,
                        response_headers: Vec::new(),
                    },
                    chunks,
                })
            }
            .boxed()
        }

        fn decoder(&self) -> Self::Decoder {
            FailureDecoder(self.0)
        }

        fn normalize(&self, event: Self::WireEvent) -> Result<Vec<String>, Error> {
            if matches!(self.0, FailurePoint::Normalize) {
                return Err(Error::InvalidResponse("normalize failed".to_string()));
            }
            Ok(vec![event])
        }
    }

    struct PendingUntilDropped(Arc<AtomicBool>);

    impl Stream for PendingUntilDropped {
        type Item = Result<ProviderStreamChunk, Error>;

        fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            Poll::Pending
        }
    }

    impl Drop for PendingUntilDropped {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    struct PendingProvider(Arc<AtomicBool>);

    impl StreamProvider<FakeRequest, String> for PendingProvider {
        type PreparedRequest = PreparedRequest;
        type WireEvent = String;
        type Decoder = FailureDecoder;

        fn transform_request(&self, _request: FakeRequest) -> Result<Self::PreparedRequest, Error> {
            Ok(PreparedRequest)
        }

        fn call(
            &'static self,
            _request: Self::PreparedRequest,
        ) -> BoxFuture<'static, Result<OpenedWireStream, Error>> {
            async move {
                Ok(OpenedWireStream {
                    metadata: StreamMetadata {
                        status_code: 200,
                        provider: StreamProviderId::OpenAi,
                        transport: StreamTransport::Http,
                        response_headers: Vec::new(),
                    },
                    chunks: Box::pin(PendingUntilDropped(self.0.clone())),
                })
            }
            .boxed()
        }

        fn decoder(&self) -> Self::Decoder {
            FailureDecoder(FailurePoint::Push)
        }

        fn normalize(&self, event: Self::WireEvent) -> Result<Vec<String>, Error> {
            Ok(vec![event])
        }
    }

    impl StreamDecoder for FakeDecoder {
        type WireEvent = String;

        fn push(&mut self, chunk: ProviderStreamChunk) -> Result<Vec<Self::WireEvent>, Error> {
            self.calls.lock().expect("call log").push("decode");
            self.pending
                .push_str(std::str::from_utf8(chunk.as_bytes()).expect("test utf-8"));
            let mut parts = self
                .pending
                .split('|')
                .map(str::to_string)
                .collect::<Vec<_>>();
            self.pending = parts.pop().expect("split always returns one item");
            Ok(parts)
        }

        fn finish(&mut self) -> Result<Vec<Self::WireEvent>, Error> {
            if self.pending.is_empty() {
                return Ok(Vec::new());
            }
            Ok(vec![std::mem::take(&mut self.pending)])
        }
    }

    impl StreamProvider<FakeRequest, String> for FakeProvider {
        type PreparedRequest = PreparedRequest;
        type WireEvent = String;
        type Decoder = FakeDecoder;

        fn transform_request(&self, _request: FakeRequest) -> Result<Self::PreparedRequest, Error> {
            self.calls.lock().expect("call log").push("transform");
            Ok(PreparedRequest)
        }

        fn call(
            &'static self,
            _request: Self::PreparedRequest,
        ) -> BoxFuture<'static, Result<OpenedWireStream, Error>> {
            self.calls.lock().expect("call log").push("call");
            async move {
                Ok(OpenedWireStream {
                    metadata: StreamMetadata {
                        status_code: 200,
                        provider: StreamProviderId::Anthropic,
                        transport: StreamTransport::Http,
                        response_headers: vec![Header {
                            name: "x-test".to_string(),
                            value: "ready".to_string(),
                        }],
                    },
                    chunks: Box::pin(stream::iter([
                        Ok(ProviderStreamChunk::new(b"one|tw".to_vec())),
                        Ok(ProviderStreamChunk::new(b"o|three".to_vec())),
                    ])),
                })
            }
            .boxed()
        }

        fn decoder(&self) -> Self::Decoder {
            FakeDecoder {
                calls: self.calls.clone(),
                pending: String::new(),
            }
        }

        fn normalize(&self, event: Self::WireEvent) -> Result<Vec<String>, Error> {
            self.calls.lock().expect("call log").push("normalize");
            Ok(vec![event.to_uppercase()])
        }
    }

    #[tokio::test]
    async fn fake_provider_proves_pipeline_order_and_fragmentation() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let provider = Box::leak(Box::new(FakeProvider {
            calls: calls.clone(),
        }));
        let mut opened = open_provider_stream(provider, FakeRequest)
            .await
            .expect("stream opens");
        let events = opened
            .events
            .by_ref()
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .expect("events normalize");

        assert_eq!(events, ["ONE", "TWO", "THREE"]);
        assert_eq!(opened.metadata.response_headers[0].name, "x-test");
        assert_eq!(
            *calls.lock().expect("call log"),
            [
                "transform",
                "call",
                "decode",
                "normalize",
                "decode",
                "normalize",
                "normalize",
            ]
        );
    }

    #[tokio::test]
    async fn request_and_open_failures_stop_before_a_stream_is_returned() {
        for (point, expected) in [
            (FailurePoint::Transform, "invalid request: transform failed"),
            (FailurePoint::Call, "upstream network error: open failed"),
        ] {
            let provider = Box::leak(Box::new(FailureProvider(point)));
            let error = match open_provider_stream(provider, FakeRequest).await {
                Ok(_) => panic!("failure should prevent the stream from opening"),
                Err(error) => error,
            };
            assert_eq!(error.to_string(), expected);
        }
    }

    #[tokio::test]
    async fn pipeline_failures_are_emitted_once_and_then_terminate() {
        for (point, expected) in [
            (FailurePoint::Chunk, "upstream network error: source failed"),
            (FailurePoint::Push, "invalid response: decoder push failed"),
            (
                FailurePoint::Finish,
                "invalid response: decoder finish failed",
            ),
            (
                FailurePoint::Normalize,
                "invalid response: normalize failed",
            ),
        ] {
            let provider = Box::leak(Box::new(FailureProvider(point)));
            let mut opened = open_provider_stream(provider, FakeRequest)
                .await
                .expect("stream should open before its terminal failure");
            let error = opened
                .events
                .next()
                .await
                .expect("stream should emit its error")
                .expect_err("first event should be the configured failure");
            assert_eq!(error.to_string(), expected);
            assert!(opened.events.next().await.is_none());
        }
    }

    #[tokio::test]
    async fn dropping_the_event_stream_drops_the_provider_chunk_stream() {
        let dropped = Arc::new(AtomicBool::new(false));
        let provider = Box::leak(Box::new(PendingProvider(dropped.clone())));
        let opened = open_provider_stream(provider, FakeRequest)
            .await
            .expect("stream should open");

        drop(opened.events);

        assert!(dropped.load(Ordering::SeqCst));
    }
}
