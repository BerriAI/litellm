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
pub enum StreamApi {
    ChatCompletions,
    Messages,
    Responses,
}

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

impl TryFrom<&str> for StreamApi {
    type Error = Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "chat_completions" => Ok(Self::ChatCompletions),
            "messages" => Ok(Self::Messages),
            "responses" => Ok(Self::Responses),
            _ => Err(Error::InvalidRequest(format!(
                "unknown streaming API: {value}"
            ))),
        }
    }
}

impl TryFrom<&str> for StreamTransport {
    type Error = Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "http" => Ok(Self::Http),
            "websocket" => Ok(Self::WebSocket),
            _ => Err(Error::InvalidRequest(format!(
                "unknown streaming transport: {value}"
            ))),
        }
    }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamCapability {
    pub api: StreamApi,
    pub provider: StreamProviderId,
    pub transport: StreamTransport,
}

pub const PLANNED_STREAM_CAPABILITIES: [StreamCapability; 6] = [
    StreamCapability {
        api: StreamApi::ChatCompletions,
        provider: StreamProviderId::Anthropic,
        transport: StreamTransport::Http,
    },
    StreamCapability {
        api: StreamApi::ChatCompletions,
        provider: StreamProviderId::BedrockConverse,
        transport: StreamTransport::Http,
    },
    StreamCapability {
        api: StreamApi::Messages,
        provider: StreamProviderId::Anthropic,
        transport: StreamTransport::Http,
    },
    StreamCapability {
        api: StreamApi::Messages,
        provider: StreamProviderId::AzureAi,
        transport: StreamTransport::Http,
    },
    StreamCapability {
        api: StreamApi::Responses,
        provider: StreamProviderId::OpenAi,
        transport: StreamTransport::Http,
    },
    StreamCapability {
        api: StreamApi::Responses,
        provider: StreamProviderId::OpenAi,
        transport: StreamTransport::WebSocket,
    },
];

pub fn supports_streaming(_capability: StreamCapability) -> bool {
    false
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct JsonObject(pub Map<String, Value>);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Header {
    pub name: String,
    pub value: String,
}

#[derive(Clone, Default, PartialEq, Deserialize)]
pub struct ProviderCredentials {
    pub api_key: Option<String>,
    pub aws_access_key_id: Option<String>,
    pub aws_secret_access_key: Option<String>,
    pub aws_session_token: Option<String>,
}

#[derive(Clone, PartialEq)]
pub struct ProviderCallContext {
    pub provider: StreamProviderId,
    pub credentials: ProviderCredentials,
    pub api_base: Option<String>,
    pub extra_headers: Vec<Header>,
    pub timeout: Option<Duration>,
    pub metadata: JsonObject,
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
    use std::sync::{Arc, Mutex};

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

    #[test]
    fn planned_capabilities_are_explicit_but_disabled() {
        assert_eq!(PLANNED_STREAM_CAPABILITIES.len(), 6);
        assert!(
            PLANNED_STREAM_CAPABILITIES
                .into_iter()
                .all(|capability| !supports_streaming(capability))
        );
    }
}
