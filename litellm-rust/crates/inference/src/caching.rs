use std::{
    future::Future,
    marker::PhantomData,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use bytes::{Bytes, BytesMut};
use futures_util::{StreamExt, TryStreamExt, stream};
use litellm_cache_response::{
    CacheKeyField, CacheKeyInput, CacheOptions, CachePolicy, ResponseCacheRequest,
    ResponseCacheService, ResponseEnvelope, ScopedCache,
};
use litellm_host::{
    call::{CallOutput, OutputOf},
    interceptors::{ExecutionFacts, Interceptors, ProviderIdentity, ResultSource, WireRequest},
    lifecycle::{CallEvent, ExecutionEvent},
    observation::ObservationSender,
    protocol::Protocol,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;
use tokio_util::codec::Decoder;

use crate::RouteError;

pub trait Cachable: Protocol<Error = RouteError> {
    const SURFACE: &'static str;

    fn reusable(_response: &Self::Response) -> bool {
        true
    }
}

pub trait CacheKeyProjection {
    fn cache_key_input(&self) -> Result<CacheKeyInput, RouteError>;
}

impl CacheKeyProjection for crate::chat_completions::types::ChatCompletionsRequest<'_> {
    fn cache_key_input(&self) -> Result<CacheKeyInput, RouteError> {
        Ok(CacheKeyInput::from_parameters(Value::Object(
            self.optional_params
                .clone()
                .into_iter()
                .chain([
                    ("model".into(), Value::String(self.model.into())),
                    ("messages".into(), self.messages.clone()),
                ])
                .collect(),
        )))
    }
}

impl CacheKeyProjection for crate::responses::types::ResponsesCall {
    fn cache_key_input(&self) -> Result<CacheKeyInput, RouteError> {
        Ok(CacheKeyInput::from_parameters(Value::Object(
            self.optional_params
                .clone()
                .into_iter()
                .chain([
                    ("model".into(), Value::String(self.model.clone())),
                    ("input".into(), self.input.clone()),
                ])
                .collect(),
        )))
    }
}

impl CacheKeyProjection for crate::messages::MessagesCall {
    fn cache_key_input(&self) -> Result<CacheKeyInput, RouteError> {
        serde_json::to_value(&self.body)
            .map(CacheKeyInput::from_parameters)
            .map_err(|error| RouteError::InvalidRequest(error.to_string().into()))
    }
}

pub struct CacheRequest {
    pub identity: ProviderIdentity,
    pub input: CacheKeyInput,
}

impl CacheRequest {
    pub(crate) fn from_logical(
        identity: ProviderIdentity,
        input: CacheKeyInput,
        original: Option<&WireRequest>,
        wire: &WireRequest,
    ) -> Self {
        let Some(original) = original else {
            return Self { identity, input };
        };
        let transport = serde_json::json!({ "provider": identity.provider, "url": wire.url, "headers": wire.headers });
        let changes = (original.url != wire.url || original.headers != wire.headers || original.body != wire.body)
            .then(|| serde_json::json!({ "url": wire.url, "headers": wire.headers, "body": wire.body }));
        Self {
            identity,
            input: CacheKeyInput {
                fields: input
                    .fields
                    .into_iter()
                    .chain([
                        CacheKeyField {
                            name: "transport".into(),
                            value: Some(transport.to_string()),
                            api_parameter: true,
                            internal_parameter: true,
                        },
                        CacheKeyField {
                            name: "wire_changes".into(),
                            value: changes.map(|value| value.to_string()),
                            api_parameter: true,
                            internal_parameter: true,
                        },
                    ])
                    .collect(),
                ..input
            },
        }
    }
}

pub trait StreamCachable: Cachable {
    const TERMINAL_EVENT: &'static str;

    fn replay(data: Bytes) -> Option<OutputOf<Self>>;
    fn bytes(chunk: &Self::Chunk) -> &[u8];
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", content = "value")]
pub enum CachedOutput<R> {
    Response(R),
    Stream(String),
}

struct CacheSession {
    service: Arc<dyn ResponseCacheService>,
    request: ResponseCacheRequest,
    key: String,
}

impl CacheSession {
    async fn prepare<P: Cachable>(
        service: Option<Arc<dyn ResponseCacheService>>,
        options: Option<CacheOptions>,
        request: &CacheRequest,
    ) -> Option<Self> {
        let options = options.filter(|options| options.policy.enabled())?;
        let service = service?;
        let input = request.input.clone();
        let request = options.request_with_key(&service.config().namespace, P::SURFACE, input);
        Self::resolve(service, request).await
    }

    async fn resolve(
        service: Arc<dyn ResponseCacheService>,
        request: ResponseCacheRequest,
    ) -> Option<Self> {
        let key = match service.resolve_key(&request).await {
            Ok(key) => key,
            Err(_) => {
                tracing::warn!("response cache lookup failed");
                return None;
            }
        };
        let ResponseCacheRequest {
            key: input,
            controls,
            context,
            max_age,
        } = request;
        let request = ResponseCacheRequest {
            key: CacheKeyInput {
                preset: Some(key.clone()),
                ..input
            },
            controls,
            context,
            max_age,
        };
        Some(Self {
            service,
            request,
            key,
        })
    }

    async fn lookup<P: Cachable>(&self) -> Option<CachedOutput<P::Response>>
    where
        P::Response: DeserializeOwned,
    {
        if !self.request.controls.reads() {
            return None;
        }
        match self.service.lookup(&self.request, now()).await {
            Ok(Some(value)) => {
                serde_json::from_value::<ResponseEnvelope<CachedOutput<P::Response>>>(value)
                    .ok()
                    .and_then(|entry| entry.decode(P::SURFACE))
            }
            Ok(None) => None,
            Err(_) => {
                tracing::warn!("response cache lookup failed");
                None
            }
        }
    }

    async fn store(&self, entry: Value) {
        if !self.request.controls.writes() {
            return;
        }
        if self
            .service
            .store(&self.request, entry, now())
            .await
            .is_err()
        {
            tracing::warn!("response cache write failed");
        }
    }

    async fn store_response<P: Cachable>(&self, response: &P::Response)
    where
        P::Response: Serialize,
    {
        if !self.request.controls.writes() || !P::reusable(response) {
            return;
        }
        if let Ok(value) = serde_json::to_value(response)
            && let Ok(entry) = serde_json::to_value(ResponseEnvelope::new(
                P::SURFACE,
                CachedOutput::Response(value),
            ))
        {
            self.store(entry).await;
        }
    }
}

pub async fn execute_unary<P, F, Fut>(
    request: CacheRequest,
    cache: Option<Arc<dyn ResponseCacheService>>,
    options: Option<CacheOptions>,
    interceptors: &impl Interceptors<RouteError>,
    observers: Option<&ObservationSender>,
    provider: F,
) -> Result<P::Response, RouteError>
where
    P: Cachable,
    P::Response: Serialize + DeserializeOwned,
    F: FnOnce() -> Fut,
    Fut: Future<Output = Result<P::Response, RouteError>>,
{
    let identity = request.identity.clone();
    crate::diagnostic::provider(&identity.model, &identity.provider);
    let session = CacheSession::prepare::<P>(cache, options, &request).await;
    let hit = match &session {
        Some(session) => session.lookup::<P>().await.and_then(|entry| match entry {
            CachedOutput::Response(response) => Some((response, session.key.clone())),
            CachedOutput::Stream(_) => None,
        }),
        None => None,
    };
    let (response, source) = match hit {
        Some((response, key)) => (response, ResultSource::Cache { key }),
        None => (provider().await?, ResultSource::Provider),
    };
    let from_provider = source == ResultSource::Provider;
    publish(
        ExecutionFacts {
            provider: identity,
            source,
        },
        interceptors,
        observers,
    )
    .await?;
    if from_provider && let Some(session) = session {
        session.store_response::<P>(&response).await;
    }
    Ok(response)
}

pub async fn execute_streaming<P, F, Fut>(
    request: CacheRequest,
    cache: Option<Arc<dyn ResponseCacheService>>,
    options: Option<CacheOptions>,
    interceptors: &impl Interceptors<RouteError>,
    observers: Option<&ObservationSender>,
    provider: F,
) -> Result<OutputOf<P>, RouteError>
where
    P: StreamCachable,
    P::Response: Serialize + DeserializeOwned,
    F: FnOnce() -> Fut,
    Fut: Future<Output = Result<OutputOf<P>, RouteError>>,
{
    let identity = request.identity.clone();
    crate::diagnostic::provider(&identity.model, &identity.provider);
    let session = CacheSession::prepare::<P>(cache, options, &request).await;
    let cache = CallCache::<P> {
        session,
        protocol: PhantomData,
    };
    let hit = cache.lookup().await;
    let (output, source) = match hit {
        Some(hit) => hit,
        None => (provider().await?, ResultSource::Provider),
    };
    publish(
        ExecutionFacts {
            provider: identity,
            source: source.clone(),
        },
        interceptors,
        observers,
    )
    .await?;
    Ok(cache.finish(output, &source).await)
}

pub struct CallCache<P> {
    session: Option<CacheSession>,
    protocol: PhantomData<P>,
}

impl<P: StreamCachable> CallCache<P> {
    pub(crate) async fn prepare(
        cache: Option<&ScopedCache>,
        policy: CachePolicy,
        input: CacheKeyInput,
    ) -> Self {
        let session = match cache.filter(|_| policy.enabled()) {
            Some(cache) => {
                let options = cache.options(Some(policy));
                let request =
                    options.request_with_key(&cache.service.config().namespace, P::SURFACE, input);
                CacheSession::resolve(cache.service.clone(), request).await
            }
            None => None,
        };
        Self {
            session,
            protocol: PhantomData,
        }
    }

    pub async fn lookup(&self) -> Option<(OutputOf<P>, ResultSource)>
    where
        P::Response: DeserializeOwned,
    {
        let session = self.session.as_ref()?;
        let output = match session.lookup::<P>().await? {
            CachedOutput::Response(response) => CallOutput::Complete(response),
            CachedOutput::Stream(data) => P::replay(Bytes::from(data))?,
        };
        Some((
            output,
            ResultSource::Cache {
                key: session.key.clone(),
            },
        ))
    }

    pub async fn finish(self, output: OutputOf<P>, source: &ResultSource) -> OutputOf<P>
    where
        P::Response: Serialize,
    {
        let Some(session) = self.session.filter(|session| {
            *source == ResultSource::Provider && session.request.controls.writes()
        }) else {
            return output;
        };
        match output {
            CallOutput::Complete(response) => {
                session.store_response::<P>(&response).await;
                CallOutput::Complete(response)
            }
            CallOutput::Stream { head, chunks } => CallOutput::Stream {
                head,
                chunks: capture_stream::<P>(chunks, session),
            },
        }
    }
}

fn capture_stream<P: StreamCachable>(
    chunks: futures_util::stream::BoxStream<'static, Result<P::Chunk, RouteError>>,
    session: CacheSession,
) -> futures_util::stream::BoxStream<'static, Result<P::Chunk, RouteError>> {
    stream::try_unfold(
        (chunks, Some(Vec::<u8>::new()), session),
        |(mut chunks, captured, session)| async move {
            match chunks.try_next().await? {
                Some(chunk) => {
                    let captured = captured.and_then(|mut data| {
                        let bytes = P::bytes(&chunk);
                        if data.len().saturating_add(bytes.len())
                            > session.service.config().max_entry_bytes
                        {
                            return None;
                        }
                        data.extend_from_slice(bytes);
                        Some(data)
                    });
                    Ok(Some((chunk, (chunks, captured, session))))
                }
                None => {
                    if let Some(data) = captured
                        && let Ok(text) = String::from_utf8(data)
                        && successful_stream(&text, P::TERMINAL_EVENT)
                        && let Ok(entry) = serde_json::to_value(ResponseEnvelope::new(
                            P::SURFACE,
                            CachedOutput::<Value>::Stream(text),
                        ))
                    {
                        session.store(entry).await;
                    }
                    Ok::<_, RouteError>(None)
                }
            }
        },
    )
    .boxed()
}

fn now() -> Duration {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
}

fn successful_stream(text: &str, terminal: &str) -> bool {
    let mut pending = BytesMut::from(text.as_bytes());
    let mut codec = litellm_framer::sse::SseCodec::default();
    let mut complete = false;
    loop {
        let event = match codec.decode(&mut pending) {
            Ok(Some(event)) => event,
            Ok(None) => return complete && pending.is_empty(),
            Err(_) => return false,
        };
        let Ok(value) = serde_json::from_str::<Value>(&event.data) else {
            return false;
        };
        let Some(kind) = value.get("type").and_then(Value::as_str) else {
            return false;
        };
        if matches!(kind, "error" | "response.failed" | "response.incomplete") {
            return false;
        }
        complete |= kind == terminal;
    }
}

async fn publish(
    facts: ExecutionFacts,
    interceptors: &impl Interceptors<RouteError>,
    observers: Option<&ObservationSender>,
) -> Result<(), RouteError> {
    if let Some(observers) = observers {
        observers.emit(CallEvent::Execution(ExecutionEvent::ResultReady {
            facts: facts.clone(),
        }));
    }
    interceptors.result_ready(facts).await
}
