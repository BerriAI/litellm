use std::{
    future::Future,
    marker::PhantomData,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use bytes::{Bytes, BytesMut};
use futures_util::{StreamExt, TryStreamExt, stream};
use litellm_cache_response::{
    CacheOptions, CachePolicy, ResponseCacheRequest, ResponseCacheService, ResponseEnvelope,
    ScopedCache, cache_key,
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

/// A route whose response is the provider's reply. Only the body is cached: a replayed
/// response carries no provider headers.
pub trait Cachable: Protocol<Error = RouteError, Response = http::Response<Self::Body>> {
    const SURFACE: &'static str;
    type Body: Serialize + DeserializeOwned + Send + 'static;

    fn reusable(_response: &Self::Response) -> bool {
        true
    }
}

pub struct CacheRequest {
    pub identity: ProviderIdentity,
    pub input: Value,
}

impl CacheRequest {
    pub fn from_wire(identity: ProviderIdentity, wire: Option<&WireRequest>) -> Self {
        Self {
            input: wire.map_or(Value::Null, |wire| {
                serde_json::json!({
                    "provider": identity.provider,
                    "model": identity.model,
                    "url": wire.url,
                    "headers": wire.headers,
                    "body": wire.body,
                })
            }),
            identity,
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
}

impl CacheSession {
    fn prepare<P: Cachable>(
        service: Option<Arc<dyn ResponseCacheService>>,
        options: Option<CacheOptions>,
        request: &CacheRequest,
    ) -> Option<Self> {
        let options = options.filter(|options| options.policy.enabled())?;
        let service = service?;
        let input = request.input.clone();
        let request = options.request(&service.config().namespace, P::SURFACE, input);
        Some(Self { service, request })
    }

    async fn lookup<P: Cachable>(&self) -> Option<CachedOutput<P::Response>> {
        if !self.request.controls.reads() {
            return None;
        }
        match self.service.lookup(&self.request, now()).await {
            Ok(Some(value)) => {
                serde_json::from_value::<ResponseEnvelope<CachedOutput<P::Body>>>(value)
                    .ok()
                    .and_then(|entry| entry.decode(P::SURFACE))
                    .map(|output| match output {
                        CachedOutput::Response(body) => {
                            CachedOutput::Response(http::Response::new(body))
                        }
                        CachedOutput::Stream(data) => CachedOutput::Stream(data),
                    })
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

    async fn store_response<P: Cachable>(&self, response: &P::Response) {
        if !self.request.controls.writes() || !P::reusable(response) {
            return;
        }
        if let Ok(value) = serde_json::to_value(response.body())
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
    F: FnOnce() -> Fut,
    Fut: Future<Output = Result<P::Response, RouteError>>,
{
    let identity = request.identity.clone();
    crate::diagnostic::provider(&identity.model, &identity.provider);
    let session = CacheSession::prepare::<P>(cache, options, &request);
    let hit = match &session {
        Some(session) => session.lookup::<P>().await.and_then(|entry| match entry {
            CachedOutput::Response(response) => Some((response, cache_key(&session.request.key))),
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
    F: FnOnce() -> Fut,
    Fut: Future<Output = Result<OutputOf<P>, RouteError>>,
{
    let identity = request.identity.clone();
    crate::diagnostic::provider(&identity.model, &identity.provider);
    let session = CacheSession::prepare::<P>(cache, options, &request);
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
    pub fn from_wire(
        cache: Option<&ScopedCache>,
        policy: CachePolicy,
        identity: &ProviderIdentity,
        wire: &WireRequest,
    ) -> Self {
        let session = cache.and_then(|cache| {
            if !policy.enabled() {
                return None;
            }
            let options = cache.options(Some(policy));
            let request = CacheRequest::from_wire(identity.clone(), Some(wire));
            Some(CacheSession {
                request: options.request(
                    &cache.service.config().namespace,
                    P::SURFACE,
                    request.input,
                ),
                service: cache.service.clone(),
            })
        });
        Self {
            session,
            protocol: PhantomData,
        }
    }

    pub async fn lookup(&self) -> Option<(OutputOf<P>, ResultSource)> {
        let session = self.session.as_ref()?;
        let output = match session.lookup::<P>().await? {
            CachedOutput::Response(response) => CallOutput::Complete(response),
            CachedOutput::Stream(data) => P::replay(Bytes::from(data))?,
        };
        Some((
            output,
            ResultSource::Cache {
                key: cache_key(&session.request.key),
            },
        ))
    }

    pub async fn finish(self, output: OutputOf<P>, source: &ResultSource) -> OutputOf<P> {
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
