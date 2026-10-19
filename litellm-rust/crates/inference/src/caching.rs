use std::{
    future::Future,
    marker::PhantomData,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use bytes::{Bytes, BytesMut};
use futures_util::{StreamExt, TryStreamExt, stream};
use litellm_cache_response::{
    CacheOptions, CachePolicy, PreparedCache, ResponseCacheService, ResponseEnvelope, ScopedCache,
};
use litellm_host::{
    call::{CallOutput, OutputOf},
    interceptors::{
        Cost, ExecutionFacts, InterceptPoints, Interceptors, ProviderIdentity, ResultSource,
        WireRequest,
    },
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

pub struct CacheRequest {
    pub identity: ProviderIdentity,
    session: Option<CacheSession>,
}

impl CacheRequest {
    pub fn prepare<P: Cachable>(
        identity: ProviderIdentity,
        service: Option<Arc<dyn ResponseCacheService>>,
        options: Option<CacheOptions>,
        input: &dyn Fn(&ProviderIdentity) -> Value,
    ) -> Self {
        let session = options
            .filter(|options| options.policy.enabled())
            .zip(service)
            .map(|(options, service)| CacheSession {
                prepared: service.prepare(options, P::SURFACE, &|| input(&identity)),
            });
        Self { identity, session }
    }

    pub fn from_wire<P: Cachable>(
        identity: ProviderIdentity,
        wire: &WireRequest,
        cache: Option<&ScopedCache>,
        policy: Option<CachePolicy>,
    ) -> Self {
        Self::prepare::<P>(
            identity,
            cache.map(|cache| cache.service.clone()),
            cache.map(|cache| cache.options(policy)),
            &|identity| {
                serde_json::json!({
                    "provider": identity.provider,
                    "model": identity.model,
                    "url": wire.url,
                    "headers": wire.headers,
                    "body": wire.body,
                })
            },
        )
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
    prepared: PreparedCache,
}

impl CacheSession {
    async fn lookup<P: Cachable>(&self) -> Option<(CachedOutput<P::Response>, String)>
    where
        P::Response: DeserializeOwned,
    {
        if !self.prepared.policy.reads() {
            return None;
        }
        match self.prepared.call.lookup(now()).await {
            Ok(Some(hit)) => {
                serde_json::from_value::<ResponseEnvelope<CachedOutput<P::Response>>>(hit.value)
                    .ok()
                    .and_then(|entry| entry.decode(P::SURFACE))
                    .map(|output| (output, hit.key))
            }
            Ok(None) => None,
            Err(_) => {
                tracing::warn!("response cache lookup failed");
                None
            }
        }
    }

    async fn store(&self, entry: Value) {
        if !self.prepared.policy.writes() {
            return;
        }
        if self.prepared.call.store(entry, now()).await.is_err() {
            tracing::warn!("response cache write failed");
        }
    }

    async fn store_response<P: Cachable>(&self, response: &P::Response)
    where
        P::Response: Serialize,
    {
        if !self.prepared.policy.writes() || !P::reusable(response) {
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
    let CacheRequest { identity, session } = request;
    crate::diagnostic::provider(&identity.model, &identity.provider);
    let hit = match &session {
        Some(session) => session
            .lookup::<P>()
            .await
            .and_then(|(entry, key)| match entry {
                CachedOutput::Response(response) => Some((response, key)),
                CachedOutput::Stream(_) => None,
            }),
        None => None,
    };
    let (response, source) = match hit {
        Some((response, key)) => (response, ResultSource::Cache { key }),
        None => (provider().await?, ResultSource::Provider),
    };
    let from_provider = source == ResultSource::Provider;
    let cost = Cost::settle(&source, None);
    publish(
        ExecutionFacts {
            provider: identity,
            source,
            cost,
            gateway_injected_cache_control: false,
        },
        interceptors,
        observers,
    )
    .await?;
    if !from_provider {
        return Ok(response);
    }
    let response = interceptors.provider_result(response).await?;
    if let Some(session) = session {
        session.store_response::<P>(&response).await;
    }
    Ok(response)
}

pub async fn execute_streaming<P, F, Fut>(
    request: CacheRequest,
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
    let CacheRequest { identity, session } = request;
    crate::diagnostic::provider(&identity.model, &identity.provider);
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
            cost: Cost::settle(&source, None),
            gateway_injected_cache_control: false,
            source: source.clone(),
        },
        interceptors,
        observers,
    )
    .await?;
    let output = provider_result::<P>(output, &source, interceptors).await?;
    Ok(cache.finish(output, &source).await)
}

/// Offers a complete provider response to the host before it is cached or delivered. A cache
/// hit and a stream pass through untouched.
pub async fn provider_result<P>(
    output: OutputOf<P>,
    source: &ResultSource,
    interceptors: &impl Interceptors<RouteError>,
) -> Result<OutputOf<P>, RouteError>
where
    P: Protocol<Error = RouteError>,
    P::Response: Serialize + DeserializeOwned,
{
    match (source, output) {
        (ResultSource::Provider, CallOutput::Complete(response)) => Ok(CallOutput::Complete(
            interceptors.provider_result(response).await?,
        )),
        (_, output) => Ok(output),
    }
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
        let request = CacheRequest::from_wire::<P>(identity.clone(), wire, cache, Some(policy));
        Self {
            session: request.session,
            protocol: PhantomData,
        }
    }

    pub async fn lookup(&self) -> Option<(OutputOf<P>, ResultSource)>
    where
        P::Response: DeserializeOwned,
    {
        let session = self.session.as_ref()?;
        let (entry, key) = session.lookup::<P>().await?;
        let output = match entry {
            CachedOutput::Response(response) => CallOutput::Complete(response),
            CachedOutput::Stream(data) => P::replay(Bytes::from(data))?,
        };
        Some((output, ResultSource::Cache { key }))
    }

    pub async fn finish(self, output: OutputOf<P>, source: &ResultSource) -> OutputOf<P>
    where
        P::Response: Serialize,
    {
        let Some(session) = self.session.filter(|session| {
            *source == ResultSource::Provider && session.prepared.policy.writes()
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
                        if data.len().saturating_add(bytes.len()) > session.prepared.max_entry_bytes
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
