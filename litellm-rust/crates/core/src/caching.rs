use std::{
    future::Future,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use bytes::{Bytes, BytesMut};
use futures_util::{StreamExt, TryStreamExt, stream};
use litellm_cache_response::{
    CacheOptions, ResponseCacheRequest, ResponseCacheService, ResponseEnvelope, cache_key,
};
use litellm_host::{
    call::{CallOutput, OutputOf},
    interceptors::{ExecutionFacts, Interceptors, ProviderIdentity, ResultSource},
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

    fn provider(request: &Self::Request) -> Result<ProviderIdentity, RouteError>;

    fn cache_input(request: &Self::Request) -> Result<Value, RouteError>;

    fn reusable(_response: &Self::Response) -> bool {
        true
    }
}

pub trait StreamCachable: Cachable {
    const TERMINAL_EVENT: &'static str;

    fn replay(data: Bytes) -> Option<OutputOf<Self>>;
    fn bytes(chunk: &Self::Chunk) -> &[u8];
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", content = "value")]
enum CachedOutput<R> {
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
        request: &P::Request,
    ) -> Result<Option<Self>, RouteError> {
        let Some(options) = options.filter(CacheOptions::enabled) else {
            return Ok(None);
        };
        let Some(service) = service else {
            return Ok(None);
        };
        let input = P::cache_input(request)?;
        let request = options.request(&service.config().namespace, P::SURFACE, input);
        Ok(Some(Self { service, request }))
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
    request: P::Request,
    cache: Option<Arc<dyn ResponseCacheService>>,
    options: Option<CacheOptions>,
    interceptors: &impl Interceptors<RouteError>,
    observers: Option<&ObservationSender>,
    provider: F,
) -> Result<P::Response, RouteError>
where
    P: Cachable,
    P::Response: Serialize + DeserializeOwned,
    F: FnOnce(P::Request) -> Fut,
    Fut: Future<Output = Result<P::Response, RouteError>>,
{
    let identity = P::provider(&request)?;
    crate::diagnostic::provider(&identity.model, &identity.provider);
    let session = CacheSession::prepare::<P>(cache, options, &request)?;
    let hit = match &session {
        Some(session) => session.lookup::<P>().await.and_then(|entry| match entry {
            CachedOutput::Response(response) => Some((response, cache_key(&session.request.key))),
            CachedOutput::Stream(_) => None,
        }),
        None => None,
    };
    let (response, source) = match hit {
        Some((response, key)) => (response, ResultSource::Cache { key }),
        None => (provider(request).await?, ResultSource::Provider),
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
    request: P::Request,
    cache: Option<Arc<dyn ResponseCacheService>>,
    options: Option<CacheOptions>,
    interceptors: &impl Interceptors<RouteError>,
    observers: Option<&ObservationSender>,
    provider: F,
) -> Result<OutputOf<P>, RouteError>
where
    P: StreamCachable,
    P::Response: Serialize + DeserializeOwned,
    F: FnOnce(P::Request) -> Fut,
    Fut: Future<Output = Result<OutputOf<P>, RouteError>>,
{
    let identity = P::provider(&request)?;
    crate::diagnostic::provider(&identity.model, &identity.provider);
    let session = CacheSession::prepare::<P>(cache, options, &request)?;
    let hit = match &session {
        Some(session) => session.lookup::<P>().await.and_then(|entry| {
            let output = match entry {
                CachedOutput::Response(response) => Some(CallOutput::Complete(response)),
                CachedOutput::Stream(data) => P::replay(Bytes::from(data)),
            };
            output.map(|output| (output, cache_key(&session.request.key)))
        }),
        None => None,
    };
    let (output, source) = match hit {
        Some((output, key)) => (output, ResultSource::Cache { key }),
        None => (provider(request).await?, ResultSource::Provider),
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
    let Some(session) =
        session.filter(|session| from_provider && session.request.controls.writes())
    else {
        return Ok(output);
    };
    match output {
        CallOutput::Complete(response) => {
            session.store_response::<P>(&response).await;
            Ok(CallOutput::Complete(response))
        }
        CallOutput::Stream { head, chunks } => {
            let captured = stream::try_unfold(
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
            .boxed();
            Ok(CallOutput::Stream {
                head,
                chunks: captured,
            })
        }
    }
}

fn now() -> Duration {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
}

fn successful_stream(text: &str, terminal: &str) -> bool {
    let mut pending = BytesMut::from(text.as_bytes());
    let mut codec = litellm_framing::sse::SseCodec::default();
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
