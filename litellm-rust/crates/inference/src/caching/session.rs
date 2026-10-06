use std::{
    future::Future,
    marker::PhantomData,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use bytes::Bytes;
use litellm_cache_response::{CacheKey, CachePolicy, ResponseCacheService, ResponseEnvelope};
use litellm_host::{
    call::{CallOutput, OutputOf},
    interceptors::{ExecutionFacts, Interceptors, ProviderIdentity, ResultSource},
    lifecycle::{CallEvent, ExecutionEvent},
    observation::ObservationSender,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;

use super::{Cachable, CachePlan, StreamCachable, stream::capture_stream};
use crate::RouteError;

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", content = "value")]
pub enum CachedOutput<R> {
    Response(R),
    Stream(String),
}

pub(crate) struct CacheSession<P> {
    service: Arc<dyn ResponseCacheService>,
    policy: CachePolicy,
    key: CacheKey,
    protocol: PhantomData<P>,
}

impl<P: Cachable> CacheSession<P> {
    pub(crate) async fn open(plan: Option<CachePlan>) -> Option<Self> {
        let CachePlan {
            service,
            options,
            input,
        } = plan?;
        let key = match service.key(&input, &options.scope).await {
            Ok(key) => key,
            Err(_) => {
                tracing::warn!("response cache lookup failed");
                return None;
            }
        };
        Some(Self {
            service,
            policy: options.policy,
            key,
            protocol: PhantomData,
        })
    }

    pub(super) fn max_entry_bytes(&self) -> usize {
        self.service.config().max_entry_bytes
    }

    fn hit(&self) -> ResultSource {
        ResultSource::Cache {
            key: self.key.as_str().to_owned(),
        }
    }

    async fn lookup(&self) -> Option<CachedOutput<P::Response>>
    where
        P::Response: DeserializeOwned,
    {
        if !self.policy.access().reads {
            return None;
        }
        match self
            .service
            .lookup(&self.key, self.policy.max_age, now())
            .await
        {
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

    pub(super) async fn store(&self, entry: Value) {
        if !self.policy.access().writes {
            return;
        }
        if self
            .service
            .store(&self.key, self.policy.ttl, entry, now())
            .await
            .is_err()
        {
            tracing::warn!("response cache write failed");
        }
    }

    async fn store_response(&self, response: &P::Response)
    where
        P::Response: Serialize,
    {
        if !self.policy.access().writes || !P::reusable(response) {
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

impl<P: StreamCachable> CacheSession<P> {
    pub(crate) async fn replay(session: Option<&Self>) -> Option<(OutputOf<P>, ResultSource)>
    where
        P::Response: DeserializeOwned,
    {
        let session = session?;
        let output = match session.lookup().await? {
            CachedOutput::Response(response) => CallOutput::Complete(response),
            CachedOutput::Stream(data) => P::replay(Bytes::from(data))?,
        };
        Some((output, session.hit()))
    }

    pub(crate) async fn finish(
        session: Option<Self>,
        output: OutputOf<P>,
        source: &ResultSource,
    ) -> OutputOf<P>
    where
        P::Response: Serialize,
    {
        let Some(session) = session
            .filter(|session| *source == ResultSource::Provider && session.policy.access().writes)
        else {
            return output;
        };
        match output {
            CallOutput::Complete(response) => {
                session.store_response(&response).await;
                CallOutput::Complete(response)
            }
            CallOutput::Stream { head, chunks } => CallOutput::Stream {
                head,
                chunks: capture_stream::<P>(chunks, session),
            },
        }
    }
}

pub async fn execute_unary<P, F, Fut>(
    identity: ProviderIdentity,
    plan: Option<CachePlan>,
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
    crate::diagnostic::provider(&identity.model, &identity.provider);
    let session = CacheSession::<P>::open(plan).await;
    let hit = match &session {
        Some(session) => match session.lookup().await {
            Some(CachedOutput::Response(response)) => Some((response, session.hit())),
            Some(CachedOutput::Stream(_)) | None => None,
        },
        None => None,
    };
    let (response, source) = match hit {
        Some(hit) => hit,
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
        session.store_response(&response).await;
    }
    Ok(response)
}

pub async fn execute_streaming<P, F, Fut>(
    identity: ProviderIdentity,
    plan: Option<CachePlan>,
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
    crate::diagnostic::provider(&identity.model, &identity.provider);
    let session = CacheSession::<P>::open(plan).await;
    let (output, source) = match CacheSession::replay(session.as_ref()).await {
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
    Ok(CacheSession::finish(session, output, &source).await)
}

fn now() -> Duration {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
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
