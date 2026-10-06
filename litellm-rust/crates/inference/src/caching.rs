use std::{
    collections::BTreeMap,
    future::Future,
    marker::PhantomData,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use bytes::{Bytes, BytesMut};
use futures_util::{StreamExt, TryStreamExt, stream};
use litellm_cache_response::{
    CacheKey, CacheKeyInput, CacheOptions, CachePolicy, CacheTarget, ResponseCacheRequest,
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
use serde_json::{Map, Value};
use tokio_util::codec::Decoder;

use crate::RouteError;

pub trait Cachable: Protocol<Error = RouteError> {
    const SURFACE: &'static str;

    fn reusable(_response: &Self::Response) -> bool {
        true
    }
}

pub trait CacheKeyProjection {
    fn cache_key_input(&self, model_group: Option<&str>) -> Result<CacheKeyInput, RouteError>;
}

fn key_input(
    target: CacheTarget,
    parameters: impl IntoIterator<Item = (String, Value)>,
    headers: impl IntoIterator<Item = (&'static str, Value)>,
) -> CacheKeyInput {
    CacheKeyInput::request(
        target,
        Value::Object(
            parameters
                .into_iter()
                .filter(|(name, _)| name != "model")
                .chain(
                    headers
                        .into_iter()
                        .map(|(name, value)| (name.to_owned(), value)),
                )
                .collect(),
        ),
    )
}

fn extra_headers(headers: Option<&Map<String, Value>>) -> Option<(&'static str, Value)> {
    headers.map(|headers| {
        (
            "extra_headers",
            Value::Object(
                headers
                    .iter()
                    .map(|(name, value)| (name.to_ascii_lowercase(), value.clone()))
                    .collect(),
            ),
        )
    })
}

impl CacheKeyProjection for crate::chat_completions::types::ChatCompletionsRequest<'_> {
    fn cache_key_input(&self, model_group: Option<&str>) -> Result<CacheKeyInput, RouteError> {
        Ok(key_input(
            CacheTarget::resolve(
                model_group,
                self.model,
                self.custom_llm_provider,
                self.api_base,
            ),
            self.optional_params
                .clone()
                .into_iter()
                .chain([("messages".into(), self.messages.clone())]),
            extra_headers(self.extra_headers.as_ref()),
        ))
    }
}

impl CacheKeyProjection for crate::responses::types::ResponsesCall {
    fn cache_key_input(&self, model_group: Option<&str>) -> Result<CacheKeyInput, RouteError> {
        Ok(key_input(
            CacheTarget::resolve(
                model_group,
                &self.model,
                self.custom_llm_provider.as_deref(),
                self.api_base.as_deref(),
            ),
            self.optional_params
                .clone()
                .into_iter()
                .chain([("input".into(), self.input.clone())]),
            extra_headers(self.extra_headers.as_ref()),
        ))
    }
}

impl CacheKeyProjection for crate::messages::MessagesCall {
    fn cache_key_input(&self, model_group: Option<&str>) -> Result<CacheKeyInput, RouteError> {
        let invalid =
            |error: serde_json::Error| RouteError::InvalidRequest(error.to_string().into());
        let Value::Object(body) = serde_json::to_value(&self.body).map_err(invalid)? else {
            return Err(RouteError::InvalidRequest(
                "messages body must be an object".into(),
            ));
        };
        let provider_specific_header = self
            .provider_specific_header
            .as_ref()
            .map(serde_json::to_value)
            .transpose()
            .map_err(invalid)?
            .map(|header| ("provider_specific_header", header));
        Ok(key_input(
            CacheTarget::resolve(
                model_group,
                &self.body.model,
                self.custom_llm_provider.as_deref(),
                self.api_base.as_deref(),
            ),
            body,
            extra_headers(self.extra_headers.as_ref())
                .into_iter()
                .chain(provider_specific_header),
        ))
    }
}

pub struct CacheRequest {
    pub identity: ProviderIdentity,
    pub input: Option<CacheKeyInput>,
}

pub(crate) struct RouteCache {
    cache: ScopedCache,
    policy: CachePolicy,
    input: CacheKeyInput,
}

impl RouteCache {
    pub(crate) fn attach(
        cache: Option<&ScopedCache>,
        policy: Option<CachePolicy>,
        request: &impl CacheKeyProjection,
        model_group: Option<&str>,
    ) -> Result<Option<Self>, RouteError> {
        let policy = policy.unwrap_or_default();
        let Some(cache) = cache.filter(|_| policy.enabled()) else {
            return Ok(None);
        };
        Ok(Some(Self {
            cache: cache.clone(),
            policy,
            input: request.cache_key_input(model_group)?,
        }))
    }

    pub(crate) fn guard(self, outbound: &WireRequest) -> GuardedCache {
        GuardedCache {
            sent: SentRequest::of(outbound),
            cache: self,
        }
    }

    pub(crate) fn into_call(
        cache: Option<Self>,
        identity: ProviderIdentity,
    ) -> (
        CacheRequest,
        Option<Arc<dyn ResponseCacheService>>,
        Option<CacheOptions>,
    ) {
        match cache {
            Some(Self {
                cache,
                policy,
                input,
            }) => (
                CacheRequest {
                    identity,
                    input: Some(input),
                },
                Some(cache.service().clone()),
                Some(cache.options(Some(policy))),
            ),
            None => (
                CacheRequest {
                    identity,
                    input: None,
                },
                None,
                None,
            ),
        }
    }
}

pub(crate) struct GuardedCache {
    cache: RouteCache,
    sent: SentRequest,
}

impl GuardedCache {
    pub(crate) fn confirm(self, wire: &WireRequest) -> Option<RouteCache> {
        if self.sent.matches(wire) {
            return Some(self.cache);
        }
        tracing::debug!("request changed before the provider call, skipping the response cache");
        None
    }
}

#[derive(PartialEq)]
struct SentRequest {
    url: String,
    headers: BTreeMap<String, String>,
    body: Value,
}

impl SentRequest {
    fn of(wire: &WireRequest) -> Self {
        Self {
            url: wire.url.clone(),
            headers: normalized_headers(&wire.headers),
            body: wire.body.clone(),
        }
    }

    fn matches(&self, wire: &WireRequest) -> bool {
        self.url == wire.url
            && self.body == wire.body
            && self.headers == normalized_headers(&wire.headers)
    }
}

fn normalized_headers(headers: &[(String, String)]) -> BTreeMap<String, String> {
    headers
        .iter()
        .map(|(name, value)| (name.to_ascii_lowercase(), value.clone()))
        .collect()
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
    key: CacheKey,
}

impl CacheSession {
    async fn prepare<P: Cachable>(
        service: Option<Arc<dyn ResponseCacheService>>,
        options: Option<CacheOptions>,
        input: Option<CacheKeyInput>,
    ) -> Option<Self> {
        let options = options.filter(|options| options.policy.enabled())?;
        let service = service?;
        let input = input?;
        Self::resolve(service, options.request(P::SURFACE, input)).await
    }

    async fn resolve(
        service: Arc<dyn ResponseCacheService>,
        request: ResponseCacheRequest,
    ) -> Option<Self> {
        let key = match service.key(&request).await {
            Ok(key) => key,
            Err(_) => {
                tracing::warn!("response cache lookup failed");
                return None;
            }
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
        if !self.request.access.reads {
            return None;
        }
        match self.service.lookup(&self.key, &self.request, now()).await {
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
        if !self.request.access.writes {
            return;
        }
        if self
            .service
            .store(&self.key, &self.request, entry, now())
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
        if !self.request.access.writes || !P::reusable(response) {
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
    let CacheRequest { identity, input } = request;
    crate::diagnostic::provider(&identity.model, &identity.provider);
    let session = CacheSession::prepare::<P>(cache, options, input).await;
    let hit = match &session {
        Some(session) => session.lookup::<P>().await.and_then(|entry| match entry {
            CachedOutput::Response(response) => Some((response, session.key.as_str().to_owned())),
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
    let CacheRequest { identity, input } = request;
    crate::diagnostic::provider(&identity.model, &identity.provider);
    let session = CacheSession::prepare::<P>(cache, options, input).await;
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
    pub(crate) async fn prepare(cache: Option<RouteCache>) -> Self {
        let session = match cache {
            Some(RouteCache {
                cache,
                policy,
                input,
            }) => {
                let request = cache.options(Some(policy)).request(P::SURFACE, input);
                CacheSession::resolve(cache.service().clone(), request).await
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
                key: session.key.as_str().to_owned(),
            },
        ))
    }

    pub async fn finish(self, output: OutputOf<P>, source: &ResultSource) -> OutputOf<P>
    where
        P::Response: Serialize,
    {
        let Some(session) = self
            .session
            .filter(|session| *source == ResultSource::Provider && session.request.access.writes)
        else {
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
