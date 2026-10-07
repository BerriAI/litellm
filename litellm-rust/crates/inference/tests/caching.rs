use std::{
    convert::Infallible,
    num::NonZeroUsize,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use bytes::Bytes;
use futures_util::{StreamExt, TryStreamExt, stream};
use litellm_cache_memory::InMemoryCache;
use litellm_cache_response::{
    CacheAccess, CacheCredential, CacheKey, CacheKeyInput, CacheOptions, CachePolicy, CacheScope,
    ResponseCache, ResponseCacheService,
};
use litellm_host::{
    call::{CallOutput, OutputOf},
    interceptors::{
        ExecutionFacts, Interceptors, ProviderIdentity, RawResponse, RequestContext, ResultSource,
        WireRequest,
    },
    lifecycle::{CallEvent, ExecutionEvent},
    observation::observation_channel,
    protocol::Protocol,
};
use litellm_inference::{
    RouteError,
    caching::{Cachable, CachePlan, StreamCachable},
};
use rstest::{fixture, rstest};
use serde_json::{Value, json};

struct TestRoute;

impl Protocol for TestRoute {
    type Request = Value;
    type Response = Value;
    type Error = RouteError;
    type HostCall = Infallible;
    type Chunk = Bytes;
    type StreamHead = ();
}

impl Cachable for TestRoute {
    const SURFACE: &'static str = "test";
}

impl StreamCachable for TestRoute {
    const TERMINAL_EVENT: &'static str = "message_stop";

    fn replay(data: Bytes) -> Option<OutputOf<Self>> {
        Some(CallOutput::Stream {
            head: (),
            chunks: stream::iter(litellm_framer::sse::RawBlocks::new(data).map(Ok)).boxed(),
        })
    }
    fn bytes(chunk: &Bytes) -> &[u8] {
        chunk
    }
}

struct CacheRequest {
    identity: ProviderIdentity,
    parameters: Value,
}

fn cache_request(input: Value) -> CacheRequest {
    CacheRequest {
        identity: ProviderIdentity {
            model: "test-model".into(),
            provider: "test-provider".into(),
        },
        parameters: input,
    }
}

fn plan<P: Cachable>(
    request: CacheRequest,
    cache: Option<Arc<dyn ResponseCacheService>>,
    options: Option<CacheOptions>,
) -> (ProviderIdentity, Option<CachePlan>) {
    let plan = cache.zip(options).and_then(|(cache, options)| {
        CachePlan::new(
            cache,
            options,
            CacheKeyInput::new(P::SURFACE, request.parameters),
        )
    });
    (request.identity, plan)
}

async fn execute_streaming<P, F, Fut>(
    request: CacheRequest,
    cache: Option<Arc<dyn ResponseCacheService>>,
    options: Option<CacheOptions>,
    interceptors: &impl Interceptors<RouteError>,
    observers: Option<&litellm_host::observation::ObservationSender>,
    provider: F,
) -> Result<OutputOf<P>, RouteError>
where
    P: StreamCachable,
    P::Response: serde::Serialize + serde::de::DeserializeOwned,
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = Result<OutputOf<P>, RouteError>>,
{
    let (identity, plan) = plan::<P>(request, cache, options);
    litellm_inference::caching::execute_streaming::<P, _, _>(
        identity,
        plan,
        interceptors,
        observers,
        provider,
    )
    .await
}

async fn execute_unary<P, F, Fut>(
    request: CacheRequest,
    cache: Option<Arc<dyn ResponseCacheService>>,
    options: Option<CacheOptions>,
    interceptors: &impl Interceptors<RouteError>,
    observers: Option<&litellm_host::observation::ObservationSender>,
    provider: F,
) -> Result<P::Response, RouteError>
where
    P: Cachable,
    P::Response: serde::Serialize + serde::de::DeserializeOwned,
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = Result<P::Response, RouteError>>,
{
    let (identity, plan) = plan::<P>(request, cache, options);
    litellm_inference::caching::execute_unary::<P, _, _>(
        identity,
        plan,
        interceptors,
        observers,
        provider,
    )
    .await
}

#[fixture]
fn cache() -> Arc<dyn ResponseCacheService> {
    cache_with_limit(4096)
}

fn cache_with_limit(max_entry_bytes: usize) -> Arc<dyn ResponseCacheService> {
    Arc::new(
        ResponseCache::new(Arc::new(InMemoryCache::new(
            Some(100),
            Some(Duration::from_secs(60)),
        )))
        .with_max_entry_bytes(max_entry_bytes),
    )
}

async fn call(
    cache: &Arc<dyn ResponseCacheService>,
    options: Option<CacheOptions>,
    calls: &AtomicUsize,
    request: Value,
) -> Value {
    let output = execute_streaming::<TestRoute, _, _>(
        cache_request(request),
        Some(cache.clone()),
        options,
        &(),
        None,
        || async {
            Ok(CallOutput::Complete(
                json!({"call": calls.fetch_add(1, Ordering::SeqCst)}),
            ))
        },
    )
    .await
    .unwrap();
    let CallOutput::Complete(response) = output else {
        panic!("expected a response");
    };
    response
}

#[rstest]
#[case::normal(CacheAccess::READ_WRITE)]
#[case::no_cache(CacheAccess { reads: false, writes: true })]
#[case::no_store(CacheAccess { reads: true, writes: false })]
#[case::disabled(CacheAccess::NONE)]
#[tokio::test]
async fn cache_controls_apply_to_both_reads_and_writes(
    cache: Arc<dyn ResponseCacheService>,
    #[case] access: CacheAccess,
) {
    let CacheAccess { reads, writes } = access;
    let calls = AtomicUsize::new(0);
    let options = Some(CacheOptions {
        policy: CachePolicy {
            access,
            ..CachePolicy::default()
        },
        ..CacheOptions::default()
    });
    let first = call(&cache, options.clone(), &calls, json!({"model":"test"})).await;
    let second = call(
        &cache,
        Some(CacheOptions::default()),
        &calls,
        json!({"model":"test"}),
    )
    .await;
    assert_eq!(first == second, writes);
    let third = call(&cache, options, &calls, json!({"model":"test"})).await;
    assert_eq!(second == third, reads);
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1 + usize::from(!writes) + usize::from(!reads)
    );
}

#[rstest]
#[tokio::test]
async fn request_identity_is_scoped(cache: Arc<dyn ResponseCacheService>) {
    let calls = AtomicUsize::new(0);
    let first = call(
        &cache,
        Some(CacheOptions::default()),
        &calls,
        json!({"model":"m", "input":{"a":1,"b":2}}),
    )
    .await;
    let other = call(
        &cache,
        Some(CacheOptions {
            scope: CacheScope {
                credential: Some(CacheCredential::new("test", "other-tenant", "key")),
            },
            ..CacheOptions::default()
        }),
        &calls,
        json!({"model":"m", "input":{"a":1,"b":2}}),
    )
    .await;
    assert_ne!(first, other);
    let changed = call(
        &cache,
        Some(CacheOptions::default()),
        &calls,
        json!({"model":"m", "input":{"a":2,"b":2}}),
    )
    .await;
    assert_ne!(first, changed);
}

async fn streamed(
    cache: &Arc<dyn ResponseCacheService>,
    calls: &AtomicUsize,
    text: &str,
    fail: bool,
) -> OutputOf<TestRoute> {
    execute_streaming::<TestRoute, _, _>(
        cache_request(json!({"stream":true})),
        Some(cache.clone()),
        Some(CacheOptions::default()),
        &(),
        None,
        || async {
            calls.fetch_add(1, Ordering::SeqCst);
            let chunks = text
                .as_bytes()
                .chunks(3)
                .map(|bytes| Ok(Bytes::copy_from_slice(bytes)))
                .collect::<Vec<_>>();
            let ending = fail.then_some(Err(RouteError::Unsupported("test transport failure")));
            Ok(CallOutput::Stream {
                head: (),
                chunks: stream::iter(chunks.into_iter().chain(ending)).boxed(),
            })
        },
    )
    .await
    .unwrap()
}

async fn consume(output: OutputOf<TestRoute>) -> Result<Vec<u8>, RouteError> {
    let CallOutput::Stream { chunks, .. } = output else {
        panic!("expected a stream");
    };
    chunks
        .try_fold(Vec::new(), |mut bytes, chunk| async move {
            bytes.extend_from_slice(&chunk);
            Ok(bytes)
        })
        .await
}

async fn collect_stream_chunks(output: OutputOf<TestRoute>) -> Vec<Bytes> {
    let CallOutput::Stream { chunks, .. } = output else {
        panic!("expected a stream");
    };
    chunks.try_collect().await.unwrap()
}

#[rstest]
#[tokio::test]
async fn cached_stream_replay_yields_one_chunk_per_event(cache: Arc<dyn ResponseCacheService>) {
    let calls = AtomicUsize::new(0);
    let first = b"data: {\"type\":\"content_block_delta\",\"text\":\"hello\"}\n\n";
    let terminal = b"data: {\"type\":\"message_stop\"}\n\n";
    let text = [first.as_slice(), terminal.as_slice()].concat();
    let text = std::str::from_utf8(&text).unwrap();

    assert_eq!(
        consume(streamed(&cache, &calls, text, false).await)
            .await
            .unwrap(),
        text.as_bytes()
    );
    assert_eq!(
        collect_stream_chunks(streamed(&cache, &calls, text, false).await).await,
        vec![Bytes::from_static(first), Bytes::from_static(terminal)]
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[rstest]
#[tokio::test]
async fn cached_stream_replay_preserves_utf8_split_across_provider_chunks(
    cache: Arc<dyn ResponseCacheService>,
) {
    let calls = AtomicUsize::new(0);
    let first = "data: {\"type\":\"content_block_delta\",\"text\":\"é🙂\"}\n\n".as_bytes();
    let terminal = b"data: {\"type\":\"message_stop\"}\n\n";
    let text = [first, terminal].concat();
    let text = std::str::from_utf8(&text).unwrap();

    assert_eq!(
        consume(streamed(&cache, &calls, text, false).await)
            .await
            .unwrap(),
        text.as_bytes()
    );
    let chunks = collect_stream_chunks(streamed(&cache, &calls, text, false).await).await;
    assert_eq!(
        chunks,
        vec![Bytes::copy_from_slice(first), Bytes::from_static(terminal)]
    );
    assert_eq!(
        chunks
            .iter()
            .flat_map(|chunk| chunk.iter())
            .copied()
            .collect::<Vec<_>>(),
        text.as_bytes()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[rstest]
#[tokio::test]
async fn dropping_cached_replay_after_first_event_polls_no_more_chunks(
    cache: Arc<dyn ResponseCacheService>,
) {
    let calls = AtomicUsize::new(0);
    let first = b"data: {\"type\":\"content_block_delta\",\"text\":\"first\"}\n\n";
    let terminal = b"data: {\"type\":\"message_stop\"}\n\n";
    let text = [first.as_slice(), terminal.as_slice()].concat();
    let text = std::str::from_utf8(&text).unwrap();
    let _stored = consume(streamed(&cache, &calls, text, false).await)
        .await
        .unwrap();
    let CallOutput::Stream { chunks, .. } = streamed(&cache, &calls, text, false).await else {
        panic!("expected a stream");
    };
    let polled = Arc::new(AtomicUsize::new(0));
    let count = polled.clone();
    let mut chunks = chunks
        .map(move |chunk| {
            count.fetch_add(1, Ordering::SeqCst);
            chunk
        })
        .boxed();

    assert_eq!(
        chunks.next().await.unwrap().unwrap(),
        Bytes::from_static(first)
    );
    drop(chunks);
    assert_eq!(polled.load(Ordering::SeqCst), 1);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[rstest]
#[case::complete("data: {\"type\":\"message_stop\"}\n\n", false, true)]
#[case::truncated("data: {\"type\":\"content_block_delta\"}\n\n", false, false)]
#[case::error_then_stop(
    "data: {\"type\":\"error\"}\n\ndata: {\"type\":\"message_stop\"}\n\n",
    false,
    false
)]
#[case::trailing_incomplete("data: {\"type\":\"message_stop\"}\n\ndata: {", false, false)]
#[case::transport_failure("data: {\"type\":\"message_stop\"}\n\n", true, false)]
#[tokio::test]
async fn stream_replay_requires_successful_exhaustion(
    cache: Arc<dyn ResponseCacheService>,
    #[case] text: &str,
    #[case] fail: bool,
    #[case] cached: bool,
) {
    let calls = AtomicUsize::new(0);
    let first = consume(streamed(&cache, &calls, text, fail).await).await;
    assert_eq!(first.is_err(), fail);
    let second = consume(streamed(&cache, &calls, text, fail).await).await;
    assert_eq!(second.is_err(), fail);
    if !fail {
        assert_eq!(first.unwrap(), second.unwrap());
    }
    assert_eq!(calls.load(Ordering::SeqCst), if cached { 1 } else { 2 });
}

#[rstest]
#[tokio::test]
async fn abandoning_a_partially_consumed_stream_does_not_store(
    cache: Arc<dyn ResponseCacheService>,
) {
    let calls = AtomicUsize::new(0);
    let text = "data: {\"type\":\"message_stop\"}\n\n";
    let CallOutput::Stream { mut chunks, .. } = streamed(&cache, &calls, text, false).await else {
        panic!();
    };
    assert!(chunks.next().await.unwrap().is_ok());
    drop(chunks);
    assert_eq!(
        consume(streamed(&cache, &calls, text, false).await)
            .await
            .unwrap(),
        text.as_bytes()
    );
    assert_eq!(
        consume(streamed(&cache, &calls, text, false).await)
            .await
            .unwrap(),
        text.as_bytes()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[rstest]
#[tokio::test]
async fn oversized_streams_are_delivered_without_being_stored() {
    let cache = cache_with_limit(8);
    let calls = AtomicUsize::new(0);
    let text = "data: {\"type\":\"message_stop\"}\n\n";
    assert_eq!(
        consume(streamed(&cache, &calls, text, false).await)
            .await
            .unwrap(),
        text.as_bytes()
    );
    assert_eq!(
        consume(streamed(&cache, &calls, text, false).await)
            .await
            .unwrap(),
        text.as_bytes()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[rstest]
#[tokio::test]
async fn a_provider_failure_never_populates_the_cache(cache: Arc<dyn ResponseCacheService>) {
    let calls = AtomicUsize::new(0);
    let first = execute_streaming::<TestRoute, _, _>(
        cache_request(json!({})),
        Some(cache.clone()),
        Some(CacheOptions::default()),
        &(),
        None,
        || async {
            calls.fetch_add(1, Ordering::SeqCst);
            Err(RouteError::Unsupported("test provider failure"))
        },
    )
    .await;
    assert!(first.is_err());
    let successful = call(&cache, Some(CacheOptions::default()), &calls, json!({})).await;
    let replayed = call(&cache, Some(CacheOptions::default()), &calls, json!({})).await;
    assert_eq!(successful, replayed);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

struct InvalidEntryCache(
    ResponseCache<InMemoryCache<litellm_cache_response::CacheEntry>>,
    Value,
);

impl ResponseCacheService for InvalidEntryCache {
    fn max_entry_bytes(&self) -> Option<usize> {
        self.0.max_entry_bytes()
    }

    fn key<'a>(
        &'a self,
        input: &'a CacheKeyInput,
        scope: &'a CacheScope,
    ) -> futures_util::future::BoxFuture<'a, Result<CacheKey, litellm_cache::Error>> {
        ResponseCacheService::key(&self.0, input, scope)
    }

    fn lookup<'a>(
        &'a self,
        key: &'a CacheKey,
        max_age: Option<Duration>,
        now: Duration,
    ) -> futures_util::future::BoxFuture<'a, Result<Option<Value>, litellm_cache::Error>> {
        Box::pin(async move {
            Ok(ResponseCacheService::lookup(&self.0, key, max_age, now)
                .await?
                .or_else(|| Some(self.1.clone())))
        })
    }

    fn store<'a>(
        &'a self,
        key: &'a CacheKey,
        ttl: Option<Duration>,
        response: Value,
        now: Duration,
    ) -> futures_util::future::BoxFuture<'a, Result<(), litellm_cache::Error>> {
        ResponseCacheService::store(&self.0, key, ttl, response, now)
    }
}

struct ResolveFailureCache {
    lookups: AtomicUsize,
    stores: AtomicUsize,
}

impl ResponseCacheService for ResolveFailureCache {
    fn max_entry_bytes(&self) -> Option<usize> {
        Some(4096)
    }

    fn key<'a>(
        &'a self,
        _: &'a CacheKeyInput,
        _: &'a CacheScope,
    ) -> futures_util::future::BoxFuture<'a, Result<CacheKey, litellm_cache::Error>> {
        Box::pin(async { Err(litellm_cache::Error::Unavailable) })
    }

    fn lookup<'a>(
        &'a self,
        _: &'a CacheKey,
        _: Option<Duration>,
        _: Duration,
    ) -> futures_util::future::BoxFuture<'a, Result<Option<Value>, litellm_cache::Error>> {
        self.lookups.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Ok(None) })
    }

    fn store<'a>(
        &'a self,
        _: &'a CacheKey,
        _: Option<Duration>,
        _: Value,
        _: Duration,
    ) -> futures_util::future::BoxFuture<'a, Result<(), litellm_cache::Error>> {
        self.stores.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Ok(()) })
    }
}

#[rstest]
#[tokio::test]
async fn key_resolution_failure_skips_cache_and_runs_provider() {
    let cache = Arc::new(ResolveFailureCache {
        lookups: AtomicUsize::new(0),
        stores: AtomicUsize::new(0),
    });
    let service: Arc<dyn ResponseCacheService> = cache.clone();
    let calls = AtomicUsize::new(0);
    let response = call(&service, Some(CacheOptions::default()), &calls, json!({})).await;
    assert_eq!(response["call"], json!(0));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(cache.lookups.load(Ordering::SeqCst), 0);
    assert_eq!(cache.stores.load(Ordering::SeqCst), 0);
}

#[rstest]
#[case::legacy(json!({"unexpected":"old-format"}))]
#[case::wrong_version(json!({"version":2,"surface":"test","output":{"kind":"Response","value":{"call":100}}}))]
#[case::wrong_surface(json!({"version":1,"surface":"other","output":{"kind":"Response","value":{"call":100}}}))]
#[tokio::test]
async fn an_invalid_cached_envelope_is_replaced_by_a_provider_result(#[case] poisoned: Value) {
    let cache: Arc<dyn ResponseCacheService> = Arc::new(InvalidEntryCache(
        ResponseCache::new(Arc::new(InMemoryCache::default())),
        poisoned,
    ));
    let request = json!({"input":"hello"});
    let calls = AtomicUsize::new(0);
    let first = call(
        &cache,
        Some(CacheOptions::default()),
        &calls,
        request.clone(),
    )
    .await;
    let second = call(&cache, Some(CacheOptions::default()), &calls, request).await;
    assert_eq!(first, second);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

struct UnavailableCache;

impl litellm_cache::BatchCache for UnavailableCache {}

impl litellm_cache::BaseCache for UnavailableCache {
    type Value = litellm_cache_response::CacheEntry;
    type Context = litellm_cache::ExactCacheContext;

    fn get_ttl(&self, _: &Self::Context) -> Option<Duration> {
        Some(Duration::from_secs(60))
    }

    fn get_cache(
        &self,
        _: &str,
        _: &Self::Context,
    ) -> Result<Option<Self::Value>, litellm_cache::Error> {
        Err(litellm_cache::Error::Unavailable)
    }

    fn set_cache(
        &self,
        _: &str,
        _: Self::Value,
        _: &Self::Context,
    ) -> Result<(), litellm_cache::Error> {
        Err(litellm_cache::Error::Unavailable)
    }
}

#[rstest]
#[tokio::test]
async fn backend_failures_do_not_fail_inference() {
    let cache: Arc<dyn ResponseCacheService> =
        Arc::new(ResponseCache::new(Arc::new(UnavailableCache)).with_max_entry_bytes(4096));
    let calls = AtomicUsize::new(0);
    let first = call(&cache, Some(CacheOptions::default()), &calls, json!({})).await;
    let second = call(&cache, Some(CacheOptions::default()), &calls, json!({})).await;
    assert_ne!(first, second);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

struct UnaryTestRoute;

#[derive(Default)]
struct CacheHitAccounting {
    calls: AtomicUsize,
    key: OnceLock<String>,
    reject: bool,
}

impl Interceptors<RouteError> for CacheHitAccounting {
    async fn result_ready(&self, facts: ExecutionFacts) -> Result<(), RouteError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let ResultSource::Cache { key } = facts.source else {
            panic!("expected cache source")
        };
        assert_eq!(
            facts.provider,
            ProviderIdentity {
                model: "test-model".into(),
                provider: "test-provider".into()
            }
        );
        self.key.set(key).unwrap();
        if self.reject {
            return Err(RouteError::Unsupported("cache accounting rejected"));
        }
        Ok(())
    }

    async fn before_provider_request(
        &self,
        wire: WireRequest,
        _: RequestContext,
    ) -> Result<WireRequest, RouteError> {
        Ok(wire)
    }

    async fn after_provider_response(&self, _: RawResponse) -> Result<(), RouteError> {
        Ok(())
    }
}

#[rstest]
#[case::unary(false)]
#[case::stream_replay(true)]
#[tokio::test]
async fn cache_hits_notify_accounting_once_and_propagate_its_failure(
    cache: Arc<dyn ResponseCacheService>,
    #[case] streaming_route: bool,
    #[values(false, true)] reject: bool,
) {
    let provider_calls = AtomicUsize::new(0);
    let accounting = CacheHitAccounting {
        reject,
        ..Default::default()
    };
    let (observer, mut events) = observation_channel(NonZeroUsize::new(4).unwrap());
    let request = if streaming_route {
        json!({"stream":true})
    } else {
        json!({"input":"hello"})
    };
    let expected = if streaming_route {
        json!(
            consume(
                streamed(
                    &cache,
                    &provider_calls,
                    "data: {\"type\":\"message_stop\"}\n\n",
                    false,
                )
                .await
            )
            .await
            .unwrap()
        )
    } else {
        unary_call(
            &cache,
            Some(CacheOptions::default()),
            &provider_calls,
            request.clone(),
        )
        .await
    };
    let result = if streaming_route {
        match execute_streaming::<TestRoute, _, _>(
            cache_request(request),
            Some(cache),
            Some(CacheOptions::default()),
            &accounting,
            Some(&observer),
            || async { panic!("a cache hit must not call the provider") },
        )
        .await
        {
            Ok(output) => consume(output).await.map(|bytes| json!(bytes)),
            Err(error) => Err(error),
        }
    } else {
        execute_unary::<UnaryTestRoute, _, _>(
            cache_request(request),
            Some(cache),
            Some(CacheOptions::default()),
            &accounting,
            Some(&observer),
            || async { panic!("a cache hit must not call the provider") },
        )
        .await
    };
    if reject {
        assert!(matches!(
            result,
            Err(RouteError::Unsupported("cache accounting rejected"))
        ));
    } else {
        assert_eq!(result.unwrap(), expected);
    }
    assert_eq!(provider_calls.load(Ordering::SeqCst), 1);
    assert_eq!(accounting.calls.load(Ordering::SeqCst), 1);
    let key = accounting.key.get().unwrap();
    assert!(!key.is_empty());
    assert!(matches!(
        events.try_recv().unwrap(),
        CallEvent::Execution(ExecutionEvent::ResultReady { facts }) if facts.source == ResultSource::Cache { key: key.clone() }
    ));
    assert!(events.try_recv().is_err());
}

impl Protocol for UnaryTestRoute {
    type Request = Value;
    type Response = Value;
    type Error = RouteError;
    type HostCall = Infallible;
    type Chunk = Infallible;
    type StreamHead = Infallible;
}

impl Cachable for UnaryTestRoute {
    const SURFACE: &'static str = "unary-test";
}

async fn unary_call(
    cache: &Arc<dyn ResponseCacheService>,
    options: Option<CacheOptions>,
    calls: &AtomicUsize,
    request: Value,
) -> Value {
    execute_unary::<UnaryTestRoute, _, _>(
        cache_request(request),
        Some(cache.clone()),
        options,
        &(),
        None,
        || async { Ok(json!({"call":calls.fetch_add(1, Ordering::SeqCst)})) },
    )
    .await
    .unwrap()
}

#[rstest]
#[case::normal(CacheAccess::READ_WRITE)]
#[case::no_cache(CacheAccess { reads: false, writes: true })]
#[case::no_store(CacheAccess { reads: true, writes: false })]
#[case::disabled(CacheAccess::NONE)]
#[tokio::test]
async fn unary_cache_controls_do_not_change_the_shared_service(
    cache: Arc<dyn ResponseCacheService>,
    #[case] access: CacheAccess,
) {
    let CacheAccess { reads, writes } = access;
    let calls = AtomicUsize::new(0);
    let options = Some(CacheOptions {
        policy: CachePolicy {
            access,
            ..CachePolicy::default()
        },
        ..CacheOptions::default()
    });
    let first = unary_call(&cache, options.clone(), &calls, json!({"input":"hello"})).await;
    let second = unary_call(
        &cache,
        Some(CacheOptions::default()),
        &calls,
        json!({"input":"hello"}),
    )
    .await;
    assert_eq!(first == second, writes);
    let third = unary_call(&cache, options, &calls, json!({"input":"hello"})).await;
    assert_eq!(second == third, reads);
    let fourth = unary_call(
        &cache,
        Some(CacheOptions::default()),
        &calls,
        json!({"input":"hello"}),
    )
    .await;
    assert_eq!(fourth, if !reads && writes { third } else { second });
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1 + usize::from(!writes) + usize::from(!reads)
    );
}

#[rstest]
#[tokio::test]
async fn surfaces_isolate_entries_on_shared_storage(cache: Arc<dyn ResponseCacheService>) {
    let calls = AtomicUsize::new(0);
    let first = call(
        &cache,
        Some(CacheOptions::default()),
        &calls,
        json!({"input":"hello"}),
    )
    .await;
    let different_surface = unary_call(
        &cache,
        Some(CacheOptions::default()),
        &calls,
        json!({"input":"hello"}),
    )
    .await;
    assert_ne!(first, different_surface);
    assert_eq!(
        call(
            &cache,
            Some(CacheOptions::default()),
            &calls,
            json!({"input":"hello"})
        )
        .await,
        first
    );
    assert_eq!(
        unary_call(
            &cache,
            Some(CacheOptions::default()),
            &calls,
            json!({"input":"hello"})
        )
        .await,
        different_surface
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}
