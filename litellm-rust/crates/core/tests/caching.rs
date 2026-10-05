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
    CacheOptions, CachePolicy, CacheScope, ResponseCache, ResponseCacheConfig,
    ResponseCacheService, ResponseEnvelope,
};
use litellm_core::{
    RouteError,
    caching::{Cachable, CacheRequest, StreamCachable, execute_streaming, execute_unary},
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
            chunks: stream::iter([Ok(data)]).boxed(),
        })
    }
    fn bytes(chunk: &Bytes) -> &[u8] {
        chunk
    }
}

fn cache_request(input: Value) -> CacheRequest {
    CacheRequest {
        identity: ProviderIdentity {
            model: "test-model".into(),
            provider: "test-provider".into(),
        },
        input,
    }
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
        .with_config(ResponseCacheConfig {
            namespace: "test".into(),
            max_entry_bytes,
        }),
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
#[case::normal(CacheOptions::new(CacheScope::Shared), true, true)]
#[case::no_cache(CacheOptions { policy: CachePolicy { no_cache: true, ..CachePolicy::default() }, ..CacheOptions::new(CacheScope::Shared) }, false, true)]
#[case::no_store(CacheOptions { policy: CachePolicy { no_store: true, ..CachePolicy::default() }, ..CacheOptions::new(CacheScope::Shared) }, true, false)]
#[case::disabled(CacheOptions { policy: CachePolicy { caching: Some(false), ..CachePolicy::default() }, ..CacheOptions::new(CacheScope::Shared) }, false, false)]
#[tokio::test]
async fn cache_controls_apply_to_both_reads_and_writes(
    cache: Arc<dyn ResponseCacheService>,
    #[case] options: CacheOptions,
    #[case] reads: bool,
    #[case] writes: bool,
) {
    let calls = AtomicUsize::new(0);
    let options = Some(options);
    let first = call(&cache, options.clone(), &calls, json!({"model":"test"})).await;
    let second = call(
        &cache,
        Some(CacheOptions::new(CacheScope::Shared)),
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
async fn request_identity_is_canonical_and_scoped(cache: Arc<dyn ResponseCacheService>) {
    let calls = AtomicUsize::new(0);
    let first = call(
        &cache,
        Some(CacheOptions::new(CacheScope::Shared)),
        &calls,
        json!({"model":"m", "input":{"a":1,"b":2}}),
    )
    .await;
    let second = call(
        &cache,
        Some(CacheOptions::new(CacheScope::Shared)),
        &calls,
        json!({"input":{"b":2,"a":1}, "model":"m"}),
    )
    .await;
    assert_eq!(first, second);
    let other = call(
        &cache,
        Some(CacheOptions {
            scope: CacheScope::Isolated("other-tenant".into()),
            ..CacheOptions::new(CacheScope::Shared)
        }),
        &calls,
        json!({"model":"m", "input":{"a":1,"b":2}}),
    )
    .await;
    assert_ne!(first, other);
    let changed = call(
        &cache,
        Some(CacheOptions::new(CacheScope::Shared)),
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
        Some(CacheOptions::new(CacheScope::Shared)),
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
        Some(CacheOptions::new(CacheScope::Shared)),
        &(),
        None,
        || async {
            calls.fetch_add(1, Ordering::SeqCst);
            Err(RouteError::Unsupported("test provider failure"))
        },
    )
    .await;
    assert!(first.is_err());
    let successful = call(
        &cache,
        Some(CacheOptions::new(CacheScope::Shared)),
        &calls,
        json!({}),
    )
    .await;
    let replayed = call(
        &cache,
        Some(CacheOptions::new(CacheScope::Shared)),
        &calls,
        json!({}),
    )
    .await;
    assert_eq!(successful, replayed);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

struct InvalidEntryCache(
    ResponseCache<InMemoryCache<litellm_cache_response::CacheEntry>>,
    Value,
);

impl ResponseCacheService for InvalidEntryCache {
    fn config(&self) -> &ResponseCacheConfig {
        self.0.config()
    }

    fn lookup<'a>(
        &'a self,
        request: &'a litellm_cache_response::ResponseCacheRequest,
        now: Duration,
    ) -> futures_util::future::BoxFuture<'a, Result<Option<Value>, litellm_cache::Error>> {
        Box::pin(async move {
            Ok(self
                .0
                .async_lookup(request, now)
                .await?
                .or_else(|| Some(self.1.clone())))
        })
    }

    fn store<'a>(
        &'a self,
        request: &'a litellm_cache_response::ResponseCacheRequest,
        response: Value,
        now: Duration,
    ) -> futures_util::future::BoxFuture<'a, Result<(), litellm_cache::Error>> {
        Box::pin(self.0.async_store(request, response, now))
    }
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
        Some(CacheOptions::new(CacheScope::Shared)),
        &calls,
        request.clone(),
    )
    .await;
    let second = call(
        &cache,
        Some(CacheOptions::new(CacheScope::Shared)),
        &calls,
        request,
    )
    .await;
    assert_eq!(first, second);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[rstest]
#[case::chat_completion(json!({"kind":"Response","value":{"id":"chat-1","model":"test","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}}))]
#[case::wrong_envelope(json!({"kind":"Stream","value":"data: [DONE]\n\n"}))]
#[tokio::test]
async fn responses_refetches_instead_of_deserializing_another_api_response(
    #[case] poisoned: Value,
) {
    use litellm_core::responses::route::Responses;
    use litellm_llms_types::formats::responses::ResponsesApiResponse;

    let cache: Arc<dyn ResponseCacheService> = Arc::new(InvalidEntryCache(
        ResponseCache::new(Arc::new(InMemoryCache::default())),
        serde_json::to_value(ResponseEnvelope::new("responses", poisoned)).unwrap(),
    ));
    let calls = AtomicUsize::new(0);
    for _ in 0..2 {
        let response = execute_unary::<Responses, _, _>(
            cache_request(json!({"input":"hello"})),
            Some(cache.clone()),
            Some(CacheOptions::new(CacheScope::Shared)),
            &(),
            None,
            || async {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(ResponsesApiResponse {
                    id: "fresh-response".into(),
                    model: "test".into(),
                    output: vec![
                        json!({"type":"message","content":[{"type":"output_text","text":"fresh"}]}),
                    ],
                    extra: [("status".into(), json!("completed"))]
                        .into_iter()
                        .collect(),
                })
            },
        )
        .await
        .unwrap();
        assert_eq!(response.id, "fresh-response");
        assert_eq!(response.output[0]["content"][0]["text"], "fresh");
    }
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[rstest]
#[case::system("system", json!("answer ALPHA"), json!("answer BETA"))]
#[case::stop_sequences("stop_sequences", json!(["STOP"]), json!(["END"]))]
#[case::top_k("top_k", json!(5), json!(10))]
#[case::tools("tools", json!([{"name":"a","input_schema":{"type":"object"}}]), json!([{"name":"b","input_schema":{"type":"object"}}]))]
#[case::tool_choice("tool_choice", json!({"type":"auto"}), json!({"type":"none"}))]
#[tokio::test]
async fn messages_cache_identity_includes_provider_native_parameters(
    cache: Arc<dyn ResponseCacheService>,
    #[case] field: &str,
    #[case] original: Value,
    #[case] changed: Value,
) {
    use litellm_core::messages::route::Messages;
    use litellm_llms_types::formats::messages::MessagesResponse;

    let calls = AtomicUsize::new(0);
    for (value, expected_call) in [(original.clone(), 0), (changed, 1), (original, 0)] {
        let response =
            execute_unary::<Messages, _, _>(
                CacheRequest::from_wire(
                    ProviderIdentity {
                        model: "test".into(),
                        provider: "anthropic".into(),
                    },
                    Some(&WireRequest {
                        url: "https://example.test/v1/messages".into(),
                        headers: vec![],
                        body: json!({
                            "model":"test", "messages":[{"role":"user","content":"hello"}],
                            "max_tokens":32, (field):value
                        }),
                    }),
                ),
                Some(cache.clone()),
                Some(CacheOptions::new(CacheScope::Shared)),
                &(),
                None,
                || async {
                    let call = calls.fetch_add(1, Ordering::SeqCst);
                    Ok(Box::new(serde_json::from_value::<MessagesResponse>(json!({
                    "id":call.to_string(), "type":"message", "role":"assistant", "model":"test",
                    "content":[{"type":"text","text":format!("answer {call}")}],
                    "stop_reason":"end_turn", "stop_sequence":null
                })).unwrap()))
                },
            )
            .await
            .unwrap();
        assert_eq!(response.id, expected_call.to_string());
        assert_eq!(
            response.content[0]["text"],
            format!("answer {expected_call}")
        );
    }
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

struct UnavailableCache;

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
    let cache: Arc<dyn ResponseCacheService> = Arc::new(
        ResponseCache::new(Arc::new(UnavailableCache)).with_config(ResponseCacheConfig {
            namespace: "test".into(),
            max_entry_bytes: 4096,
        }),
    );
    let calls = AtomicUsize::new(0);
    let first = call(
        &cache,
        Some(CacheOptions::new(CacheScope::Shared)),
        &calls,
        json!({}),
    )
    .await;
    let second = call(
        &cache,
        Some(CacheOptions::new(CacheScope::Shared)),
        &calls,
        json!({}),
    )
    .await;
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
            Some(CacheOptions::new(CacheScope::Shared)),
            &provider_calls,
            request.clone(),
        )
        .await
    };
    let result = if streaming_route {
        match execute_streaming::<TestRoute, _, _>(
            cache_request(request),
            Some(cache),
            Some(CacheOptions::new(CacheScope::Shared)),
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
            Some(CacheOptions::new(CacheScope::Shared)),
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
#[case::normal(CacheOptions::new(CacheScope::Shared), true, true)]
#[case::no_cache(CacheOptions { policy: CachePolicy { no_cache: true, ..CachePolicy::default() }, ..CacheOptions::new(CacheScope::Shared) }, false, true)]
#[case::no_store(CacheOptions { policy: CachePolicy { no_store: true, ..CachePolicy::default() }, ..CacheOptions::new(CacheScope::Shared) }, true, false)]
#[case::disabled(CacheOptions { policy: CachePolicy { caching: Some(false), ..CachePolicy::default() }, ..CacheOptions::new(CacheScope::Shared) }, false, false)]
#[tokio::test]
async fn unary_cache_controls_do_not_change_the_shared_service(
    cache: Arc<dyn ResponseCacheService>,
    #[case] options: CacheOptions,
    #[case] reads: bool,
    #[case] writes: bool,
) {
    let calls = AtomicUsize::new(0);
    let options = Some(options);
    let first = unary_call(&cache, options.clone(), &calls, json!({"input":"hello"})).await;
    let second = unary_call(
        &cache,
        Some(CacheOptions::new(CacheScope::Shared)),
        &calls,
        json!({"input":"hello"}),
    )
    .await;
    assert_eq!(first == second, writes);
    let third = unary_call(&cache, options, &calls, json!({"input":"hello"})).await;
    assert_eq!(second == third, reads);
    let fourth = unary_call(
        &cache,
        Some(CacheOptions::new(CacheScope::Shared)),
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
async fn namespaces_and_surfaces_isolate_entries_on_shared_storage() {
    let storage = Arc::new(InMemoryCache::default());
    let first_cache: Arc<dyn ResponseCacheService> = Arc::new(
        ResponseCache::new(storage.clone()).with_config(ResponseCacheConfig {
            namespace: "first".into(),
            max_entry_bytes: 4096,
        }),
    );
    let second_cache: Arc<dyn ResponseCacheService> = Arc::new(
        ResponseCache::new(storage).with_config(ResponseCacheConfig {
            namespace: "second".into(),
            max_entry_bytes: 4096,
        }),
    );
    let calls = AtomicUsize::new(0);
    let first = call(
        &first_cache,
        Some(CacheOptions::new(CacheScope::Shared)),
        &calls,
        json!({"input":"hello"}),
    )
    .await;
    let different_namespace = call(
        &second_cache,
        Some(CacheOptions::new(CacheScope::Shared)),
        &calls,
        json!({"input":"hello"}),
    )
    .await;
    let different_surface = unary_call(
        &first_cache,
        Some(CacheOptions::new(CacheScope::Shared)),
        &calls,
        json!({"input":"hello"}),
    )
    .await;
    assert_ne!(first, different_namespace);
    assert_ne!(first, different_surface);
    assert_eq!(
        call(
            &first_cache,
            Some(CacheOptions::new(CacheScope::Shared)),
            &calls,
            json!({"input":"hello"})
        )
        .await,
        first
    );
    assert_eq!(
        call(
            &second_cache,
            Some(CacheOptions::new(CacheScope::Shared)),
            &calls,
            json!({"input":"hello"})
        )
        .await,
        different_namespace
    );
    assert_eq!(
        unary_call(
            &first_cache,
            Some(CacheOptions::new(CacheScope::Shared)),
            &calls,
            json!({"input":"hello"})
        )
        .await,
        different_surface
    );
    assert_eq!(calls.load(Ordering::SeqCst), 3);
}

#[rstest]
#[case::completed("completed", 1)]
#[case::incomplete("incomplete", 2)]
#[tokio::test]
async fn responses_cache_only_reuses_completed_responses(
    cache: Arc<dyn ResponseCacheService>,
    #[case] status: &str,
    #[case] expected_calls: usize,
) {
    use litellm_core::responses::route::Responses;
    use litellm_llms_types::formats::responses::ResponsesApiResponse;

    let calls = AtomicUsize::new(0);
    for _ in 0..2 {
        let response = execute_unary::<Responses, _, _>(
            cache_request(json!({"input":"hello"})),
            Some(cache.clone()),
            Some(CacheOptions::new(CacheScope::Shared)),
            &(),
            None,
            || async {
                let call = calls.fetch_add(1, Ordering::SeqCst);
                Ok(ResponsesApiResponse {
                    id: call.to_string(),
                    model: "test".into(),
                    output: Vec::new(),
                    extra: [("status".into(), json!(status))].into_iter().collect(),
                })
            },
        )
        .await
        .unwrap();
        assert_eq!(response.extra.get("status"), Some(&json!(status)));
    }
    assert_eq!(calls.load(Ordering::SeqCst), expected_calls);
}

mod support;
use support::traces;

#[rstest]
#[case::without_cache(false)]
#[case::with_cache(true)]
#[tokio::test]
async fn the_same_route_entrypoint_reports_facts_with_or_without_caching(
    cache: Arc<dyn ResponseCacheService>,
    #[case] caching: bool,
    traces: support::TraceCapture,
) {
    use litellm_cache_response::ScopedCache;
    use litellm_core::chat_completions::types::ChatCompletionsRequest;
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

    let upstream = MockServer::start().await;
    let body = json!({"id":"msg-test","type":"message","role":"assistant","model":"cache-test-model",
        "content":[{"type":"text","text":"cached answer"}],"stop_reason":"end_turn",
        "stop_sequence":null,"usage":{"input_tokens":11,"output_tokens":4}});
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(if caching { 1 } else { 2 })
        .mount(&upstream)
        .await;
    let route = support::chat_completions_route();
    let route = if caching {
        route.with_cache(ScopedCache::new(cache, CacheScope::Shared))
    } else {
        route
    };
    let (observer, mut events) = observation_channel(NonZeroUsize::new(16).unwrap());
    let base = upstream.uri();
    for _ in 0..2 {
        let response = traces
            .logger()
            .instrument(route.execute(
                ChatCompletionsRequest {
                    model: "anthropic/cache-test-model",
                    messages: json!([{"role":"user","content":"hello"}]),
                    optional_params: [("max_tokens".into(), json!(16))].into_iter().collect(),
                    api_key: Some("test-key"),
                    api_base: Some(&base),
                    custom_llm_provider: None,
                    extra_headers: None,
                    timeout: None,
                },
                &(),
                Some(observer.clone()),
            ))
            .await
            .unwrap();
        assert_eq!(
            serde_json::to_value(response).unwrap()["usage"]["total_tokens"],
            15
        );
    }
    let facts: Vec<_> = std::iter::from_fn(|| events.try_recv().ok())
        .filter_map(|event| match event {
            CallEvent::Execution(ExecutionEvent::ResultReady { facts }) => Some(facts),
            _ => None,
        })
        .collect();
    assert_eq!(facts.len(), 2);
    assert_eq!(
        facts[0].provider,
        ProviderIdentity {
            model: "cache-test-model".into(),
            provider: "anthropic".into()
        }
    );
    assert_eq!(facts[1].provider, facts[0].provider);
    assert_eq!(facts[0].source, ResultSource::Provider);
    match &facts[1].source {
        ResultSource::Provider => assert!(!caching),
        ResultSource::Cache { key } => {
            assert!(caching);
            assert!(!key.is_empty());
        }
    }
    let summaries = traces.summaries("litellm.route");
    assert_eq!(summaries.len(), 2);
    for summary in summaries {
        assert_eq!(summary["provider"], "anthropic");
        assert_eq!(summary["resolved_model"], "cache-test-model");
        assert_eq!(summary["outcome"], "success");
    }
    upstream.verify().await;
}

struct ChangingSecrets {
    revision: AtomicUsize,
    endpoints: [String; 2],
    change_credentials: bool,
}

impl litellm_secrets::source::SecretSource for ChangingSecrets {
    fn get_secret_str<'a>(
        &'a self,
        name: &'a str,
    ) -> futures_util::future::BoxFuture<
        'a,
        Result<Option<litellm_secrets::SecretValue>, litellm_secrets::Error>,
    > {
        Box::pin(async move {
            let revision = self.revision.load(Ordering::SeqCst);
            let value = if name.ends_with("_API_KEY") {
                Some(format!(
                    "key-{}",
                    if self.change_credentials { revision } else { 0 }
                ))
            } else if name.ends_with("_API_BASE") {
                Some(self.endpoints[revision].clone())
            } else {
                None
            };
            Ok(value.map(litellm_secrets::SecretValue::new))
        })
    }
}

#[derive(Default)]
struct ChangingHooks {
    calls: AtomicUsize,
    rewrite: bool,
    facts: std::sync::Mutex<Vec<ExecutionFacts>>,
}

impl Interceptors<RouteError> for ChangingHooks {
    async fn before_provider_request(
        &self,
        mut wire: WireRequest,
        _: RequestContext,
    ) -> Result<WireRequest, RouteError> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        if self.rewrite {
            wire.body["temperature"] = json!(if call < 2 { 0.1 } else { 0.8 });
        }
        Ok(wire)
    }

    async fn after_provider_response(&self, _: RawResponse) -> Result<(), RouteError> {
        Ok(())
    }

    async fn result_ready(&self, facts: ExecutionFacts) -> Result<(), RouteError> {
        self.facts.lock().unwrap().push(facts);
        Ok(())
    }
}

#[rstest]
#[case::chat_credentials("chat", "credentials")]
#[case::chat_endpoint("chat", "endpoint")]
#[case::chat_callback("chat", "callback")]
#[case::messages_credentials("messages", "credentials")]
#[case::messages_endpoint("messages", "endpoint")]
#[case::messages_callback("messages", "callback")]
#[case::responses_credentials("responses", "credentials")]
#[case::responses_endpoint("responses", "endpoint")]
#[case::responses_callback("responses", "callback")]
#[tokio::test]
async fn cache_identity_follows_resolved_configuration_and_request_callbacks(
    cache: Arc<dyn ResponseCacheService>,
    #[case] surface: &str,
    #[case] change: &str,
) {
    use litellm_cache_response::ScopedCache;
    use litellm_core::{
        chat_completions::{ChatCompletionsRoute, types::ChatCompletionsRequest},
        messages::MessagesCall,
        responses::types::ResponsesCall,
    };
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

    let first = MockServer::start().await;
    let second = MockServer::start().await;
    let response = if surface == "responses" {
        json!({"id":"response-test", "model":"test", "output":[], "status":"completed"})
    } else {
        json!({"id":"message-test", "type":"message", "role":"assistant", "model":"test",
            "content":[{"type":"text", "text":"answer"}], "stop_reason":"end_turn", "stop_sequence":null,
            "usage":{"input_tokens":3,"output_tokens":2}})
    };
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response.clone()))
        .expect(if change == "endpoint" { 1 } else { 2 })
        .mount(&first)
        .await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response))
        .expect(if change == "endpoint" { 1 } else { 0 })
        .mount(&second)
        .await;
    let secrets = Arc::new(ChangingSecrets {
        revision: AtomicUsize::new(0),
        endpoints: [
            first.uri(),
            if change == "endpoint" {
                second.uri()
            } else {
                first.uri()
            },
        ],
        change_credentials: change == "credentials",
    });
    let hooks = ChangingHooks {
        rewrite: change == "callback",
        ..Default::default()
    };
    for call in 0..4 {
        secrets
            .revision
            .store(usize::from(call >= 2), Ordering::SeqCst);
        let cache = ScopedCache::new(cache.clone(), CacheScope::Shared);
        match surface {
            "chat" => {
                ChatCompletionsRoute::new(
                    litellm_http::Client::plain_for_test(),
                    Arc::new(Default::default()),
                    secrets.clone(),
                )
                .with_cache(cache)
                .execute(
                    ChatCompletionsRequest {
                        model: "anthropic/cache-test-model",
                        messages: json!([{"role":"user","content":"hello"}]),
                        optional_params: [("max_tokens".into(), json!(32))].into_iter().collect(),
                        api_key: None,
                        api_base: None,
                        custom_llm_provider: None,
                        extra_headers: None,
                        timeout: None,
                    },
                    &hooks,
                    None,
                )
                .await
                .unwrap();
            }
            "messages" => {
                support::messages_route(secrets.clone()).with_cache(cache).execute(MessagesCall {
                    body: serde_json::from_value(json!({"model":"anthropic/cache-test-model","messages":[{"role":"user","content":"hello"}],"max_tokens":32})).unwrap(),
                    api_key:None,api_base:None,custom_llm_provider:None,extra_headers:None,provider_specific_header:None,timeout:None,shaping:Default::default(),
                }, &hooks, None).await.unwrap();
            }
            "responses" => {
                support::responses_route(secrets.clone())
                    .with_cache(cache)
                    .execute(
                        ResponsesCall {
                            model: "test".into(),
                            input: json!("hello"),
                            optional_params: Default::default(),
                            api_key: None,
                            api_base: None,
                            custom_llm_provider: None,
                            extra_headers: None,
                            timeout: None,
                        },
                        &hooks,
                        None,
                    )
                    .await
                    .unwrap();
            }
            _ => unreachable!(),
        }
    }
    assert_eq!(hooks.calls.load(Ordering::SeqCst), 4);
    {
        let facts = hooks.facts.lock().unwrap();
        assert_eq!(facts[0].source, ResultSource::Provider);
        assert_eq!(facts[2].source, ResultSource::Provider);
        let (ResultSource::Cache { key: first_key }, ResultSource::Cache { key: second_key }) =
            (&facts[1].source, &facts[3].source)
        else {
            panic!("unchanged effective requests must hit the cache");
        };
        assert_ne!(first_key, second_key);
    }
    let requests = first.received_requests().await.unwrap();
    if change == "credentials" {
        let header = if surface == "responses" {
            "authorization"
        } else {
            "x-api-key"
        };
        assert_ne!(requests[0].headers[header], requests[1].headers[header]);
    }
    if change == "callback" {
        assert_eq!(
            serde_json::from_slice::<Value>(&requests[0].body).unwrap()["temperature"],
            0.1
        );
        assert_eq!(
            serde_json::from_slice::<Value>(&requests[1].body).unwrap()["temperature"],
            0.8
        );
    }
    first.verify().await;
    second.verify().await;
}

#[rstest]
#[tokio::test]
async fn signed_requests_bypass_response_caching(cache: Arc<dyn ResponseCacheService>) {
    use litellm_cache_response::ScopedCache;
    use litellm_core::chat_completions::types::ChatCompletionsRequest;
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "output":{"message":{"role":"assistant","content":[{"text":"answer"}]}},
            "stopReason":"end_turn", "usage":{"inputTokens":3,"outputTokens":2,"totalTokens":5}
        })))
        .expect(2)
        .mount(&upstream)
        .await;
    let route =
        support::chat_completions_route().with_cache(ScopedCache::new(cache, CacheScope::Shared));
    let hooks = ChangingHooks::default();
    for _ in 0..2 {
        let response = route.execute(ChatCompletionsRequest {
            model:"bedrock/anthropic.cache-test-model",
            messages:json!([{"role":"user","content":"hello"}]),
            optional_params:json!({"aws_access_key_id":"test-access","aws_secret_access_key":"test-secret","aws_region_name":"eu-west-1"}).as_object().unwrap().clone(),
            api_key:None,api_base:Some(&upstream.uri()),custom_llm_provider:None,extra_headers:None,timeout:None,
        }, &hooks, None).await.unwrap();
        assert_eq!(
            serde_json::to_value(response).unwrap()["usage"]["total_tokens"],
            5
        );
    }
    assert!(
        hooks
            .facts
            .lock()
            .unwrap()
            .iter()
            .all(|facts| facts.source == ResultSource::Provider)
    );
    upstream.verify().await;
}
