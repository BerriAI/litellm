use litellm_cache_memory::InMemoryCache;
use litellm_cache_response::{
    CacheKey, CacheKeyInput, CacheOptions, CachePolicy, CacheScope, CacheTarget, ResponseCache,
    ResponseCacheConfig, ResponseCacheService, ResponseEnvelope,
};
use litellm_host::interceptors::{Interceptors, ProviderIdentity};
use litellm_inference::{
    RouteError,
    caching::{Cachable, CachePlan},
};
use rstest::{fixture, rstest};
use serde_json::{Value, json};
use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
struct CacheRequest {
    identity: ProviderIdentity,
    target: CacheTarget,
    parameters: Value,
}

fn cache_request(input: Value) -> CacheRequest {
    CacheRequest {
        identity: ProviderIdentity {
            model: "test-model".into(),
            provider: "test-provider".into(),
        },
        target: CacheTarget::resolve(None, "test-model", None, None),
        parameters: input,
    }
}

fn shared() -> CacheOptions {
    CacheOptions::shared(CachePolicy::default())
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
            CacheKeyInput::new(P::SURFACE, request.target, request.parameters),
        )
    });
    (request.identity, plan)
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
        .with_config(ResponseCacheConfig {
            namespace: "test".into(),
            max_entry_bytes,
        }),
    )
}

struct InvalidEntryCache(
    ResponseCache<InMemoryCache<litellm_cache_response::CacheEntry>>,
    Value,
);

impl ResponseCacheService for InvalidEntryCache {
    fn config(&self) -> &ResponseCacheConfig {
        self.0.config()
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

#[rstest]
#[case::chat_completion(json!({"kind":"Response","value":{"id":"chat-1","model":"test","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}}))]
#[case::wrong_envelope(json!({"kind":"Stream","value":"data: [DONE]\n\n"}))]
#[tokio::test]
async fn responses_refetches_instead_of_deserializing_another_api_response(
    #[case] poisoned: Value,
) {
    use litellm_inference_responses::route::Responses;
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
            Some(shared()),
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
#[case::completed("completed", 1)]
#[case::incomplete("incomplete", 2)]
#[tokio::test]
async fn responses_cache_only_reuses_completed_responses(
    cache: Arc<dyn ResponseCacheService>,
    #[case] status: &str,
    #[case] expected_calls: usize,
) {
    use litellm_inference_responses::route::Responses;
    use litellm_llms_types::formats::responses::ResponsesApiResponse;

    let calls = AtomicUsize::new(0);
    for _ in 0..2 {
        let response = execute_unary::<Responses, _, _>(
            cache_request(json!({"input":"hello"})),
            Some(cache.clone()),
            Some(shared()),
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

use litellm_host::interceptors::{
    ExecutionFacts, RawResponse, RequestContext, ResultSource, WireRequest,
};
mod support;
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
    reshape_headers: bool,
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
        if self.reshape_headers {
            wire.headers = wire
                .headers
                .into_iter()
                .rev()
                .map(|(name, value)| (name.to_ascii_uppercase(), value))
                .collect();
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
#[case::credentials("credentials")]
#[case::endpoint("endpoint")]
#[case::callback("callback")]
#[case::headers("headers")]
#[tokio::test]
async fn cache_identity_ignores_deployment_settings_and_skips_rewritten_requests(
    cache: Arc<dyn ResponseCacheService>,
    #[case] change: &str,
) {
    use litellm_inference_responses::types::ResponsesCall;
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

    let first = MockServer::start().await;
    let second = MockServer::start().await;
    let response = json!({"id":"response-test", "model":"test", "output":[], "status":"completed"});
    let rewritten = change == "callback";
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response.clone()))
        .expect(if rewritten { 4 } else { 1 })
        .mount(&first)
        .await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response))
        .expect(0)
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
        reshape_headers: change == "headers",
        ..Default::default()
    };
    for call in 0..4 {
        secrets
            .revision
            .store(usize::from(call >= 2), Ordering::SeqCst);
        let cache = cache.clone();
        let model = "cache-test-model";

        support::responses_route(secrets.clone())
            .with_cache(cache)
            .execute(
                ResponsesCall {
                    model: model.into(),
                    input: json!("hello"),
                    optional_params: Default::default(),
                    api_key: None,
                    api_base: None,
                    custom_llm_provider: None,
                    extra_headers: None,
                    timeout: None,
                },
                &hooks,
                shared(),
            )
            .await
            .unwrap();
    }
    assert_eq!(hooks.calls.load(Ordering::SeqCst), 4);
    let sources = hooks
        .facts
        .lock()
        .unwrap()
        .iter()
        .map(|facts| facts.source.clone())
        .collect::<Vec<_>>();
    match sources.as_slice() {
        [
            ResultSource::Provider,
            ResultSource::Provider,
            ResultSource::Provider,
            ResultSource::Provider,
        ] if rewritten => {}
        [
            ResultSource::Provider,
            ResultSource::Cache { key: first },
            ResultSource::Cache { key: second },
            ResultSource::Cache { key: third },
        ] if !rewritten && first == second && second == third => {}
        _ => panic!("unexpected result sources for {change}: {sources:?}"),
    }
    if rewritten {
        let temperatures = first
            .received_requests()
            .await
            .unwrap()
            .iter()
            .map(|request| {
                serde_json::from_slice::<Value>(&request.body).unwrap()["temperature"].clone()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            temperatures,
            [json!(0.1), json!(0.1), json!(0.8), json!(0.8)]
        );
    }
    first.verify().await;
    second.verify().await;
}

use bytes::Bytes;
use futures_util::TryStreamExt;
use litellm_host::call::CallOutput;
use litellm_inference::caching::StreamCachable;
use litellm_inference_responses::route::Responses;

#[rstest]
#[tokio::test]
async fn replay_yields_one_chunk_per_sse_event() {
    let data = Bytes::from_static(
        b"data: {\"type\":\"content_block_delta\",\"text\":\"hello\"}\n\ndata: {\"type\":\"message_stop\"}\n\n",
    );
    let expected = vec![
        Bytes::from_static(b"data: {\"type\":\"content_block_delta\",\"text\":\"hello\"}\n\n"),
        Bytes::from_static(b"data: {\"type\":\"message_stop\"}\n\n"),
    ];

    let CallOutput::Stream { chunks, .. } = <Responses as StreamCachable>::replay(data).unwrap()
    else {
        panic!("expected a stream");
    };
    assert_eq!(chunks.try_collect::<Vec<_>>().await.unwrap(), expected);
}
