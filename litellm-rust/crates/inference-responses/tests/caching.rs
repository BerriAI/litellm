mod support;
use litellm_host::protocol::InterceptRequest;

use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use litellm_cache_memory::InMemoryCache;
use litellm_cache_response::{
    CacheOptions, CacheScope, ResponseCache, ResponseCacheConfig, ResponseCacheService,
    ResponseEnvelope, ScopedCache,
};
use litellm_host::interceptors::{
    ExecutionFacts, Interceptors, ProviderIdentity, RawResponse, RequestContext, ResultSource,
    WireRequest,
};
use litellm_inference::{
    RouteError,
    caching::{CacheRequest, execute_unary},
};
use rstest::{fixture, rstest};
use serde_json::{Value, json};

fn cache_request<P: litellm_inference::caching::Cachable>(
    input: Value,
    cache: Option<Arc<dyn ResponseCacheService>>,
    options: Option<CacheOptions>,
) -> CacheRequest {
    CacheRequest::prepare::<P>(
        ProviderIdentity {
            model: "test-model".into(),
            provider: "test-provider".into(),
        },
        cache,
        options,
        &|_| input.clone(),
    )
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
    fn prepare(
        self: Arc<Self>,
        options: CacheOptions,
        surface: &str,
        input: &dyn Fn() -> Value,
    ) -> litellm_cache_response::PreparedCache {
        litellm_cache_response::PreparedCache {
            policy: options.policy,
            max_entry_bytes: self.0.config().max_entry_bytes,
            call: Arc::new(InvalidEntryCall {
                request: options.request(&self.0.config().namespace, surface, input()),
                service: self,
            }),
        }
    }
}

struct InvalidEntryCall {
    service: Arc<InvalidEntryCache>,
    request: litellm_cache_response::ResponseCacheRequest,
}

impl litellm_cache_response::ResponseCacheCall for InvalidEntryCall {
    fn lookup<'a>(
        &'a self,
        now: Duration,
    ) -> futures_util::future::BoxFuture<
        'a,
        Result<Option<litellm_cache_response::CacheHit>, litellm_cache::Error>,
    > {
        Box::pin(async move {
            Ok(Some(litellm_cache_response::CacheHit {
                value: self
                    .service
                    .0
                    .async_lookup(&self.request, now)
                    .await?
                    .unwrap_or_else(|| self.service.1.clone()),
                key: litellm_cache_response::cache_key(&self.request.key),
            }))
        })
    }

    fn store<'a>(
        &'a self,
        response: Value,
        now: Duration,
    ) -> futures_util::future::BoxFuture<'a, Result<(), litellm_cache::Error>> {
        Box::pin(self.service.0.async_store(&self.request, response, now))
    }
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
    async fn handle(&self, request: InterceptRequest) -> Result<(), RouteError> {
        match request {
            InterceptRequest::BeforeProviderRequest {
                wire,
                context,
                reply,
            } => reply.send(self.before_provider_request(*wire, *context).await?),
            InterceptRequest::AfterProviderResponse { raw, reply } => {
                reply.send(self.after_provider_response(raw).await?);
            }
            InterceptRequest::ResultReady { facts, reply } => {
                reply.send(self.result_ready(facts).await?);
            }
            other => other.pass_through(),
        }
        Ok(())
    }
}

impl ChangingHooks {
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
            cache_request::<Responses>(
                json!({"input":"hello"}),
                Some(cache.clone()),
                Some(CacheOptions::new(CacheScope::Shared)),
            ),
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
            cache_request::<Responses>(
                json!({"input":"hello"}),
                Some(cache.clone()),
                Some(CacheOptions::new(CacheScope::Shared)),
            ),
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

#[rstest]
#[case::credentials("credentials")]
#[case::endpoint("endpoint")]
#[case::callback("callback")]
#[tokio::test]
async fn responses_cache_identity_follows_resolved_configuration_and_request_callbacks(
    cache: Arc<dyn ResponseCacheService>,
    #[case] change: &str,
) {
    use litellm_inference_responses::types::ResponsesCall;
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

    let first = MockServer::start().await;
    let second = MockServer::start().await;
    let response = json!({"id":"response-test", "model":"test", "output":[], "status":"completed"});
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
        assert_ne!(
            requests[0].headers["authorization"],
            requests[1].headers["authorization"]
        );
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
