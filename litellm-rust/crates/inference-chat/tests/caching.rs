mod support;

use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use litellm_cache_memory::InMemoryCache;
use litellm_cache_response::{
    CacheScope, ResponseCache, ResponseCacheConfig, ResponseCacheService,
};
use litellm_host::interceptors::{
    ExecutionFacts, Interceptors, ProviderIdentity, RawResponse, RequestContext, ResultSource,
    WireRequest,
};
use litellm_inference::RouteError;
use rstest::{fixture, rstest};
use serde_json::{Value, json};

use support::traces;

#[fixture]
fn cache() -> Arc<dyn ResponseCacheService> {
    Arc::new(
        ResponseCache::new(Arc::new(InMemoryCache::new(
            Some(100),
            Some(Duration::from_secs(60)),
        )))
        .with_config(ResponseCacheConfig {
            namespace: "test".into(),
            max_entry_bytes: 4096,
        }),
    )
}

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
    use litellm_inference_chat::types::ChatCompletionsRequest;
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
    let hooks = ChangingHooks::default();
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
                &hooks,
                None,
            ))
            .await
            .unwrap();
        assert_eq!(
            serde_json::to_value(response).unwrap()["usage"]["total_tokens"],
            15
        );
    }
    let facts = hooks.facts.lock().unwrap().clone();
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
#[case::credentials("credentials")]
#[case::endpoint("endpoint")]
#[case::callback("callback")]
#[tokio::test]
async fn chat_cache_identity_follows_resolved_configuration_and_request_callbacks(
    cache: Arc<dyn ResponseCacheService>,
    #[case] change: &str,
) {
    use litellm_cache_response::ScopedCache;
    use litellm_inference_chat::{ChatCompletionsRoute, types::ChatCompletionsRequest};
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

    let first = MockServer::start().await;
    let second = MockServer::start().await;
    let response = json!({"id":"message-test", "type":"message", "role":"assistant", "model":"test",
        "content":[{"type":"text", "text":"answer"}], "stop_reason":"end_turn", "stop_sequence":null,
        "usage":{"input_tokens":3,"output_tokens":2}});
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
            requests[0].headers["x-api-key"],
            requests[1].headers["x-api-key"]
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

#[rstest]
#[tokio::test]
async fn signed_requests_bypass_response_caching(cache: Arc<dyn ResponseCacheService>) {
    use litellm_cache_response::ScopedCache;
    use litellm_inference_chat::types::ChatCompletionsRequest;
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
