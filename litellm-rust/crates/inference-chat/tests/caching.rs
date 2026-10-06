mod support;
use std::{
    num::NonZeroUsize,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use litellm_cache_memory::InMemoryCache;
use litellm_cache_response::{CacheOptions, ResponseCache, ResponseCacheService};
use litellm_host::{
    interceptors::{
        ExecutionFacts, Interceptors, ProviderIdentity, RawResponse, RequestContext, ResultSource,
        WireRequest,
    },
    lifecycle::{CallEvent, ExecutionEvent},
    observation::observation_channel,
};
use litellm_inference::RouteError;
use rstest::{fixture, rstest};
use serde_json::{Value, json};
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
        route.with_cache(cache)
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
                litellm_inference::CallOptions {
                    cache: caching.then(CacheOptions::default),
                    observers: Some(observer.clone()),
                    ..Default::default()
                },
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
    use litellm_inference_chat::{ChatCompletionsRoute, types::ChatCompletionsRequest};
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

    let first = MockServer::start().await;
    let second = MockServer::start().await;
    let response = json!({"id":"message-test", "type":"message", "role":"assistant", "model":"test",
        "content":[{"type":"text", "text":"answer"}], "stop_reason":"end_turn", "stop_sequence":null,
        "usage":{"input_tokens":3,"output_tokens":2}});
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

        ChatCompletionsRoute::new(
            litellm_http::Client::plain_for_test(),
            Arc::new(Default::default()),
            secrets.clone(),
        )
        .with_cache(cache)
        .execute(
            ChatCompletionsRequest {
                model: &format!("anthropic/{model}"),
                messages: json!([{"role":"user","content":"hello"}]),
                optional_params: [("max_tokens".into(), json!(32))].into_iter().collect(),
                api_key: None,
                api_base: None,
                custom_llm_provider: None,
                extra_headers: None,
                timeout: None,
            },
            &hooks,
            CacheOptions::default(),
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

#[rstest]
#[tokio::test]
async fn signed_requests_bypass_response_caching(cache: Arc<dyn ResponseCacheService>) {
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
    let route = support::chat_completions_route().with_cache(cache);
    let hooks = ChangingHooks::default();
    for _ in 0..2 {
        let response = route.execute(ChatCompletionsRequest {
        model:"bedrock/anthropic.cache-test-model",
        messages:json!([{"role":"user","content":"hello"}]),
        optional_params:json!({"aws_access_key_id":"test-access","aws_secret_access_key":"test-secret","aws_region_name":"eu-west-1"}).as_object().unwrap().clone(),
        api_key:None,api_base:Some(&upstream.uri()),custom_llm_provider:None,extra_headers:None,timeout:None,
    }, &hooks, CacheOptions::default()).await.unwrap();
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
