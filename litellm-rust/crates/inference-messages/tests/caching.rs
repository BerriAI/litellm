use litellm_cache_memory::InMemoryCache;
use litellm_cache_response::{
    CacheKeyInput, CacheOptions, CachePolicy, CacheScope, CacheTarget, ResponseCache,
    ResponseCacheConfig, ResponseCacheService,
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
            ..ResponseCacheConfig::default()
        }),
    )
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
    use litellm_inference_messages::route::Messages;
    use litellm_llms_types::formats::messages::MessagesResponse;

    let calls = AtomicUsize::new(0);
    for (value, expected_call) in [(original.clone(), 0), (changed, 1), (original, 0)] {
        let response =
            execute_unary::<Messages, _, _>(
                CacheRequest {
                    identity: ProviderIdentity {
                        model: "test".into(),
                        provider: "anthropic".into(),
                    },
                    target: CacheTarget::resolve(None, "test", None, None),
                    parameters: json!({
                        "messages":[{"role":"user","content":"hello"}],
                        "max_tokens":32, (field):value
                    }),
                },
                Some(cache.clone()),
                Some(shared()),
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
    use litellm_inference_messages::MessagesCall;
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

        support::messages_route(secrets.clone()).with_cache(cache).execute(MessagesCall {
                body: serde_json::from_value(json!({"model":format!("anthropic/{model}"),"messages":[{"role":"user","content":"hello"}],"max_tokens":32})).unwrap(),
                api_key:None,api_base:None,custom_llm_provider:None,extra_headers:None,provider_specific_header:None,timeout:None,shaping:Default::default(),
            }, &hooks, shared()).await.unwrap();
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
#[case::same_request("none", true)]
#[case::other_beta_header("provider_specific_header", false)]
#[case::other_extra_header("extra_headers", false)]
#[case::other_api_base_without_a_group("api_base", false)]
#[tokio::test]
async fn forwarded_headers_and_the_deployment_decide_messages_cache_reuse(
    cache: Arc<dyn ResponseCacheService>,
    #[case] change: &str,
    #[case] hit: bool,
) {
    use litellm_inference_messages::MessagesCall;
    use litellm_llms_types::headers::{ProviderSpecificHeader, ProviderSpecificHeaders};
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id":"message-test", "type":"message", "role":"assistant", "model":"test",
            "content":[{"type":"text", "text":"answer"}], "stop_reason":"end_turn",
            "stop_sequence":null, "usage":{"input_tokens":3,"output_tokens":2}
        })))
        .expect(if hit { 1 } else { 2 })
        .mount(&upstream)
        .await;
    let route = support::messages_route(support::no_secrets()).with_cache(cache);
    let hooks = ChangingHooks::default();
    let call = |change: &str| MessagesCall {
        body: serde_json::from_value(json!({
            "model":"claude-cache-test",
            "messages":[{"role":"user","content":"hello"}],
            "max_tokens":32
        }))
        .unwrap(),
        api_key: Some("sk-test".into()),
        api_base: Some(match change {
            "api_base" => format!("{}/", upstream.uri()),
            _ => upstream.uri(),
        }),
        custom_llm_provider: Some("anthropic".into()),
        extra_headers: (change == "extra_headers").then(|| {
            json!({"anthropic-version":"2023-06-01"})
                .as_object()
                .unwrap()
                .clone()
        }),
        provider_specific_header: (change == "provider_specific_header").then(|| {
            ProviderSpecificHeaders::One(ProviderSpecificHeader {
                custom_llm_provider: "anthropic".into(),
                extra_headers: json!({"anthropic-beta":"context-1m-2025-08-07"})
                    .as_object()
                    .unwrap()
                    .clone(),
            })
        }),
        timeout: None,
        shaping: Default::default(),
    };
    for change in ["none", change] {
        route.execute(call(change), &hooks, shared()).await.unwrap();
    }
    assert_eq!(
        matches!(
            hooks.facts.lock().unwrap()[1].source,
            ResultSource::Cache { .. }
        ),
        hit
    );
    upstream.verify().await;
}

use bytes::Bytes;
use futures_util::TryStreamExt;
use litellm_host::call::CallOutput;
use litellm_inference::caching::StreamCachable;
use litellm_inference_messages::route::Messages;

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

    let CallOutput::Stream { chunks, .. } = <Messages as StreamCachable>::replay(data).unwrap()
    else {
        panic!("expected a stream");
    };
    assert_eq!(chunks.try_collect::<Vec<_>>().await.unwrap(), expected);
}
