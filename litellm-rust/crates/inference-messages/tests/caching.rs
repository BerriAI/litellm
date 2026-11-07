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
    CacheOptions, CacheScope, ResponseCache, ResponseCacheConfig, ResponseCacheService, ScopedCache,
};
use litellm_host::interceptors::{
    Cost, ExecutionFacts, Interceptors, ProviderIdentity, RawResponse, RequestContext,
    ResultSource, WireRequest,
};
use litellm_inference::{
    Error,
    caching::{CacheRequest, execute_unary},
};
use rstest::{fixture, rstest};
use serde_json::{Value, json};

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
        Result<Option<litellm_secret_value::SecretValue>, litellm_secrets::Error>,
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
            Ok(value.map(litellm_secret_value::SecretValue::new))
        })
    }
}

#[derive(Default)]
struct ChangingHooks {
    calls: AtomicUsize,
    rewrite: bool,
    facts: std::sync::Mutex<Vec<ExecutionFacts>>,
}

impl Interceptors<Error> for ChangingHooks {
    async fn before_provider_request(
        &self,
        mut wire: WireRequest,
        _: RequestContext,
    ) -> Result<WireRequest, Error> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        if self.rewrite {
            wire.body["temperature"] = json!(if call < 2 { 0.1 } else { 0.8 });
        }
        Ok(wire)
    }

    async fn after_provider_response(&self, _: RawResponse) -> Result<(), Error> {
        Ok(())
    }

    async fn result_ready(&self, facts: ExecutionFacts) -> Result<(), Error> {
        self.facts.lock().unwrap().push(facts);
        Ok(())
    }
}

#[rstest]
#[case::credentials("credentials")]
#[case::endpoint("endpoint")]
#[case::callback("callback")]
#[tokio::test]
async fn messages_cache_identity_follows_resolved_configuration_and_request_callbacks(
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
        support::messages_route(secrets.clone()).with_cache(cache).execute(MessagesCall {
            body: serde_json::from_value(json!({"model":"anthropic/cache-test-model","messages":[{"role":"user","content":"hello"}],"max_tokens":32})).unwrap(),
            api_key:None,api_base:None,custom_llm_provider:None,litellm_params:Default::default(),extra_headers:None,provider_specific_header:None,timeout:None,shaping:Default::default(),
        }, &hooks, None).await.unwrap();
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
#[case::reported(json!(0.37), Some(0.37))]
#[case::free(json!(0), Some(0.0))]
#[case::malformed(json!("invalid"), None)]
#[case::missing(Value::Null, None)]
#[tokio::test]
async fn provider_reported_cost_is_delivered_once_and_never_charged_for_a_cache_hit(
    cache: Arc<dyn ResponseCacheService>,
    #[case] cost: Value,
    #[case] expected: Option<f64>,
) {
    use litellm_inference_messages::{MessagesCall, MessagesCallResponse};
    let response = json!({"id": "msg_cost", "type": "message", "role": "assistant", "model": "native-test",
        "content": [{"type": "text", "text": "answer"}], "stop_reason": "end_turn", "stop_sequence": null,
        "usage": {"input_tokens": 3, "output_tokens": 2}, "cost": cost});
    let upstream = support::upstream([support::json_response(response)]).await;
    let hooks = ChangingHooks::default();
    let route = support::messages_route(litellm_inference_testing::no_secrets())
        .with_cache(ScopedCache::new(cache, CacheScope::Shared));
    let request = || {
        MessagesCall {
        body: serde_json::from_value(json!({"model": "native-test", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16})).unwrap(),
        api_key: Some("eden-key".into()), api_base: Some(upstream.uri()), custom_llm_provider: Some("edenai".into()),
        litellm_params: Default::default(), extra_headers: None, provider_specific_header: None,
        timeout: None, shaping: Default::default(),
    }
    };
    let first = route.execute(request(), &hooks, None).await.unwrap();
    let second = route.execute(request(), &hooks, None).await.unwrap();
    let (MessagesCallResponse::Complete(first), MessagesCallResponse::Complete(second)) =
        (first, second)
    else {
        panic!("expected Messages responses");
    };
    assert_eq!(first, second);
    assert_eq!(first.extra.get("cost"), Some(&cost));
    assert_eq!(support::received(&upstream).await.len(), 1);
    let facts = hooks.facts.lock().unwrap();
    let [provider, cached] = facts.as_slice() else {
        panic!("expected provider and cache results");
    };
    assert_eq!(provider.source, ResultSource::Provider);
    assert_eq!(
        provider.cost,
        expected.map_or(Cost::Deferred, |amount| Cost::Reported {
            amount: serde_json::Number::from_f64(amount).unwrap()
        })
    );
    assert!(matches!(cached.source, ResultSource::Cache { .. }));
    assert_eq!(cached.cost, Cost::CacheHit);
}

#[fixture]
fn streaming_call() -> litellm_inference_messages::MessagesCall {
    litellm_inference_messages::MessagesCall {
        body: serde_json::from_value(json!({
            "model": "anthropic/cache-test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 32,
            "stream": true
        }))
        .unwrap(),
        api_key: Some("stream-key".into()),
        api_base: None,
        custom_llm_provider: None,
        litellm_params: Default::default(),
        extra_headers: None,
        provider_specific_header: None,
        timeout: Some(Duration::from_secs(5)),
        shaping: Default::default(),
    }
}

const COMPLETE_STREAM: &str = "event: message_start\ndata: {\"type\":\"message_start\"}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";

async fn streamed_bytes(output: litellm_inference_messages::MessagesCallResponse) -> Vec<u8> {
    use futures_util::TryStreamExt;
    let litellm_inference_messages::MessagesCallResponse::Stream { chunks, .. } = output else {
        panic!("expected a Messages stream");
    };
    chunks.try_collect::<Vec<_>>().await.unwrap().concat()
}

#[rstest]
#[case::complete(COMPLETE_STREAM.to_owned(), true)]
#[case::data_only("data: {\"type\":\"message_stop\"}\n\n".into(), true)]
#[case::crlf("event: message_stop\r\ndata: {\"type\":\"message_stop\"}\r\n\r\n".into(), true)]
#[case::empty(String::new(), false)]
#[case::missing_stop("data: {\"type\":\"message_start\"}\n\n".into(), false)]
#[case::unterminated_stop("data: {\"type\":\"message_stop\"}".into(), false)]
#[case::partial_tail(format!("{COMPLETE_STREAM}data: {{"), false)]
#[case::error_before_stop(format!("data: {{\"type\":\"error\",\"error\":{{\"message\":\"failed\"}}}}\n\n{COMPLETE_STREAM}"), false)]
#[case::error_after_stop(format!("{COMPLETE_STREAM}data: {{\"type\":\"error\"}}\n\n"), false)]
#[case::invalid_json(format!("data: invalid\n\n{COMPLETE_STREAM}"), false)]
#[case::event_name_without_type("event: message_stop\ndata: {}\n\n".into(), false)]
#[case::over_entry_limit(format!("data: {{\"type\":\"content_block_delta\",\"text\":\"{}\"}}\n\n{COMPLETE_STREAM}", "x".repeat(4096)), false)]
#[tokio::test]
async fn stream_cache_eligibility_preserves_every_delivered_byte(
    cache: Arc<dyn ResponseCacheService>,
    streaming_call: litellm_inference_messages::MessagesCall,
    #[case] wire: String,
    #[case] cache_hit: bool,
) {
    use wiremock::ResponseTemplate;
    let response = ResponseTemplate::new(200).set_body_raw(wire.clone(), "text/event-stream");
    let upstream = support::upstream([response.clone(), response]).await;
    let route = support::messages_route(litellm_inference_testing::no_secrets())
        .with_cache(ScopedCache::new(cache, CacheScope::Shared));
    let request = || litellm_inference_messages::MessagesCall {
        body: streaming_call.body.clone(),
        api_base: Some(upstream.uri()),
        ..self::streaming_call()
    };
    let hooks = ChangingHooks::default();

    let first = streamed_bytes(route.execute(request(), &hooks, None).await.unwrap()).await;
    let second = streamed_bytes(route.execute(request(), &hooks, None).await.unwrap()).await;

    assert_eq!(first, wire.as_bytes());
    assert_eq!(second, wire.as_bytes());
    assert_eq!(
        support::received(&upstream).await.len(),
        if cache_hit { 1 } else { 2 }
    );
    let facts = hooks.facts.lock().unwrap();
    let [first, second] = facts.as_slice() else {
        panic!("each call reports exactly one result");
    };
    assert_eq!(first.source, ResultSource::Provider);
    assert_eq!(
        matches!(second.source, ResultSource::Cache { .. }),
        cache_hit
    );
    assert_eq!(
        second.cost,
        if cache_hit {
            Cost::CacheHit
        } else {
            Cost::Deferred
        }
    );
}

#[rstest]
#[case::before_first_chunk(false)]
#[case::after_terminal_chunk(true)]
#[tokio::test]
async fn dropping_a_stream_before_eof_cancels_once_and_does_not_cache(
    cache: Arc<dyn ResponseCacheService>,
    streaming_call: litellm_inference_messages::MessagesCall,
    #[case] read_chunk: bool,
) {
    use futures_util::StreamExt;
    use litellm_host::lifecycle::{CallEvent, ExecutionEvent};
    use litellm_inference_messages::MessagesCallResponse;
    use wiremock::ResponseTemplate;

    let response = ResponseTemplate::new(200).set_body_raw(COMPLETE_STREAM, "text/event-stream");
    let upstream = support::upstream([response.clone(), response]).await;
    let route = support::messages_route(litellm_inference_testing::no_secrets())
        .with_cache(ScopedCache::new(cache, CacheScope::Shared));
    let request = || litellm_inference_messages::MessagesCall {
        body: streaming_call.body.clone(),
        api_base: Some(upstream.uri()),
        ..self::streaming_call()
    };
    let hooks = ChangingHooks::default();
    let observations = support::Observations::default();
    let first = route
        .execute(request(), &hooks, Some(observations.sender.clone()))
        .await
        .unwrap();
    let MessagesCallResponse::Stream { mut chunks, .. } = first else {
        panic!("expected a Messages stream");
    };
    if read_chunk {
        let mut delivered = Vec::new();
        while delivered.len() < COMPLETE_STREAM.len() {
            delivered.extend_from_slice(&chunks.next().await.unwrap().unwrap());
        }
        assert_eq!(delivered, COMPLETE_STREAM.as_bytes());
    }
    assert!(matches!(
        observations.lock().unwrap().as_slice(),
        [
            CallEvent::Started { .. },
            CallEvent::Execution(ExecutionEvent::ResultReady { .. })
        ]
    ));
    drop(chunks);
    assert!(matches!(
        observations.lock().unwrap().as_slice(),
        [
            CallEvent::Started { .. },
            CallEvent::Execution(ExecutionEvent::ResultReady { .. }),
            CallEvent::Cancelled { .. }
        ]
    ));

    let second = streamed_bytes(route.execute(request(), &hooks, None).await.unwrap()).await;
    let third = streamed_bytes(route.execute(request(), &hooks, None).await.unwrap()).await;

    assert_eq!(second, COMPLETE_STREAM.as_bytes());
    assert_eq!(third, COMPLETE_STREAM.as_bytes());
    assert_eq!(support::received(&upstream).await.len(), 2);
    let facts = hooks.facts.lock().unwrap();
    let [cancelled, completed, cached] = facts.as_slice() else {
        panic!("each call reports exactly one result");
    };
    assert_eq!(cancelled.source, ResultSource::Provider);
    assert_eq!(completed.source, ResultSource::Provider);
    assert!(matches!(cached.source, ResultSource::Cache { .. }));
    assert_eq!(cached.cost, Cost::CacheHit);
}

#[rstest]
#[case::platform("bedrock/claude_platform/claude-test")]
#[case::mantle("bedrock_mantle/claude-test")]
#[case::mantle_alias("bedrock/mantle/claude-test")]
#[tokio::test]
async fn signed_messages_bypass_the_response_cache(
    cache: Arc<dyn ResponseCacheService>,
    #[case] model: &str,
) {
    use litellm_inference_testing::RecordingSecrets;
    use support::ReceivedRequest;

    let response = support::json_response(json!({
        "id": "signed-message", "type": "message", "role": "assistant", "model": "claude-test",
        "content": [{"type": "text", "text": "signed reply"}], "stop_reason": "end_turn"
    }));
    let upstream = support::upstream([response.clone(), response]).await;
    let secrets = Arc::new(RecordingSecrets::new([
        ("AWS_ACCESS_KEY_ID", "test-access"),
        ("AWS_SECRET_ACCESS_KEY", "test-secret"),
        ("AWS_SESSION_TOKEN", "test-session"),
        ("AWS_REGION_NAME", "us-west-2"),
        ("ANTHROPIC_AWS_WORKSPACE_ID", "test-workspace"),
    ]));
    let route =
        support::messages_route(secrets).with_cache(ScopedCache::new(cache, CacheScope::Shared));
    let request = || litellm_inference_messages::MessagesCall {
        body: serde_json::from_value(json!({
            "model": model, "messages": [{"role": "user", "content": "hello"}], "max_tokens": 32
        }))
        .unwrap(),
        api_key: None,
        api_base: Some(upstream.uri()),
        ..streaming_call()
    };
    let hooks = ChangingHooks::default();

    let first = route.execute(request(), &hooks, None).await.unwrap();
    let second = route.execute(request(), &hooks, None).await.unwrap();

    let (
        litellm_inference_messages::MessagesCallResponse::Complete(first),
        litellm_inference_messages::MessagesCallResponse::Complete(second),
    ) = (first, second)
    else {
        panic!("expected completed signed messages");
    };
    assert_eq!(first, second);
    let requests = support::received(&upstream).await;
    let [first, second] = requests.as_slice() else {
        panic!("signed requests must each reach the provider");
    };
    assert!(
        first
            .header("authorization")
            .unwrap()
            .starts_with("AWS4-HMAC-SHA256 Credential=test-access/")
    );
    assert!(
        second
            .header("authorization")
            .unwrap()
            .starts_with("AWS4-HMAC-SHA256 Credential=test-access/")
    );
    let facts = hooks.facts.lock().unwrap();
    let [first, second] = facts.as_slice() else {
        panic!("each signed request reports one result");
    };
    assert_eq!(first.source, ResultSource::Provider);
    assert_eq!(second.source, ResultSource::Provider);
}

#[rstest]
#[tokio::test]
async fn oversized_response_integers_are_rejected_before_cache_storage(
    cache: Arc<dyn ResponseCacheService>,
) {
    use litellm_inference_messages::{MessagesCall, MessagesCallResponse};
    use wiremock::ResponseTemplate;

    let invalid = r#"{"id":"bad","type":"message","role":"assistant","model":"cache-test","content":[],"future_extension":{"wide_integer":18446744073709551617}}"#;
    let valid = json!({"id":"good","type":"message","role":"assistant","model":"cache-test","content":[],"future_extension":{"wide_integer":u64::MAX}});
    let upstream = support::upstream([
        ResponseTemplate::new(200).set_body_raw(invalid, "application/json"),
        support::json_response(valid.clone()),
    ])
    .await;
    let route = support::messages_route(litellm_inference_testing::no_secrets())
        .with_cache(ScopedCache::new(cache, CacheScope::Shared));
    let request = || {
        MessagesCall {
        body: serde_json::from_value(json!({"model":"anthropic/cache-test","messages":[{"role":"user","content":"hi"}],"max_tokens":16})).unwrap(),
        api_key: Some("synthetic".into()), api_base: Some(upstream.uri()), custom_llm_provider: None,
        litellm_params: Default::default(), extra_headers: None, provider_specific_header: None,
        timeout: None, shaping: Default::default(),
    }
    };
    let rejected = route
        .execute(request(), &(), None)
        .await
        .err()
        .expect("oversized provider integer must fail");
    assert!(rejected.is_response());
    assert!(!rejected.is_request());
    let complete = route.execute(request(), &(), None).await.unwrap();
    let cached = route.execute(request(), &(), None).await.unwrap();
    let (MessagesCallResponse::Complete(complete), MessagesCallResponse::Complete(cached)) =
        (complete, cached)
    else {
        panic!("expected complete messages");
    };
    assert_eq!(
        complete.extra["future_extension"],
        valid["future_extension"]
    );
    assert_eq!(cached, complete);
    assert_eq!(support::received(&upstream).await.len(), 2);
}
