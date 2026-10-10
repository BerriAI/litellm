use litellm_host::{
    interceptors::{ExecutionFacts, ResultSource},
    lifecycle::ExecutionEvent,
};
use litellm_inference_messages::{MessagesCallResponse, messages_body};
use litellm_inference_testing::{
    RecordingSecrets, http_config, no_secrets, provider_http, resources,
};
use rstest::rstest;

use super::*;

#[rstest]
#[case::neither(false, false)]
#[case::hooks_only(true, false)]
#[case::observer_only(false, true)]
#[case::both(true, true)]
#[tokio::test]
async fn calls_defer_execution_until_polled(
    call: MessagesCall,
    #[case] with_hooks: bool,
    #[case] with_observer: bool,
) {
    use futures_util::future::BoxFuture;

    use litellm_host::lifecycle::CallEvent;

    let upstream = upstream([message_response()]).await;
    let secrets = Arc::new(RecordingSecrets::new([("ANTHROPIC_API_KEY", "test-key")]));
    let route = messages_route(secrets.clone());
    let host = RecordingCall::<Messages>::new(MessagesCall {
        api_base: Some(upstream.uri()),
        ..call
    });
    let request = host.request().unwrap();
    let observer: Option<litellm_host::observation::ObservationSender> =
        with_observer.then(|| host.events.0.sender.clone());
    let future: BoxFuture<'_, Result<MessagesCallResponse, Error>> = if with_hooks {
        Box::pin(route.execute(request, &host, observer))
    } else {
        Box::pin(route.execute(request, &(), observer))
    };

    assert!(secrets.requested().is_empty());
    assert!(host.events.0.lock().unwrap().is_empty());
    assert!(received(&upstream).await.is_empty());

    let MessagesCallResponse::Complete(response) = future.await.unwrap() else {
        panic!("expected a completed message");
    };
    assert_eq!(
        response.body().content,
        message_body()["content"].as_array().unwrap().as_slice()
    );
    assert!(secrets.requested().contains(&"ANTHROPIC_API_KEY".into()));
    let sent = only_request(&upstream).await;
    assert_eq!(sent.header("x-api-key"), Some("test-key"));
    assert_eq!(sent.header("x-hook"), with_hooks.then_some("called"));
    let events = host.events.0.lock().unwrap();
    assert!(matches!(
        (with_hooks, with_observer, events.as_slice()),
        (false, false, [])
            | (true, false, [])
            | (
                false,
                true,
                [
                    CallEvent::Started { .. },
                    CallEvent::Execution(ExecutionEvent::ProviderResponseReceived { .. }),
                    CallEvent::Execution(ExecutionEvent::ResultReady {
                        facts: ExecutionFacts {
                            source: ResultSource::Provider,
                            ..
                        }
                    }),
                    CallEvent::Succeeded { .. }
                ]
            )
            | (
                true,
                true,
                [
                    CallEvent::Started { .. },
                    CallEvent::Execution(ExecutionEvent::ProviderResponseReceived { .. }),
                    CallEvent::Execution(ExecutionEvent::ResultReady {
                        facts: ExecutionFacts {
                            source: ResultSource::Provider,
                            ..
                        }
                    }),
                    CallEvent::Succeeded { .. }
                ]
            )
    ));
}

#[rstest]
#[case::anthropic("anthropic")]
#[case::azure_ai("azure_ai")]
#[tokio::test]
async fn the_provider_message_is_returned(call: MessagesCall, #[case] provider: &str) {
    let upstream = upstream([message_response()]).await;

    let message = run_message(MessagesCall {
        custom_llm_provider: Some(provider.into()),
        litellm_params: Default::default(),
        api_key: Some("sk".into()),
        api_base: Some(upstream.uri()),
        ..call
    })
    .await;

    assert_eq!(message.id, "msg_1");
    assert_eq!(message.content, [json!({"type": "text", "text": "hi"})]);
    assert_eq!(message.stop_reason.as_deref(), Some("end_turn"));
}

/// A refusal and fields the route does not model come back exactly as the provider sent
/// them, since the Python side returns the raw message and the router decides what to do.
#[rstest]
#[tokio::test]
async fn the_message_passes_through_losslessly(call: MessagesCall) {
    let upstream_body = json!({
        "id": "msg_2",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [
            {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "q"}},
            {"type": "text", "text": "no", "citations": [{"type": "web_search_result_location", "url": "https://e.x"}]}
        ],
        "stop_reason": "refusal",
        "stop_sequence": null,
        "stop_details": {"type": "safeguard", "safeguard_types": ["dangerous_tool_use"]},
        "container": {"id": "container_1", "expires_at": "2026-01-01T00:00:00Z"},
        "context_management": {"applied_edits": []},
        "usage": {"input_tokens": 1, "output_tokens": 2, "server_tool_use": {"web_search_requests": 1}},
        "unknown_future_field": {"nested": true}
    });
    let upstream = upstream([json_response(upstream_body.clone())]).await;

    let message = run_message(MessagesCall {
        api_key: Some("sk".into()),
        api_base: Some(upstream.uri()),
        ..call
    })
    .await;

    assert_eq!(message.stop_reason.as_deref(), Some("refusal"));
    assert_eq!(serde_json::to_value(&message).unwrap(), upstream_body);
}

#[rstest]
#[tokio::test]
async fn a_json_error_envelope_is_kept_verbatim(call: MessagesCall) {
    let envelope =
        json!({"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}});
    let upstream = upstream([status_response(400, envelope.clone())]).await;

    let error = run(MessagesCall {
        api_key: Some("sk".into()),
        api_base: Some(upstream.uri()),
        ..call
    })
    .await
    .expect_err("upstream error propagates");

    let Error::Rejected(rejected) = error else {
        panic!("{error:?}");
    };
    assert_eq!(rejected.status().as_u16(), 400);
    assert_eq!(
        serde_json::from_str::<Value>(&rejected.text()).unwrap(),
        envelope
    );
}

#[rstest]
#[tokio::test]
async fn a_long_error_body_is_truncated_at_the_documented_cap(call: MessagesCall) {
    let long = "x".repeat(600);
    let upstream = upstream([ResponseTemplate::new(500).set_body_string(long.clone())]).await;

    let error = run(MessagesCall {
        api_key: Some("sk".into()),
        api_base: Some(upstream.uri()),
        ..call
    })
    .await
    .expect_err("upstream error propagates");

    let Error::Rejected(rejected) = error else {
        panic!("unexpected error: {error:?}");
    };
    assert_eq!(rejected.status().as_u16(), 500);
    assert_eq!(rejected.text(), format!("{}... (truncated)", &long[..256]));
}

#[rstest]
#[case::bad_request(400)]
#[case::unauthorized(401)]
#[case::rate_limited(429)]
#[case::server_error(500)]
#[case::overloaded(529)]
#[tokio::test]
async fn an_upstream_error_keeps_its_status_body_and_headers(
    call: MessagesCall,
    #[case] status: u16,
) {
    let upstream = upstream([ResponseTemplate::new(status)
        .set_body_string("upstream said no")
        .append_header("Retry-After", "17")
        .append_header("X-Provider-Trace", "first")
        .append_header("X-Provider-Trace", "second")])
    .await;

    let error = run(MessagesCall {
        api_key: Some("sk".into()),
        api_base: Some(upstream.uri()),
        ..call
    })
    .await
    .expect_err("upstream error propagates");

    let Error::Rejected(rejected) = error else {
        panic!("unexpected error: {error:?}");
    };
    assert_eq!(rejected.status().as_u16(), status);
    assert_eq!(rejected.text(), "upstream said no");
    assert_eq!(rejected.headers()["retry-after"], "17");
    assert_eq!(
        rejected
            .headers()
            .get_all("x-provider-trace")
            .iter()
            .collect::<Vec<_>>(),
        ["first", "second"]
    );
}

#[rstest]
#[case::not_json(ResponseTemplate::new(200).set_body_string("not json"))]
#[case::not_a_message(json_response(json!({"unexpected": true})))]
#[tokio::test]
async fn an_unreadable_success_body_is_an_invalid_response(
    call: MessagesCall,
    #[case] response: ResponseTemplate,
) {
    let upstream = upstream([response]).await;

    let error = run(MessagesCall {
        api_key: Some("sk".into()),
        api_base: Some(upstream.uri()),
        ..call
    })
    .await
    .expect_err("an unreadable body fails");

    assert!(matches!(error, Error::InvalidResponse(_)), "{error:?}");
}

#[rstest]
#[tokio::test]
async fn a_provider_slower_than_the_timeout_fails_the_call(call: MessagesCall) {
    let upstream = upstream([message_response().set_delay(Duration::from_secs(5))]).await;

    let error = run(MessagesCall {
        api_key: Some("sk".into()),
        api_base: Some(upstream.uri()),
        timeout: Some(Duration::from_millis(100)),
        ..call
    })
    .await
    .expect_err("the call times out");

    assert!(matches!(error, Error::Transport(_)), "{error:?}");
}

#[rstest]
#[tokio::test]
async fn the_facade_sends_through_the_injected_http_pool_configuration(call: MessagesCall) {
    let upstream = upstream([message_response()]).await;
    let base = upstream.uri();
    let settings = HttpSettings {
        user_agent: Some("host-owned/1".into()),
        ..HttpSettings::default()
    };

    let resources = resources();
    let response = litellm_inference_messages::MessagesRoute::new(
        provider_http(&resources, &Resolution::from(&settings).config),
        resources.auth,
        no_secrets(),
    )
    .execute(
        MessagesCall {
            api_key: Some("sk-ant".into()),
            api_base: Some(base),
            ..call
        },
        &(),
        None,
    )
    .await
    .expect("messages request succeeds");

    let MessagesCallResponse::Complete(message) = response else {
        panic!("a non-streaming request returns a message");
    };
    assert_eq!(message.body().id, "msg_1");
    let sent = only_request(&upstream).await;
    assert_eq!(sent.header("x-api-key"), Some("sk-ant"));
    assert_eq!(sent.header("user-agent"), Some("host-owned/1"));
}

#[rstest]
#[case::mistyped_param(json!({"model": MODEL, "messages": [], "max_tokens": "16"}))]
#[case::missing_messages(json!({"model": MODEL, "max_tokens": 16}))]
fn a_body_that_does_not_parse_is_an_invalid_request(#[case] raw: Value) {
    let error = messages_body(object(raw)).expect_err("the body is rejected");

    assert!(
        matches!(&error, Error::InvalidRequest(message) if message.to_string().starts_with("invalid Anthropic messages request: ")),
        "{error:?}"
    );
}

#[rstest]
#[tokio::test]
async fn message_route_summary_excludes_payload_diagnostics(
    call: MessagesCall,
    traces: TraceCapture,
) {
    let upstream = upstream([message_response()]).await;
    let model = call.body.model.clone();
    traces
        .logger()
        .instrument(run_message(MessagesCall {
            api_key: Some("private-key-sentinel".into()),
            api_base: Some(upstream.uri()),
            ..call
        }))
        .await;
    let summaries = traces.summaries("litellm.route");
    assert_eq!(summaries.len(), 1);
    assert_eq!(summaries[0]["route"], "messages");
    assert_eq!(summaries[0]["model"], model);
    assert_eq!(
        summaries[0]["resolved_model"],
        only_request(&upstream).await.json()["model"]
    );
    assert_eq!(summaries[0]["provider"], "anthropic");
    assert_eq!(summaries[0]["outcome"], "success");
    assert_eq!(summaries[0]["stream"], false);
    assert!(summaries[0].get("body").is_none());
    assert!(!format!("{:?}", traces.records()).contains("private-key-sentinel"));
}

#[rstest]
#[case::uncached(false, 2)]
#[case::cached(true, 1)]
#[tokio::test]
async fn route_uses_injected_dependencies_and_optional_cache(
    #[case] caching: bool,
    #[case] expected_requests: usize,
) {
    use litellm_cache_memory::InMemoryCache;
    use litellm_cache_response::{CacheScope, ResponseCache, ScopedCache};
    use litellm_inference_messages::MessagesRoute;

    let upstream = upstream([message_response(), message_response()]).await;
    let resources = resources();
    let route = MessagesRoute::new(
        provider_http(&resources, &http_config()),
        resources.auth.clone(),
        Arc::new(RecordingSecrets::new([("ANTHROPIC_API_KEY", "route-key")])),
    );
    let route = if caching {
        route.with_cache(ScopedCache::new(
            Arc::new(ResponseCache::new(Arc::new(InMemoryCache::new(
                Some(100),
                Some(Duration::from_secs(60)),
            )))),
            CacheScope::Shared,
        ))
    } else {
        route
    };
    for _ in 0..2 {
        let request = MessagesCall {
            api_base: Some(upstream.uri()),
            ..super::call()
        };
        let MessagesCallResponse::Complete(response) =
            route.execute(request, &(), None).await.unwrap()
        else {
            panic!("expected a completed message");
        };
        assert_eq!(
            response.body().content,
            message_body()["content"].as_array().unwrap().as_slice()
        );
    }
    let requests = received(&upstream).await;
    assert_eq!(requests.len(), expected_requests);
    assert_eq!(requests[0].header("x-api-key"), Some("route-key"));
}

#[rstest]
#[tokio::test]
async fn cache_overrides_preserve_the_routes_isolated_scope(call: MessagesCall) {
    use litellm_cache_memory::InMemoryCache;
    use litellm_cache_response::{CachePolicy, CacheScope, ResponseCache, ScopedCache};

    let first_body = message_body();
    let second_body = Value::Object(
        first_body
            .as_object()
            .unwrap()
            .iter()
            .map(|(key, value)| {
                (
                    key.clone(),
                    if key == "id" {
                        json!("msg_second")
                    } else {
                        value.clone()
                    },
                )
            })
            .collect(),
    );
    let upstream = upstream([
        json_response(first_body.clone()),
        json_response(second_body.clone()),
    ])
    .await;
    let service = Arc::new(ResponseCache::new(Arc::new(InMemoryCache::new(
        Some(100),
        Some(Duration::from_secs(60)),
    ))));
    let first = messages_route(no_secrets()).with_cache(ScopedCache::new(
        service.clone(),
        CacheScope::Isolated("first".into()),
    ));
    let second = messages_route(no_secrets()).with_cache(ScopedCache::new(
        service,
        CacheScope::Isolated("second".into()),
    ));
    for (route, expected) in [
        (&first, &first_body),
        (&second, &second_body),
        (&first, &first_body),
        (&second, &second_body),
    ] {
        let request = MessagesCall {
            body: call.body.clone(),
            api_key: Some("same-key".into()),
            api_base: Some(upstream.uri()),
            ..super::call()
        };
        let override_options = CachePolicy {
            ttl: Some(Duration::from_secs(30)),
            ..CachePolicy::default()
        };
        let MessagesCallResponse::Complete(response) =
            route.execute(request, &(), override_options).await.unwrap()
        else {
            panic!("expected a completed message");
        };
        assert_eq!(response.body().id, expected["id"].as_str().unwrap());
    }
    assert_eq!(received(&upstream).await.len(), 2);
}
