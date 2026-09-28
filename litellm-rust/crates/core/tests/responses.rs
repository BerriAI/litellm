use litellm_host::{
    interceptors::{ExecutionFacts, ResultSource},
    lifecycle::ExecutionEvent,
};
use std::sync::Arc;

use futures_util::TryStreamExt;
use litellm_core::responses::{
    route::Responses,
    types::{ResponsesCall, ResponsesOutput},
};
use litellm_host::{call::HostedCompletion, lifecycle::CallEvent};
use rstest::{fixture, rstest};
use serde_json::json;
use wiremock::ResponseTemplate;

mod support;
use support::*;

#[fixture]
fn call() -> ResponsesCall {
    ResponsesCall {
        model: "openai/test-model".into(),
        input: json!("hello"),
        optional_params: Default::default(),
        api_key: Some("test-key".into()),
        api_base: None,
        custom_llm_provider: None,
        extra_headers: None,
        timeout: None,
    }
}

#[rstest]
#[case::direct(false)]
#[case::hosted(true)]
#[tokio::test]
async fn http_responses_share_execution_and_hooks(call: ResponsesCall, #[case] hosted: bool) {
    let body = json!({"id": "response-1", "model": "test-model", "output": [{"type":"message", "content":[]}], "usage":{"total_tokens":7}, "provider_extra": true});
    let upstream = upstream([json_response(body.clone())]).await;
    let host = RecordingCall::<Responses>::new(ResponsesCall {
        api_base: Some(upstream.uri()),
        ..call
    });
    let response = if hosted {
        let HostedCompletion::Complete(response) = litellm_host_native::in_process::run_hosted(
            responses_route(no_secrets())
                .machine(host.request().unwrap(), Some(host.events.0.sender.clone())),
            host.runtime(),
        )
        .await
        .unwrap() else {
            panic!()
        };
        response
    } else {
        let call = host.request.lock().unwrap().take().unwrap();
        let ResponsesOutput::Complete(response) = responses_route(no_secrets())
            .execute(call, &host, Some(host.events.0.sender.clone()))
            .await
            .unwrap()
        else {
            panic!()
        };
        response
    };
    assert_eq!(serde_json::to_value(response).unwrap(), body);
    let sent = only_request(&upstream).await;
    assert_eq!(sent.url.path(), "/responses");
    assert_eq!(sent.header("authorization"), Some("Bearer test-key"));
    assert_eq!(sent.header("x-hook"), Some("called"));
    assert_eq!(sent.json(), json!({"model":"test-model", "input":"hello"}));
    assert!(matches!(
        &host.events.0.lock().unwrap()[..],
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
    ));
}

#[rstest]
#[case::direct(false)]
#[case::hosted(true)]
#[tokio::test]
async fn streaming_keeps_headers_and_bytes_and_finishes_after_consumption(
    call: ResponsesCall,
    #[case] hosted: bool,
) {
    let body = "event: response.completed\ndata: {\"type\":\"response.completed\"}\n\n";
    let upstream = upstream([ResponseTemplate::new(200)
        .insert_header("x-request-id", "response-stream")
        .set_body_raw(body, "text/event-stream")])
    .await;
    let host = RecordingCall::<Responses>::new(ResponsesCall {
        api_base: Some(upstream.uri()),
        optional_params: json!({"stream":true}).as_object().unwrap().clone(),
        ..call
    });
    let (headers, bytes) = if hosted {
        assert_eq!(
            litellm_host_native::in_process::run_hosted(
                responses_route(no_secrets())
                    .machine(host.request().unwrap(), Some(host.events.0.sender.clone())),
                host.runtime(),
            )
            .await
            .unwrap(),
            HostedCompletion::StreamEnded
        );
        (
            host.head.lock().unwrap().take().unwrap().headers,
            host.chunks.lock().unwrap().concat(),
        )
    } else {
        let call = host.request.lock().unwrap().take().unwrap();
        let ResponsesOutput::Stream { head, chunks } = responses_route(no_secrets())
            .execute(call, &host, Some(host.events.0.sender.clone()))
            .await
            .unwrap()
        else {
            panic!()
        };
        assert!(matches!(
            &host.events.0.lock().unwrap()[..],
            [
                CallEvent::Started { .. },
                CallEvent::Execution(ExecutionEvent::ResultReady {
                    facts: ExecutionFacts {
                        source: ResultSource::Provider,
                        ..
                    }
                }),
            ]
        ));
        (
            head.headers,
            chunks.try_collect::<Vec<_>>().await.unwrap().concat(),
        )
    };
    assert!(headers.contains(&("x-request-id".into(), "response-stream".into())));
    assert_eq!(bytes, body.as_bytes());
    assert!(matches!(
        &host.events.0.lock().unwrap()[..],
        [
            CallEvent::Started { .. },
            CallEvent::Execution(ExecutionEvent::ResultReady {
                facts: ExecutionFacts {
                    source: ResultSource::Provider,
                    ..
                }
            }),
            CallEvent::Succeeded { .. }
        ]
    ));
}

#[rstest]
#[case::http(429, json!({"error":"limited"}))]
#[case::invalid_response(200, json!({"unexpected":true}))]
#[tokio::test]
async fn provider_failures_emit_failure_once(
    call: ResponsesCall,
    #[case] status: u16,
    #[case] body: serde_json::Value,
) {
    let upstream = upstream([ResponseTemplate::new(status).set_body_json(body)]).await;
    let host = RecordingCall::<Responses>::new(ResponsesCall {
        api_base: Some(upstream.uri()),
        ..call
    });
    let call = host.request.lock().unwrap().take().unwrap();
    assert!(
        responses_route(no_secrets())
            .execute(call, &host, Some(host.events.0.sender.clone()))
            .await
            .is_err()
    );
    assert_eq!(received(&upstream).await.len(), 1);
    let events = host.events.0.lock().unwrap();
    assert!(matches!(events.last(), Some(CallEvent::Failed { .. })));
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event,
                CallEvent::Failed { .. } | CallEvent::Succeeded { .. }
            ))
            .count(),
        1
    );
}

#[rstest]
#[case::explicit(true)]
#[case::from_secrets(false)]
#[tokio::test]
async fn credentials_and_endpoint_are_resolved_only_when_needed(
    call: ResponsesCall,
    #[case] explicit: bool,
) {
    let upstream = upstream([json_response(
        json!({"id":"response", "model":"test-model", "output":[]}),
    )])
    .await;
    let base = upstream.uri();
    let key = "resolved-test-key";
    let secrets = Arc::new(RecordingSecrets::new([
        ("OPENAI_API_KEY", key),
        ("OPENAI_BASE_URL", base.as_str()),
    ]));
    let call = ResponsesCall {
        api_key: explicit.then(|| key.into()),
        api_base: explicit.then(|| base.clone()),
        ..call
    };
    responses_route(secrets.clone())
        .execute(call, &(), None)
        .await
        .unwrap();
    assert_eq!(
        only_request(&upstream).await.header("authorization"),
        Some(format!("Bearer {key}").as_str())
    );
    if explicit {
        assert!(secrets.requested().is_empty());
    } else {
        assert!(secrets.requested().contains(&"OPENAI_API_KEY".into()));
        assert!(secrets.requested().contains(&"OPENAI_BASE_URL".into()));
    }
}

#[rstest]
#[case::provider("other/model", "other")]
#[case::conflicting_prefix("other/model", "openai")]
#[tokio::test]
async fn unsupported_providers_fail_before_secrets_or_transport(
    call: ResponsesCall,
    #[case] model: &str,
    #[case] provider: &str,
) {
    let upstream = upstream([]).await;
    let secrets = Arc::new(RecordingSecrets::failing());
    let call = ResponsesCall {
        model: model.into(),
        custom_llm_provider: Some(provider.into()),
        api_base: Some(upstream.uri()),
        ..call
    };
    assert!(
        responses_route(secrets.clone())
            .execute(call, &(), None)
            .await
            .is_err()
    );
    assert!(secrets.requested().is_empty());
    assert!(received(&upstream).await.is_empty());
}

#[rstest]
#[case::success(200, json!({"id":"response-1", "model":"test-model", "output":[]}), "success")]
#[case::invalid_response(200, json!("private-response-sentinel"), "failure")]
#[case::upstream_error(429, json!({"error":"private-response-sentinel"}), "failure")]
#[tokio::test]
async fn route_tracing_covers_native_and_hosted_outcomes(
    call: ResponsesCall,
    traces: TraceCapture,
    #[case] status: u16,
    #[case] body: serde_json::Value,
    #[case] outcome: &str,
    #[values(false, true)] hosted: bool,
) {
    let upstream = upstream([status_response(status, body)]).await;
    let host = RecordingCall::<Responses>::new(ResponsesCall {
        api_base: Some(upstream.uri()),
        input: json!("private-prompt-sentinel"),
        api_key: Some("private-key-sentinel".into()),
        ..call
    });
    let route = responses_route(no_secrets());
    let result = traces
        .logger()
        .instrument(async {
            if hosted {
                litellm_host_native::in_process::run_hosted(
                    route
                        .clone()
                        .machine(host.request().unwrap(), Some(host.events.0.sender.clone())),
                    host.runtime(),
                )
                .await
                .map(|_| ())
            } else {
                route
                    .execute(host.request().unwrap(), &(), None)
                    .await
                    .map(|_| ())
            }
        })
        .await;
    assert_eq!(result.is_ok(), outcome == "success");
    let summaries = traces.summaries("litellm.route");
    let [summary] = summaries.as_slice() else {
        panic!("expected one route summary: {summaries:?}")
    };
    assert_eq!(summary["route"], "responses");
    assert_eq!(summary["model"], "openai/test-model");
    assert_eq!(summary["resolved_model"], "test-model");
    assert_eq!(summary["provider"], "openai");
    assert_eq!(summary["stream"], false);
    assert_eq!(summary["outcome"], outcome);
    assert!(summary["duration_ms"].as_f64().unwrap() >= 0.0);
    let sends = traces.summaries("litellm.provider.send");
    assert_eq!(sends.len(), 1);
    assert_eq!(sends[0]["status"], status);
    assert!(!format!("{:?}", traces.records()).contains("private-"));
}

#[rstest]
#[case::exhausted(true, "success")]
#[case::dropped(false, "cancelled")]
#[tokio::test]
async fn stream_trace_survives_handoff_and_closes_before_the_stream_object_is_dropped(
    call: ResponsesCall,
    traces: TraceCapture,
    #[case] exhaust: bool,
    #[case] outcome: &str,
) {
    let upstream = upstream([ResponseTemplate::new(200).set_body_string("stream-bytes")]).await;
    let output = traces
        .logger()
        .instrument(async {
            responses_route(no_secrets())
                .execute(
                    ResponsesCall {
                        api_base: Some(upstream.uri()),
                        optional_params: json!({"stream":true}).as_object().unwrap().clone(),
                        ..call
                    },
                    &(),
                    None,
                )
                .await
        })
        .await
        .unwrap();
    assert!(traces.summaries("litellm.route").is_empty());
    let ResponsesOutput::Stream { mut chunks, .. } = output else {
        panic!("expected stream")
    };
    let captured = traces.clone();
    tokio::spawn(async move {
        if exhaust {
            while chunks.try_next().await.unwrap().is_some() {}
            assert_eq!(captured.summaries("litellm.route").len(), 1);
            assert!(chunks.try_next().await.unwrap().is_none());
        }
        drop(chunks);
    })
    .await
    .unwrap();
    let summaries = traces.summaries("litellm.route");
    assert_eq!(summaries.len(), 1);
    assert_eq!(summaries[0]["stream"], true);
    assert_eq!(summaries[0]["outcome"], outcome);
}

#[rstest]
#[tokio::test]
async fn preparation_failure_is_traced_but_unpolled_builders_are_not(
    call: ResponsesCall,
    traces: TraceCapture,
) {
    traces.logger().scope(|| {
        drop(responses_route(no_secrets()).execute(
            ResponsesCall {
                model: "unknown/model".into(),
                ..call
            },
            &(),
            None,
        ))
    });
    assert!(traces.records().is_empty());
    let result = traces
        .logger()
        .instrument(async {
            responses_route(no_secrets())
                .execute(
                    ResponsesCall {
                        model: "unknown/model".into(),
                        ..self::call()
                    },
                    &(),
                    None,
                )
                .await
        })
        .await;
    assert!(result.is_err());
    let summaries = traces.summaries("litellm.route");
    assert_eq!(summaries.len(), 1);
    assert_eq!(summaries[0]["outcome"], "failure");
    assert!(traces.summaries("litellm.provider.send").is_empty());
}

#[rstest]
#[case::plain("", "/responses")]
#[case::query_fragment("?tenant=a#f", "/responses?tenant=a")]
#[tokio::test]
async fn websocket_operations_trace_outcomes_without_capturing_frames_or_credentials(
    traces: TraceCapture,
    #[case] suffix: &str,
    #[case] expected_target: &str,
) {
    use futures_util::{SinkExt, StreamExt};
    use litellm_core::responses::websocket::ResponsesWebSocketConnection;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let expected_target = expected_target.to_owned();
    let server = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        let mut socket =
            tokio_tungstenite::accept_hdr_async(socket, HandshakeTarget(expected_target))
                .await
                .unwrap();
        let message = socket.next().await.unwrap().unwrap();
        socket.send(message).await.unwrap();
        let _ = socket.next().await;
    });
    traces
        .logger()
        .instrument(async {
            let connection = ResponsesWebSocketConnection::connect_url(
                &format!("ws://{address}/responses{suffix}"),
                &std::collections::HashMap::from([(
                    "authorization".into(),
                    "private-key-sentinel".into(),
                )]),
                None,
            )
            .await
            .unwrap();
            connection
                .send_text("private-frame-sentinel".into())
                .await
                .unwrap();
            assert_eq!(
                connection.recv_text().await.unwrap().as_deref(),
                Some("private-frame-sentinel")
            );
            connection.close().await.unwrap();
            assert!(
                connection
                    .send_text("private-frame-sentinel".into())
                    .await
                    .is_err()
            );
        })
        .await;
    server.await.unwrap();
    let sends = traces.summaries("litellm.websocket.send_text");
    assert_eq!(sends.len(), 2);
    assert_eq!(sends[0]["outcome"], "success");
    assert_eq!(sends[1]["outcome"], "failure");
    assert_eq!(
        traces.summaries("litellm.websocket.connect_url")[0]["outcome"],
        "success"
    );
    assert_eq!(
        traces.summaries("litellm.websocket.recv_text")[0]["outcome"],
        "success"
    );
    assert_eq!(
        traces.summaries("litellm.websocket.close")[0]["outcome"],
        "success"
    );
    assert!(!format!("{:?}", traces.records()).contains("private-"));
}

#[rstest]
#[case::base("/prefix/v1?tenant=a#f")]
#[case::complete("/prefix/v1/responses?tenant=a#f")]
#[tokio::test]
async fn completion_preserves_base_query(call: ResponsesCall, #[case] suffix: &str) {
    let upstream = upstream([json_response(
        json!({"id":"response-1", "model":"test-model", "output":[]}),
    )])
    .await;
    responses_route(no_secrets())
        .execute(
            ResponsesCall {
                api_base: Some(format!("{}{suffix}", upstream.uri())),
                ..call
            },
            &(),
        )
        .await
        .unwrap();
    let sent = only_request(&upstream).await;
    assert_eq!(sent.url.path(), "/prefix/v1/responses");
    assert_eq!(sent.url.query(), Some("tenant=a"));
}

struct RewriteUrl(String);
impl litellm_host::hooks::RouteHooks<litellm_core::responses::Error> for RewriteUrl {
    async fn before_provider_request(
        &self,
        wire: litellm_host::event::WireRequest,
        _: litellm_host::event::RequestContext,
    ) -> Result<litellm_host::event::WireRequest, litellm_core::responses::Error> {
        Ok(litellm_host::event::WireRequest {
            url: self.0.clone(),
            ..wire
        })
    }
    async fn on_event(
        &self,
        _: litellm_host::event::MachineEvent,
    ) -> Result<(), litellm_core::responses::Error> {
        Ok(())
    }
}

#[rstest]
#[case::relative("relative/path")]
#[case::unsupported("ftp://example.test")]
#[tokio::test]
async fn invalid_host_url_fails_before_sending(call: ResponsesCall, #[case] rewritten: &str) {
    let upstream = upstream([]).await;
    let result = responses_route(no_secrets())
        .execute(
            ResponsesCall {
                api_base: Some(upstream.uri()),
                ..call
            },
            &RewriteUrl(rewritten.into()),
        )
        .await;
    assert!(result.is_err());
    assert!(upstream.received_requests().await.unwrap().is_empty());
}

#[rstest]
#[tokio::test]
async fn hook_url_is_the_sent_url(call: ResponsesCall) {
    let upstream = upstream([json_response(
        json!({"id":"response-1", "model":"test-model", "output":[]}),
    )])
    .await;
    responses_route(no_secrets())
        .execute(
            ResponsesCall {
                api_base: Some("https://unused.test".into()),
                ..call
            },
            &RewriteUrl(format!("{}/rewritten?tenant=b#f", upstream.uri())),
        )
        .await
        .unwrap();
    let sent = only_request(&upstream).await;
    assert_eq!(sent.url.path(), "/rewritten");
    assert_eq!(sent.url.query(), Some("tenant=b"));
}

struct HandshakeTarget(String);
impl tokio_tungstenite::tungstenite::handshake::server::Callback for HandshakeTarget {
    fn on_request(
        self,
        request: &tokio_tungstenite::tungstenite::handshake::server::Request,
        response: tokio_tungstenite::tungstenite::handshake::server::Response,
    ) -> Result<
        tokio_tungstenite::tungstenite::handshake::server::Response,
        tokio_tungstenite::tungstenite::handshake::server::ErrorResponse,
    > {
        assert_eq!(request.uri().to_string(), self.0);
        Ok(response)
    }
}
