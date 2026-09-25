use litellm_core::{
    Phase,
    messages::{MessagesResponse, messages, messages_body},
};
use litellm_http::transport::Error as TransportError;
use rstest::rstest;

use super::*;

#[rstest]
#[case::anthropic("anthropic")]
#[case::azure_ai("azure_ai")]
#[tokio::test]
async fn the_provider_message_is_returned(call: MessagesCall, #[case] provider: &str) {
    let upstream = upstream([message_response()]).await;

    let message = run_message(MessagesCall {
        custom_llm_provider: Some(provider.into()),
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
    .err()
    .expect("upstream error propagates");

    let Error::Transport(TransportError::Http { status, body }) = error else {
        panic!("{error:?}");
    };
    assert_eq!(status, 400);
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap(), envelope);
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
    .err()
    .expect("upstream error propagates");

    assert_eq!(
        error,
        Error::Transport(TransportError::Http {
            status: 500,
            body: format!("{}... (truncated)", &long[..256])
        })
    );
}

#[rstest]
#[case::bad_request(400)]
#[case::unauthorized(401)]
#[case::rate_limited(429)]
#[case::server_error(500)]
#[case::overloaded(529)]
#[tokio::test]
async fn an_upstream_error_keeps_its_status_and_body(call: MessagesCall, #[case] status: u16) {
    let upstream =
        upstream([ResponseTemplate::new(status).set_body_string("upstream said no")]).await;

    let error = run(MessagesCall {
        api_key: Some("sk".into()),
        api_base: Some(upstream.uri()),
        ..call
    })
    .await
    .err()
    .expect("upstream error propagates");

    assert_eq!(
        error,
        Error::Transport(TransportError::Http {
            status,
            body: "upstream said no".into()
        })
    );
}

#[rstest]
#[tokio::test]
async fn an_invalid_thinking_signature_retries_without_replayed_thinking(call: MessagesCall) {
    let upstream = upstream([
        status_response(
            400,
            json!({
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "messages.3.content.0.thinking.signature.str: Input should be a valid string"
                }
            }),
        ),
        message_response(),
    ])
    .await;
    let history = json!([
        {"role": "user", "content": [{"type": "text", "text": "first question"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "first answer"}]},
        {"role": "user", "content": [{"type": "text", "text": "second question"}]},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "replayed from another provider", "signature": null},
            {"type": "tool_use", "id": "call-1", "name": "lookup", "input": {"key": "value"}}
        ]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call-1", "content": "found"}]}
    ]);

    let response = run(MessagesCall {
        api_key: Some("sk-ant".into()),
        api_base: Some(upstream.uri()),
        body: object(json!({
            "model": MODEL,
            "max_tokens": 64,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "tools": [{"name": "lookup", "input_schema": {"type": "object", "properties": {"key": {"type": "string"}}}}],
            "messages": history,
        })),
        ..call
    })
    .await;
    let requests = received(&upstream).await;

    assert_eq!(
        requests.len(),
        2,
        "expected one recovery retry after the signature error"
    );
    let first = requests[0].json();
    let retry = requests[1].json();
    assert_eq!(first["messages"][3]["content"][0]["type"], "thinking");
    assert_eq!(
        first["thinking"],
        json!({"type": "enabled", "budget_tokens": 1024})
    );
    assert_eq!(
        retry["messages"],
        json!([
            {"role": "user", "content": [{"type": "text", "text": "first question"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "first answer"}]},
            {"role": "user", "content": [{"type": "text", "text": "second question"}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "call-1", "name": "lookup", "input": {"key": "value"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call-1", "content": "found"}]}
        ])
    );
    assert!(retry.get("thinking").is_none());
    assert!(matches!(response, Ok(MessagesOutput::Message(_))));
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
    .err()
    .expect("an unreadable body fails");

    assert_eq!(error.phase(), Phase::AfterSend, "{error:?}");
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
    .err()
    .expect("the call times out");

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

    let response = messages(
        &support::resources(),
        &Resolution::from(&settings).config,
        &RecordingSecrets::empty(),
        MessagesCall {
            api_key: Some("sk-ant".into()),
            api_base: Some(base),
            ..call
        },
    )
    .await
    .expect("messages request succeeds");

    let MessagesResponse::Message(message) = response else {
        panic!("a non-streaming request returns a message");
    };
    assert_eq!(message.id, "msg_1");
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
        matches!(&error, Error::InvalidRequest(message) if message.starts_with("invalid Anthropic messages request: ")),
        "{error:?}"
    );
}

#[rstest]
#[case::unrelated_bad_request(400, "invalid tool signature", 1)]
#[case::server_error(500, "invalid thinking signature", 1)]
#[case::bounded_recovery(400, "invalid thinking signature", 2)]
#[tokio::test]
async fn thinking_recovery_is_specific_and_bounded(
    call: MessagesCall,
    #[case] status: u16,
    #[case] message: &str,
    #[case] attempts: usize,
) {
    let error = json!({"error": {"type": "invalid_request_error", "message": message}});
    let upstream = upstream([
        status_response(status, error.clone()),
        status_response(status, error),
    ])
    .await;
    let result = run(MessagesCall {
        api_key: Some("sk-ant".into()),
        api_base: Some(upstream.uri()),
        ..call
    })
    .await;
    assert!(
        matches!(result, Err(Error::Transport(TransportError::Http { status: actual, .. })) if actual == status)
    );
    assert_eq!(received(&upstream).await.len(), attempts);
}
