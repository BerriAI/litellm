use litellm_core::messages::{messages, types::MessagesRequest};
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

    assert!(error.is_response(), "{error:?}");
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

fn facade_request(body: Value, api_base: &str) -> MessagesRequest<'_> {
    MessagesRequest {
        model: MODEL,
        body,
        api_key: Some("sk-ant"),
        api_base: Some(api_base),
        custom_llm_provider: Some("anthropic"),
        extra_headers: None,
        provider_specific_header: None,
        timeout: Some(Duration::from_secs(5)),
        shaping: MessagesShaping::default(),
    }
}

#[tokio::test]
async fn the_facade_sends_through_the_injected_http_pool_configuration() {
    let upstream = upstream([message_response()]).await;
    let base = upstream.uri();
    let settings = HttpSettings {
        user_agent: Some("host-owned/1".into()),
        ..HttpSettings::default()
    };

    let message = messages(
        &http_pool(),
        &Resolution::from(&settings).config,
        facade_request(
            json!({"model": MODEL, "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}),
            &base,
        ),
    )
    .await
    .expect("messages request succeeds");

    assert_eq!(message.id, "msg_1");
    let sent = only_request(&upstream).await;
    assert_eq!(sent.header("x-api-key"), Some("sk-ant"));
    assert_eq!(sent.header("user-agent"), Some("host-owned/1"));
}

#[tokio::test]
async fn the_facade_rejects_a_body_that_is_not_an_object() {
    let error = messages(
        &http_pool(),
        &http_config(),
        facade_request(json!([]), UNREACHABLE_BASE),
    )
    .await
    .expect_err("a non-object body is rejected");

    assert_eq!(
        error,
        Error::InvalidRequest("messages body must be an object".into())
    );
}
