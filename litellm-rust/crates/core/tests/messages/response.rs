use litellm_core::messages::{messages, types::MessagesRequest};
use litellm_http::transport::Error as TransportError;
use rstest::rstest;

use super::*;

#[rstest]
#[tokio::test]
async fn the_provider_message_is_returned(call: MessagesCall) {
    let upstream = upstream([message_response()]).await;

    let message = run_message(MessagesCall {
        api_key: Some("sk".into()),
        api_base: Some(upstream.uri()),
        ..call
    })
    .await;

    assert_eq!(message.id, "msg_1");
    assert_eq!(message.content, [json!({"type": "text", "text": "hi"})]);
    assert_eq!(message.stop_reason.as_deref(), Some("end_turn"));
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
async fn the_facade_runs_the_route_in_process() {
    let upstream = upstream([message_response()]).await;
    let base = upstream.uri();

    let message = messages(facade_request(
        json!({"model": MODEL, "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}),
        &base,
    ))
    .await
    .expect("messages request succeeds");

    assert_eq!(message.id, "msg_1");
    assert_eq!(
        only_request(&upstream).await.header("x-api-key"),
        Some("sk-ant")
    );
}

#[tokio::test]
async fn the_facade_rejects_a_body_that_is_not_an_object() {
    let error = messages(facade_request(json!([]), UNREACHABLE_BASE))
        .await
        .expect_err("a non-object body is rejected");

    assert_eq!(
        error,
        Error::InvalidRequest("messages body must be an object".into())
    );
}
