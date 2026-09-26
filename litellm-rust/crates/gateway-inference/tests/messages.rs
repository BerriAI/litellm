mod support;

use axum::{
    body::{Body, to_bytes},
    http::Request,
};
use rstest::rstest;
use serde_json::json;
use tower::ServiceExt;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_json, header, method, path},
};

#[rstest]
#[case(false)]
#[case(true)]
#[tokio::test]
async fn messages_reaches_the_provider_and_preserves_json_or_sse(#[case] streaming: bool) {
    let upstream = MockServer::start().await;
    let message = json!({"id": "msg_test", "type": "message", "role": "assistant",
        "model": "test-model", "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}});
    let sse = "event: message_start\ndata: {\"type\":\"message_start\"}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
    let template = if streaming {
        ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream")
    } else {
        ResponseTemplate::new(200).set_body_json(message.clone())
    };
    let messages = json!([{"role": "user", "content": "hi"}]);
    Mock::given(method("POST")).and(path("/v1/messages"))
        .and(header("x-api-key", "test-key"))
        .and(header("anthropic-beta", "test-feature"))
        .and(body_json(json!({"model": "test-model", "messages": messages, "max_tokens": 16, "stream": streaming})))
        .respond_with(template).expect(1).mount(&upstream).await;
    let request = Request::post("/v1/messages")
        .header("content-type", "application/json").header("anthropic-beta", "test-feature")
        .body(Body::from(json!({"model": "public/model", "messages": messages, "max_tokens": 16, "stream": streaming}).to_string())).unwrap();
    let response = support::app("anthropic/test-model", &upstream.uri())
        .oneshot(request)
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    if streaming {
        assert_eq!(response.headers()["content-type"], "text/event-stream");
        assert_eq!(to_bytes(response.into_body(), 4096).await.unwrap(), sse);
    } else {
        let body = support::json(response).await;
        assert_eq!(body["content"], message["content"]);
        assert_eq!(body["usage"], message["usage"]);
    }
}

#[tokio::test]
async fn invalid_messages_stays_an_anthropic_error() {
    let response = support::post(
        support::app("anthropic/test-model", "http://127.0.0.1:1"),
        "/v1/messages",
        json!({"model": "public/model", "messages": "invalid", "max_tokens": 16}),
    )
    .await;
    assert_eq!(response.status(), 400);
    let body = support::json(response).await;
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], "invalid_request_error");
}
