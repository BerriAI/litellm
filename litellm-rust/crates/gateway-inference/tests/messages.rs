mod support;

use axum::{
    body::{Body, to_bytes},
    http::Request,
};
use rstest::rstest;
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tower::ServiceExt;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_json, header, method, path},
};

#[rstest]
#[case::anthropic("anthropic/test-model", "/v1/messages", true)]
#[case::azure("azure_ai/test-model", "/anthropic/v1/messages", false)]
#[case::deepseek("deepseek/test-model", "/anthropic/v1/messages", false)]
#[tokio::test]
async fn messages_reaches_the_provider_and_preserves_json_or_sse(
    #[case] model: &str,
    #[case] upstream_path: &str,
    #[case] anthropic_headers: bool,
    #[values(false, true)] streaming: bool,
) {
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
    let mut mock = Mock::given(method("POST"))
        .and(path(upstream_path))
        .and(header("x-api-key", "test-key"));
    if anthropic_headers {
        mock = mock
            .and(header("anthropic-beta", "test-feature"))
            .and(header("anthropic-version", "test-version"));
    }
    mock.and(body_json(
        json!({"model": "test-model", "messages": messages, "max_tokens": 16, "stream": streaming}),
    ))
    .respond_with(support::provider_headers(template))
    .expect(1)
    .mount(&upstream)
    .await;
    let request = Request::post("/v1/messages")
        .header("content-type", "application/json").header("anthropic-beta", "test-feature")
        .header("anthropic-version", "test-version")
        .header("x-api-key", "caller-key")
        .header("authorization", "Bearer proxy-key")
        .header("x-request-id", "caller-request")
        .body(Body::from(json!({"model": "public/model", "messages": messages, "max_tokens": 16, "stream": streaming}).to_string())).unwrap();
    let response = support::app(model, &upstream.uri())
        .oneshot(request)
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    support::assert_provider_headers(response.headers());
    if streaming {
        assert_eq!(response.headers()["content-type"], "text/event-stream");
        assert_eq!(to_bytes(response.into_body(), 4096).await.unwrap(), sse);
    } else {
        let body = support::json(response).await;
        assert_eq!(body["content"], message["content"]);
        assert_eq!(body["usage"], message["usage"]);
    }
    let requests = upstream.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    assert!(!requests[0].headers.contains_key("authorization"));
    assert!(!requests[0].headers.contains_key("x-request-id"));
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

/// Answers with the SSE head and one event, then drops the connection short of the
/// announced body length.
async fn truncating_upstream() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = vec![0; 4096];
        let _ = socket.read(&mut request).await;
        socket
            .write_all(
                format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\n\r\n{FIRST_EVENT}",
                    FIRST_EVENT.len() * 2
                )
                .as_bytes(),
            )
            .await
            .unwrap();
    });
    base
}

const FIRST_EVENT: &str = "event: message_start\ndata: {}\n\n";

#[tokio::test]
async fn a_stream_that_fails_after_opening_ends_with_an_sse_error_frame() {
    let base = truncating_upstream().await;
    let request = Request::post("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"model": "public/model", "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16, "stream": true})
            .to_string(),
        ))
        .unwrap();

    let response = support::app("anthropic/test-model", &base)
        .oneshot(request)
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let body = to_bytes(response.into_body(), 4096).await.unwrap();
    let text = std::str::from_utf8(&body).unwrap();
    let frame = text
        .strip_prefix(FIRST_EVENT)
        .and_then(|rest| rest.strip_prefix("event: error\ndata: "))
        .unwrap_or_else(|| panic!("the delivered event then one error frame, got {text:?}"));
    let error: serde_json::Value = serde_json::from_str(frame.trim_end()).unwrap();
    assert_eq!(error["type"], "error");
    assert_eq!(error["error"]["type"], "api_error");
}

#[rstest]
#[tokio::test]
async fn hosted_provider_failure_preserves_status_body_and_request_id() {
    let upstream = MockServer::start().await;
    let error =
        json!({"type": "error", "error": {"type": "rate_limit_error", "message": "retry later"}});
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(429).set_body_json(error.clone()))
        .expect(1)
        .mount(&upstream)
        .await;
    let request = Request::post("/v1/messages")
        .header("content-type", "application/json")
        .header("x-request-id", "host-http-request")
        .body(Body::from(json!({
            "model": "public/model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16,
        }).to_string()))
        .unwrap();
    let response = support::app("anthropic/test-model", &upstream.uri())
        .oneshot(request)
        .await
        .unwrap();
    assert_eq!(response.status(), 429);
    let body = support::json(response).await;
    assert_eq!(body["type"], error["type"]);
    assert_eq!(body["error"], error["error"]);
    assert_eq!(body["request_id"], "host-http-request");
}

#[rstest]
#[case::messages("/v1/messages")]
#[case::chat("/v1/chat/completions")]
#[case::model_path("/engines/path-model/chat/completions")]
#[case::ocr("/ocr")]
#[case::transcription("/audio/transcriptions")]
#[tokio::test]
async fn model_permissions_prevent_provider_calls(#[case] path: &str) {
    let upstream = MockServer::start().await;
    let app = support::app_with_permissions(
        "anthropic/test-model",
        &upstream.uri(),
        litellm_gateway_auth::Permissions::None,
    );
    let response = support::post(app, path, json!({"model": "public/model"})).await;
    assert_eq!(response.status(), 403);
    assert!(upstream.received_requests().await.unwrap().is_empty());
}
