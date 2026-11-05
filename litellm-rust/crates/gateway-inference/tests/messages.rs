mod support;

use axum::{
    body::{Body, to_bytes},
    http::Request,
};
use futures_util::{StreamExt, TryStreamExt};
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
#[case::deepseek("deepseek/test-model", "/anthropic/v1/messages", true)]
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
    .respond_with(template)
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

#[rstest]
#[case::invalid_json("{".into())]
#[case::array("[]".into())]
#[case::null("null".into())]
#[case::missing_model(json!({"messages": [], "max_tokens": 16}).to_string())]
#[case::unknown_model(json!({"model": "missing/model", "messages": [], "max_tokens": 16}).to_string())]
#[case::invalid_messages(json!({"model": "public/model", "messages": "invalid", "max_tokens": 16}).to_string())]
#[case::missing_max_tokens(json!({"model": "public/model", "messages": [{"role": "user", "content": "hello"}]}).to_string())]
#[tokio::test]
async fn invalid_messages_stays_an_anthropic_error_without_provider_io(#[case] body: String) {
    let upstream = MockServer::start().await;
    let request = Request::post("/v1/messages")
        .header("content-type", "application/json")
        .header("x-request-id", "invalid-messages-request")
        .body(Body::from(body))
        .unwrap();
    let response = support::app("anthropic/test-model", &upstream.uri())
        .oneshot(request)
        .await
        .unwrap();

    assert_eq!(response.status(), 400);
    let body = support::json(response).await;
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(body["request_id"], "invalid-messages-request");
    assert!(!body["error"]["message"].as_str().unwrap().is_empty());
    assert!(upstream.received_requests().await.unwrap().is_empty());
}

const FIRST_EVENT: &str = "event: message_start\ndata: {}\n\n";

async fn interrupted_stream(
    prefix: &'static [u8],
) -> (
    String,
    tokio::sync::oneshot::Sender<()>,
    tokio::task::JoinHandle<()>,
) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let (release, released) = tokio::sync::oneshot::channel();
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = vec![0; 4096];
        assert!(socket.read(&mut request).await.unwrap() > 0);
        socket.write_all(
            format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\n\r\n",
                prefix.len() + 1024
            ).as_bytes()
        ).await.unwrap();
        socket.write_all(prefix).await.unwrap();
        let _ = released.await;
    });
    (base, release, task)
}

#[rstest]
#[case::complete_lf(FIRST_EVENT.as_bytes())]
#[case::complete_crlf(b"event: message_start\r\ndata: {}\r\n\r\n")]
#[case::partial_event(b"event: content_blo")]
#[case::partial_data(b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\"")]
#[case::partial_string(b"event: content_block_delta\ndata: {\"delta\":{\"text\":\"partial")]
#[case::partial_utf8(b"event: content_block_delta\ndata: {\"delta\":{\"text\":\"\xe9")]
#[tokio::test]
async fn a_stream_that_fails_after_opening_ends_with_an_sse_error_frame(
    #[case] prefix: &'static [u8],
) {
    let (base, release, upstream) = interrupted_stream(prefix).await;
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
    let mut chunks = response.into_body().into_data_stream();
    let mut delivered = Vec::with_capacity(prefix.len());
    while delivered.len() < prefix.len() {
        let chunk = tokio::time::timeout(std::time::Duration::from_secs(5), chunks.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        delivered.extend_from_slice(&chunk);
    }
    assert_eq!(delivered, prefix);
    release.send(()).unwrap();
    let tail = chunks.try_collect::<Vec<_>>().await.unwrap().concat();
    upstream.await.unwrap();
    let text = std::str::from_utf8(&tail).unwrap();
    let frame = text
        .strip_prefix("\n\nevent: error\ndata: ")
        .unwrap_or_else(|| {
            panic!("an independent error frame after the delivered prefix, got {text:?}")
        });
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

#[rstest]
#[case::completion(false)]
#[case::streaming(true)]
#[tokio::test]
async fn openrouter_receives_native_messages_headers_and_its_deployment_bearer(
    #[case] streaming: bool,
) {
    let upstream = MockServer::start().await;
    let message = json!({"id": "msg_test", "type": "message", "role": "assistant", "model": "anthropic/test-model", "content": [{"type": "text", "text": "hello"}], "stop_reason": "end_turn", "stop_sequence": null, "usage": {"input_tokens": 1, "output_tokens": 1, "cost": 0.001, "cost_details": "unexpected"}, "openrouter_metadata": {"provider": "test-provider"}});
    let sse = "event: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"cost\":0.001}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
    let template = if streaming {
        ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream")
    } else {
        ResponseTemplate::new(200).set_body_json(message.clone())
    };
    Mock::given(method("POST"))
        .and(path("/api/v1/messages"))
        .and(header("authorization", "Bearer test-key"))
        .and(header("anthropic-beta", "test-feature"))
        .and(header("anthropic-version", "test-version"))
        .and(header("x-openrouter-metadata", "enabled"))
        .and(body_json(json!({"model": "anthropic/test-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16, "stream": streaming, "provider": {"order": ["test-provider"]}})))
        .respond_with(template).expect(1).mount(&upstream).await;
    let request = Request::post("/v1/messages")
        .header("content-type", "application/json")
        .header("anthropic-beta", "test-feature")
        .header("anthropic-version", "test-version")
        .header("x-openrouter-metadata", "enabled")
        .header("x-api-key", "proxy-key")
        .header("authorization", "Bearer proxy-key")
        .body(Body::from(json!({"model": "public/model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16, "stream": streaming, "provider": {"order": ["test-provider"]}}).to_string())).unwrap();
    let response = support::app(
        "openrouter/anthropic/test-model",
        &format!("{}/api/v1", upstream.uri()),
    )
    .oneshot(request)
    .await
    .unwrap();
    assert_eq!(response.status(), 200);
    if streaming {
        assert_eq!(to_bytes(response.into_body(), 4096).await.unwrap(), sse);
    } else {
        assert_eq!(support::json(response).await, message);
    }
    let requests = upstream.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    assert!(!requests[0].headers.contains_key("x-api-key"));
}

#[rstest]
#[case::u64_max("18446744073709551615".to_string(), 200)]
#[case::above_u64("18446744073709551617".to_string(), 500)]
#[case::i64_min("-9223372036854775808".to_string(), 200)]
#[case::below_i64("-9223372036854775809".to_string(), 500)]
#[case::long_numeric_string(format!(r#""{}""#, "9".repeat(4096)), 200)]
#[case::decimal("18446744073709551617.0".to_string(), 200)]
#[case::exponent("1e30".to_string(), 200)]
#[tokio::test]
async fn messages_response_integer_overflow_is_a_provider_error_not_rounded_success(
    #[case] token: String,
    #[case] status: u16,
) {
    let upstream = MockServer::start().await;
    let body = format!(
        r#"{{"id":"m","type":"message","role":"assistant","model":"test-model","content":[],"future_extension":{{"wide_integer":{token}}}}}"#,
    );
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "application/json"))
        .expect(1)
        .mount(&upstream)
        .await;
    let request = Request::post("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(json!({"model":"public/model","messages":[{"role":"user","content":"hi"}],"max_tokens":16}).to_string()))
        .unwrap();
    let response = support::app("anthropic/test-model", &upstream.uri())
        .oneshot(request)
        .await
        .unwrap();
    assert_eq!(response.status(), status);
    let response = support::json(response).await;
    if status == 200 {
        assert_eq!(
            response["future_extension"]["wide_integer"],
            serde_json::from_str::<serde_json::Value>(&token).unwrap()
        );
    } else {
        assert_eq!(response["type"], "error");
        assert_eq!(response["error"]["type"], "api_error");
        assert!(
            response["error"]["message"]
                .as_str()
                .unwrap()
                .contains("response integer outside")
        );
    }
    assert_eq!(upstream.received_requests().await.unwrap().len(), 1);
}
