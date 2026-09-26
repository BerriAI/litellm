mod support;

use std::{sync::Arc, time::Duration};

use axum::{
    Router,
    body::{Body, to_bytes},
    http::Request,
    routing::post,
};
use futures_util::StreamExt;
use rstest::rstest;
use serde_json::json;
use tokio::sync::Notify;
use tower::ServiceExt;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_json, header, method, path},
};

#[rstest]
#[case::unversioned("/responses")]
#[case::versioned("/v1/responses")]
#[tokio::test]
async fn create_resolves_alias_and_forwards_only_provider_headers(#[case] route: &str) {
    let upstream = MockServer::start().await;
    let expected = json!({"id": "resp_test", "object": "response", "output": [], "usage": null, "status": "queued"});
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(header("authorization", "Bearer test-key"))
        .and(header("openai-project", "project-test"))
        .and(header("openai-organization", "org-test"))
        .and(header("openai-beta", "test-beta"))
        .and(header("x-client-request-id", "caller-request"))
        .and(body_json(json!({"model": "test-model", "input": "hello", "previous_response_id": "resp_previous", "store": false})))
        .respond_with(ResponseTemplate::new(200).set_body_json(&expected)
            .insert_header("x-request-id", "upstream-request")
            .insert_header("x-ratelimit-remaining-requests", "12")
            .insert_header("set-cookie", "private=value"))
        .expect(1).mount(&upstream).await;
    let response = support::app("openai/test-model", &upstream.uri()).oneshot(
        Request::post(route)
            .header("authorization", "Bearer gateway-key")
            .header("openai-project", "project-test")
            .header("openai-organization", "org-test")
            .header("openai-beta", "test-beta")
            .header("x-client-request-id", "caller-request")
            .header("x-private-header", "private")
            .body(Body::from(json!({"model": "public/model", "input": "hello", "previous_response_id": "resp_previous", "store": false}).to_string()))
            .unwrap()
    ).await.unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["x-request-id"], "upstream-request");
    assert_eq!(response.headers()["x-ratelimit-remaining-requests"], "12");
    assert!(!response.headers().contains_key("set-cookie"));
    assert_eq!(support::json(response).await, expected);
    assert!(
        !upstream.received_requests().await.unwrap()[0]
            .headers
            .contains_key("x-private-header")
    );
}

#[rstest]
#[case::unknown_model(json!({"model": "missing", "input": "hi"}), 400)]
#[case::missing_model(json!({"input": "hi"}), 400)]
#[case::bad_input(json!({"model": "public/model", "input": 7}), 400)]
#[case::bad_stream(json!({"model": "public/model", "input": "hi", "stream": "true"}), 400)]
#[case::non_object(json!([]), 400)]
#[tokio::test]
async fn invalid_requests_return_openai_errors(
    #[case] body: serde_json::Value,
    #[case] status: u16,
) {
    let response = support::post(
        support::app("openai/test-model", "http://127.0.0.1:1"),
        "/responses",
        body,
    )
    .await;
    assert_eq!(response.status(), status);
    assert_eq!(
        support::json(response).await["error"]["type"],
        "invalid_request_error"
    );
}

#[rstest]
#[case::json(false)]
#[case::stream(true)]
#[tokio::test]
async fn upstream_failure_is_an_http_error_before_streaming(#[case] streaming: bool) {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(429).set_body_json(json!({"error": {"message": "slow down"}})),
        )
        .expect(1)
        .mount(&upstream)
        .await;
    let response = support::post(
        support::app("openai/test-model", &upstream.uri()),
        "/v1/responses",
        json!({"model": "public/model", "input": "hi", "stream": streaming}),
    )
    .await;
    assert_eq!(response.status(), 429);
    assert_eq!(response.headers()["content-type"], "application/json");
    let body = support::json(response).await;
    assert_eq!(body["error"]["type"], "rate_limit_error");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("slow down")
    );
}

#[rstest]
#[case::unversioned("/responses")]
#[case::versioned("/v1/responses")]
#[tokio::test]
async fn streaming_delivers_events_before_upstream_finishes(#[case] route: &str) {
    let first = "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n\n";
    let last = "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_test\"}}\n\n";
    let release = Arc::new(Notify::new());
    let signal = release.clone();
    let upstream = Router::new().route(
        "/responses",
        post(move || {
            let signal = signal.clone();
            async move {
                let stream =
                    futures_util::stream::unfold((0, signal), move |(index, signal)| async move {
                        match index {
                            0 => Some((Ok::<_, std::convert::Infallible>(first), (1, signal))),
                            1 => {
                                signal.notified().await;
                                Some((Ok(last), (2, signal)))
                            }
                            _ => None,
                        }
                    });
                (
                    [("content-type", "text/event-stream; charset=utf-8")],
                    Body::from_stream(stream),
                )
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, upstream).await.unwrap() });
    let response = tokio::time::timeout(
        Duration::from_secs(2),
        support::post(
            support::app("openai/test-model", &base),
            route,
            json!({"model": "public/model", "input": "hi", "stream": true}),
        ),
    )
    .await
    .expect("stream headers must not wait for completion");
    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    let mut chunks = response.into_body().into_data_stream();
    let chunk = tokio::time::timeout(Duration::from_secs(2), chunks.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(chunk, first);
    release.notify_one();
    let rest = to_bytes(Body::from_stream(chunks), 1024).await.unwrap();
    assert_eq!(rest, last);
    server.abort();
}
