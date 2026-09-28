mod support;

use axum::body::to_bytes;
use rstest::rstest;
use serde_json::json;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_json, header, method, path},
};

#[rstest]
#[case::completed(false)]
#[case::streaming(true)]
#[tokio::test]
async fn responses_aliases_run_the_core_route(
    #[case] stream: bool,
    #[values("/responses", "/v1/responses")] route: &str,
) {
    let upstream = MockServer::start().await;
    let completed =
        json!({"id": "response-1", "model": "test-model", "output": [], "provider_extra": true});
    let events = format!(
        "event: response.completed\ndata: {}\n\n",
        json!({"type": "response.completed", "response": completed})
    );
    let template = if stream {
        ResponseTemplate::new(200)
            .insert_header("content-type", "text/event-stream")
            .set_body_string(&events)
    } else {
        ResponseTemplate::new(200).set_body_json(&completed)
    };
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(header("authorization", "Bearer test-key"))
        .and(body_json(json!({"model": "test-model", "input": "hello", "stream": stream, "metadata": {"caller": "test"}})))
        .respond_with(template)
        .expect(1)
        .mount(&upstream).await;
    let response = support::post(support::app("openai/test-model", &upstream.uri()), route,
        json!({"model": "public/model", "input": "hello", "stream": stream, "metadata": {"caller": "test"}})).await;
    assert_eq!(response.status(), 200);
    if stream {
        assert_eq!(response.headers()["content-type"], "text/event-stream");
        assert_eq!(
            to_bytes(response.into_body(), 4096).await.unwrap().as_ref(),
            events.as_bytes()
        );
    } else {
        assert_eq!(support::json(response).await, completed);
    }
}

#[rstest]
#[case::missing_input(json!({"model": "public/model"}), 400)]
#[case::invalid_stream(json!({"model": "public/model", "input": "hello", "stream": "yes"}), 400)]
#[case::unknown_model(json!({"model": "unknown", "input": "hello"}), 400)]
#[case::invalid_extra_body(json!({"model": "public/model", "input": "hello", "extra_body": []}), 400)]
#[case::late_stream_override(json!({"model": "public/model", "input": "hello", "extra_body": {"stream": true}}), 400)]
#[tokio::test]
async fn invalid_responses_requests_never_reach_the_provider(
    #[case] body: serde_json::Value,
    #[case] status: u16,
) {
    let upstream = MockServer::start().await;
    let response = support::post(
        support::app("openai/test-model", &upstream.uri()),
        "/v1/responses",
        body,
    )
    .await;
    assert_eq!(response.status(), status);
    assert_eq!(support::json(response).await["error"]["code"], status);
    assert!(upstream.received_requests().await.unwrap().is_empty());
}

#[rstest]
#[tokio::test]
async fn responses_preserve_upstream_errors_without_retry() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(429).set_body_json(json!({"error": {"message": "slow down"}})),
        )
        .expect(1)
        .mount(&upstream)
        .await;
    let response = support::post(
        support::app("openai/test-model", &upstream.uri()),
        "/v1/responses",
        json!({"model": "public/model", "input": "hello"}),
    )
    .await;
    assert_eq!(response.status(), 429);
    assert!(
        support::json(response).await["error"]["message"]
            .as_str()
            .unwrap()
            .contains("slow down")
    );
}

#[rstest]
#[tokio::test]
async fn responses_authorize_the_public_model_before_provider_execution() {
    let upstream = MockServer::start().await;
    let app = support::app_with_permissions(
        "openai/test-model",
        &upstream.uri(),
        litellm_gateway_auth::Permissions::None,
    );
    let response = support::post(
        app,
        "/v1/responses",
        json!({"model": "public/model", "input": "hello"}),
    )
    .await;
    assert_eq!(response.status(), 403);
    assert!(upstream.received_requests().await.unwrap().is_empty());
}
