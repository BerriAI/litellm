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
#[case::deployment_fills_unset_params(json!({}), json!({"max_output_tokens": 300}))]
#[case::client_params_win(json!({"max_output_tokens": 50}), json!({"max_output_tokens": 50}))]
#[tokio::test]
async fn responses_send_deployment_litellm_params_the_client_leaves_unset(
    #[case] client_params: serde_json::Value,
    #[case] sent_params: serde_json::Value,
) {
    let upstream = MockServer::start().await;
    let sent = json!({"model": "test-model", "input": "hello"});
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(body_json(support::merged(sent, sent_params)))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"id": "response-1", "model": "test-model", "output": []})),
        )
        .expect(1)
        .mount(&upstream)
        .await;
    let app = support::app_with_litellm_params(
        "openai/test-model",
        &upstream.uri(),
        json!({
            "model": "openai/test-model",
            "max_output_tokens": 300,
            "metadata": {"team": "search"},
            "input_cost_per_token": 0.000001,
        }),
    );
    let body = json!({"model": "public/model", "input": "hello"});
    let response = support::post(app, "/v1/responses", support::merged(body, client_params)).await;
    assert_eq!(response.status(), 200);
}

#[rstest]
#[case::missing_input(json!({"model": "public/model"}), 400)]
#[case::invalid_stream(json!({"model": "public/model", "input": "hello", "stream": "yes"}), 400)]
#[case::unknown_model(json!({"model": "unknown", "input": "hello"}), 400)]
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

const RESPONSE_CREATED: &str = "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"response-1\",\"model\":\"test-model\",\"output\":[]}}\n\n";

#[rstest]
#[tokio::test]
async fn a_stream_that_fails_after_opening_ends_with_the_responses_error_event() {
    let base = support::truncating_upstream(RESPONSE_CREATED).await;
    let response = support::post(
        support::app("openai/test-model", &base),
        "/v1/responses",
        json!({"model": "public/model", "input": "hello", "stream": true}),
    )
    .await;
    assert_eq!(response.status(), 200);
    let body = to_bytes(response.into_body(), 4096).await.unwrap();
    let text = std::str::from_utf8(&body).unwrap();
    let frame = text
        .strip_prefix(RESPONSE_CREATED)
        .and_then(|rest| rest.strip_prefix("event: error\ndata: "))
        .unwrap_or_else(|| panic!("the delivered event then one error frame, got {text:?}"));
    let error: serde_json::Value = serde_json::from_str(frame.trim_end()).unwrap();
    assert_eq!(error["type"], "error");
    assert_eq!(error["code"], "500");
    assert!(
        error["message"]
            .as_str()
            .is_some_and(|message| !message.is_empty())
    );
}
