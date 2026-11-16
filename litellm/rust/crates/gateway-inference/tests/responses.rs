mod support;

use axum::body::to_bytes;
use litellm_inference_responses::ids::{self, IdScope};
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
    let deployment = response.headers()["x-litellm-model-id"]
        .to_str()
        .unwrap()
        .to_owned();
    let managed = ids::response_id(
        &IdScope {
            provider: Some("openai".to_owned()),
            model_id: Some(deployment),
            encrypted_content_affinity: false,
        },
        "response-1",
    );
    let returned =
        json!({"id": managed, "model": "test-model", "output": [], "provider_extra": true});
    if stream {
        assert_eq!(response.headers()["content-type"], "text/event-stream");
        assert_eq!(
            to_bytes(response.into_body(), 4096).await.unwrap().as_ref(),
            format!(
                "event: response.completed\ndata: {}\n\n",
                json!({"type": "response.completed", "response": returned})
            )
            .as_bytes()
        );
    } else {
        assert_eq!(support::json(response).await, returned);
    }
}

/// A follow-up carrying reasoning `east` minted, and continuing a response it served, from a
/// client the Python proxy or this gateway answered before.
fn follow_up() -> serde_json::Value {
    json!({
        "model": "public/model",
        "previous_response_id": ids::response_id(
            &IdScope {
                provider: Some("openai".to_owned()),
                model_id: Some("east".to_owned()),
                encrypted_content_affinity: true,
            },
            "resp_upstream",
        ),
        "input": [
            {"type": "reasoning", "id": ids::encrypted_item_id("east", "rs_1"),
             "encrypted_content": "gAAAA", "summary": [{"type": "summary_text", "text": "thought"}]},
            {"type": "message", "role": "user", "content": "and then?"},
        ],
    })
}

#[rstest]
#[case::encrypted_content_affinity(
    json!({"optional_pre_call_checks": ["encrypted_content_affinity"]})
)]
#[case::responses_api_deployment_check(
    json!({"model_group_affinity_config": {"public/model": ["responses_api_deployment_check"]}})
)]
#[tokio::test]
async fn a_follow_up_reaches_the_deployment_that_served_it_with_the_provider_s_own_ids(
    #[case] router_settings: serde_json::Value,
) {
    let east = MockServer::start().await;
    let west = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(body_json(json!({
            "model": "test-model",
            "previous_response_id": "resp_upstream",
            "input": [
                {"type": "reasoning", "id": "rs_1", "encrypted_content": "gAAAA",
                 "summary": [{"type": "summary_text", "text": "thought"}]},
                {"type": "message", "role": "user", "content": "and then?"},
            ],
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"id": "resp_next", "model": "test-model", "output": []})),
        )
        .expect(5)
        .mount(&east)
        .await;
    let app = support::group_app(
        &[("east", &east.uri()), ("west", &west.uri())],
        router_settings,
    );
    for _ in 0..5 {
        let response = support::post(app.clone(), "/v1/responses", follow_up()).await;
        assert_eq!(response.status(), 200);
        let returned = support::json(response).await["id"]
            .as_str()
            .unwrap()
            .to_owned();
        assert_eq!(
            ids::previous_response_deployment(&returned).as_deref(),
            Some("east")
        );
    }
    assert!(west.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn reasoning_another_account_minted_reaches_a_new_deployment_as_its_summary() {
    let west = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(body_json(json!({
            "model": "test-model",
            "previous_response_id": "resp_upstream",
            "input": [
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thought"}]},
                {"type": "message", "role": "user", "content": "and then?"},
            ],
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"id": "resp_next", "model": "test-model", "output": []})),
        )
        .expect(1)
        .mount(&west)
        .await;
    let app = support::group_app(
        &[("west", &west.uri())],
        json!({"optional_pre_call_checks": ["encrypted_content_affinity"]}),
    );
    let response = support::post(app, "/v1/responses", follow_up()).await;
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
async fn responses_preserve_upstream_errors_once_the_router_retries_run_out() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(429).set_body_json(json!({"error": {"message": "slow down"}})),
        )
        .expect(3)
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
        .split_once("\n\nevent: error\ndata: ")
        .filter(|(delivered, _)| delivered.starts_with("event: response.created\n"))
        .map(|(_, frame)| frame)
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
