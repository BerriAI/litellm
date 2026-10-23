mod support;

use rstest::rstest;
use serde_json::{Value, json};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_json, header, method, path},
};

#[rstest]
#[case::unversioned("/tokenize")]
#[case::versioned("/v1/tokenize")]
#[tokio::test]
async fn model_aliases_reach_tokenization_with_deployment_credentials(#[case] endpoint: &str) {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/tokenize"))
        .and(header("authorization", "Bearer test-key"))
        .and(body_json(
            json!({"model":"org/model","prompt":"hello","add_special_tokens":true,"return_token_strs":true}),
        ))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"count":1,"tokens":[42],"token_strs":["hello"]})),
        )
        .expect(1)
        .mount(&upstream)
        .await;
    let response = support::post(
        support::app("hosted_vllm/org/model", &format!("{}/v1", upstream.uri())),
        endpoint,
        json!({"model":"public/model","prompt":"hello"}),
    )
    .await;
    assert_eq!(response.status(), 200);
    assert_eq!(
        support::json(response).await,
        json!({"model":"org/model","count":1,"token_ids":[42],"tokens":["hello"]})
    );
}

#[rstest]
#[case::missing_prompt(json!({"model":"public/model"}))]
#[case::wrong_prompt_type(json!({"model":"public/model","prompt":[]}))]
#[case::unknown_model(json!({"model":"missing","prompt":"hi"}))]
#[case::connection_override(json!({"model":"public/model","prompt":"hi","api_base":"https://example.com"}))]
#[tokio::test]
async fn invalid_requests_do_not_reach_the_provider(#[case] body: Value) {
    let upstream = MockServer::start().await;
    let response = support::post(
        support::app("hosted_vllm/model", &upstream.uri()),
        "/tokenize",
        body,
    )
    .await;
    assert_eq!(response.status(), 400);
    assert_eq!(
        support::json(response).await["error"]["type"],
        "invalid_request_error"
    );
    assert!(upstream.received_requests().await.unwrap().is_empty());
}

#[rstest]
#[tokio::test]
async fn model_permissions_are_checked_before_tokenization() {
    let upstream = MockServer::start().await;
    let response = support::post(
        support::app_with_permissions(
            "hosted_vllm/model",
            &upstream.uri(),
            litellm_gateway_auth::Permissions::None,
        ),
        "/tokenize",
        json!({"model":"public/model","prompt":"hi"}),
    )
    .await;
    assert_eq!(response.status(), 403);
    assert!(upstream.received_requests().await.unwrap().is_empty());
}

#[rstest]
#[tokio::test]
async fn upstream_errors_keep_their_status_in_the_error_envelope() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(429).set_body_json(json!({"error":"busy"})))
        .expect(1)
        .mount(&upstream)
        .await;
    let response = support::post(
        support::app("hosted_vllm/model", &upstream.uri()),
        "/tokenize",
        json!({"model":"public/model","prompt":"hi"}),
    )
    .await;
    assert_eq!(response.status(), 429);
    assert_eq!(support::json(response).await["error"]["code"], 429);
}
