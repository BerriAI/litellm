mod support;

use litellm_core_utils::decline::Decline;
use rstest::rstest;
use serde_json::{Value, json};
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::any};

async fn provider() -> MockServer {
    let provider = MockServer::start().await;
    Mock::given(any())
        .respond_with(ResponseTemplate::new(500).set_body_string("provider failed"))
        .mount(&provider)
        .await;
    provider
}

#[rstest]
#[case::chat_streaming(
    "anthropic/test-model",
    "/v1/chat/completions",
    json!({"model": "public/model", "messages": [{"role": "user", "content": "hi"}], "stream": true}),
    "streaming"
)]
#[case::messages_bedrock_invoke(
    "bedrock/invoke/anthropic.claude-3-5-sonnet-20240620-v1:0",
    "/v1/messages",
    json!({"model": "public/model", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}),
    "Bedrock invoke messages request shaping"
)]
#[case::responses_provider(
    "gemini/test-model",
    "/v1/responses",
    json!({"model": "public/model", "input": "hi"}),
    "native HTTP responses provider"
)]
#[tokio::test]
async fn a_core_decline_is_a_501_that_carries_the_decline(
    #[case] model: &str,
    #[case] path: &str,
    #[case] body: Value,
    #[case] reason: &'static str,
) {
    let provider = provider().await;
    let response = support::post(support::app(model, &provider.uri()), path, body).await;

    assert_eq!(response.status(), 501);
    assert_eq!(
        response.extensions().get::<Decline>(),
        Some(&Decline::new(reason))
    );
    let body = support::json(response).await;
    assert!(body.to_string().contains(reason));
    assert!(provider.received_requests().await.unwrap().is_empty());
}

#[rstest]
#[case::invalid_request(json!({"model": "public/model", "messages": "hi"}), 400)]
#[case::unknown_model(json!({"model": "other", "messages": [{"role": "user", "content": "hi"}]}), 400)]
#[case::provider_failure(json!({"model": "public/model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}), 500)]
#[tokio::test]
async fn a_terminal_error_carries_no_decline(#[case] body: Value, #[case] status: u16) {
    let provider = provider().await;
    let response = support::post(
        support::app("anthropic/test-model", &provider.uri()),
        "/v1/chat/completions",
        body,
    )
    .await;

    assert_eq!(response.status(), status);
    assert!(response.extensions().get::<Decline>().is_none());
}
