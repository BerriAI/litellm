mod support;

use rstest::rstest;
use serde_json::{Value, json};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_partial_json, method},
};

#[rstest]
#[case("/chat/completions", Some("public/model"))]
#[case("/v1/chat/completions", Some("public/model"))]
#[case("/engines/public/model/chat/completions", None)]
#[case("/openai/deployments/public/model/chat/completions", None)]
#[case("/openai/deployments/unused/chat/completions", Some("public/model"))]
#[tokio::test]
async fn chat_aliases_call_core_and_use_the_body_model_before_the_path(
    #[case] route: &str,
    #[case] model: Option<&str>,
) {
    let upstream = MockServer::start().await;
    let messages = json!([{"role": "user", "content": "hi"}]);
    Mock::given(method("POST"))
        .and(body_partial_json(
            json!({"model": "test-model", "max_tokens": 16}),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_test", "model": "test-model", "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}
        })))
        .expect(1)
        .mount(&upstream)
        .await;
    let response = support::post(
        support::app("anthropic/test-model", &upstream.uri()),
        route,
        json!({"model": model, "messages": messages, "max_tokens": 16}),
    )
    .await;
    assert_eq!(response.status(), 200);
    assert_eq!(
        support::json(response).await["choices"][0]["message"]["content"],
        "hello"
    );
}

#[rstest]
#[case("/responses")]
#[case("/v1/responses")]
#[case("/embeddings")]
#[case("/v1/embeddings")]
#[case("/completions")]
#[case("/v1/completions")]
#[case("/engines/public/model/embeddings")]
#[case("/openai/deployments/public/model/completions")]
#[tokio::test]
async fn unimplemented_routes_return_an_explicit_error(#[case] path: &str) {
    let response = support::post(
        support::app("anthropic/test-model", "http://127.0.0.1:1"),
        path,
        json!({}),
    )
    .await;
    assert_eq!(response.status(), 501);
    assert!(
        support::json(response).await["error"]["message"]
            .as_str()
            .unwrap()
            .contains("not implemented")
    );
}

#[rstest]
#[case("/audio/transcriptions")]
#[case("/v1/audio/transcriptions")]
#[tokio::test]
async fn transcription_aliases_reach_core_validation(#[case] path: &str) {
    let response = support::post(
        support::app("bedrock/test-model", "http://127.0.0.1:1"),
        path,
        json!({"model": "public/model", "audio": {"data": "YWJj", "format": "invalid"}}),
    )
    .await;
    assert_eq!(response.status(), 400);
    let body: Value = support::json(response).await;
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("audio.format")
    );
}
