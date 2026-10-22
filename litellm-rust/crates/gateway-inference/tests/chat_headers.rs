mod support;

use axum::{body::Body, http::Request};
use rstest::rstest;
use serde_json::{Value, json};
use tower::ServiceExt;
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

const MESSAGE: &str = r#"{"id":"msg_1","type":"message","role":"assistant","model":"test-model","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}"#;
const COMPLETION: &str = r#"{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"gpt-6-astra","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}"#;

#[rstest]
#[case::anthropic("anthropic/test-model", "", MESSAGE, true)]
#[case::openai("openai/gpt-6-astra", "/v1", COMPLETION, false)]
#[tokio::test]
async fn chat_sends_the_body_extra_headers_and_the_anthropic_headers_its_provider_takes(
    #[case] model: &str,
    #[case] base_path: &str,
    #[case] answer: &str,
    #[case] anthropic: bool,
) {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(answer, "application/json"))
        .expect(1)
        .mount(&upstream)
        .await;
    let request = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("anthropic-beta", "test-feature")
        .header("anthropic-version", "test-version")
        .body(Body::from(
            json!({"model": "public/model", "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
                "extra_headers": {"x-trace": "trace-1"}})
            .to_string(),
        ))
        .unwrap();

    let response = support::app(model, &format!("{}{base_path}", upstream.uri()))
        .oneshot(request)
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let sent = &upstream.received_requests().await.unwrap()[0];
    assert_eq!(sent.headers["x-trace"], "trace-1");
    let body: Value = serde_json::from_slice(&sent.body).unwrap();
    assert!(body.get("extra_headers").is_none(), "{body}");
    let beta = sent
        .headers
        .get("anthropic-beta")
        .map(|value| value.to_str().unwrap().to_owned());
    match anthropic {
        true => {
            assert!(
                beta.unwrap()
                    .split(',')
                    .any(|beta| beta.trim() == "test-feature")
            );
            assert_eq!(sent.headers["anthropic-version"], "test-version");
        }
        false => {
            assert_eq!(beta, None);
            assert!(!sent.headers.contains_key("anthropic-version"));
        }
    }
}
