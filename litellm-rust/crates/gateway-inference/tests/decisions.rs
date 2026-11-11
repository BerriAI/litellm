mod support;

use std::time::Duration;

use axum::http::StatusCode;
use litellm_gateway_inference::Deployment;
use rstest::rstest;
use serde_json::{Value, json};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_json, header, method, path},
};

const ANSWERS: &str = r#"{"model":"gpt-6-luna","answers":[{"type":"choice","choice":"high","probabilities":[{"value":"high","probability":0.8},{"value":"low","probability":0.2}],"confidence":0.8},{"type":"score","score":2,"probabilities":[],"confidence":0.7}],"usage":{"input_tokens":2000,"output_tokens":20,"total_tokens":2020}}"#;

fn questions() -> Value {
    json!([
        {"type": "choice", "name": "urgency", "instructions": "How urgent?",
         "choices": [{"value": "low"}, {"value": "high"}]},
        {"type": "predicate", "name": "refund", "instructions": ""},
    ])
}

fn request() -> Value {
    json!({"model": "public/model", "input": "refund order 4411", "questions": questions(),
        "metadata": {"tag": "x"}})
}

async fn upstream(
    route: &str,
    auth: (&'static str, &'static str),
    status: u16,
    body: &str,
) -> MockServer {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(route))
        .and(header(auth.0, auth.1))
        .and(body_json(json!({
            "model": "gpt-6-luna",
            "input": "refund order 4411",
            "questions": [
                {"type": "choice", "name": "urgency", "instructions": "How urgent?",
                 "choices": [{"value": "low"}, {"value": "high"}]},
                {"type": "predicate", "name": "refund", "instructions": "Is this true of the input?"},
            ],
        })))
        .respond_with(ResponseTemplate::new(status).set_body_raw(body, "application/json"))
        .expect(1..)
        .mount(&upstream)
        .await;
    upstream
}

#[rstest]
#[case::versioned("/v1/decisions")]
#[case::unversioned("/decisions")]
#[tokio::test]
async fn openai_answers_are_aligned_to_the_questions(#[case] route: &str) {
    let upstream = upstream(
        "/v1/decisions",
        ("authorization", "Bearer test-key"),
        200,
        ANSWERS,
    )
    .await;

    let response = support::post(
        support::app("openai/gpt-6-luna", &format!("{}/v1", upstream.uri())),
        route,
        request(),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        support::json(response).await,
        json!({
            "model": "gpt-6-luna",
            "answers": [
                {"type": "choice", "name": "urgency", "choice": "high", "confidence": 0.8,
                 "probabilities": [{"value": "low", "probability": 0.2}, {"value": "high", "probability": 0.8}]},
                {"type": "refusal", "name": "refund"},
            ],
            "usage": {"input_tokens": 2000, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens": 20, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 2020},
        })
    );
}

#[rstest]
#[tokio::test]
async fn azure_sends_its_key_in_api_key() {
    let upstream = upstream(
        "/openai/v1/decisions",
        ("api-key", "test-key"),
        200,
        ANSWERS,
    )
    .await;

    let response = support::post(
        support::app("azure/gpt-6-luna", &upstream.uri()),
        "/v1/decisions",
        request(),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
}

#[rstest]
#[tokio::test]
async fn an_upstream_error_keeps_its_status_and_message() {
    let upstream = upstream(
        "/v1/decisions",
        ("authorization", "Bearer test-key"),
        429,
        r#"{"error":{"message":"slow down","type":"rate_limit_error"}}"#,
    )
    .await;

    let response = support::post(
        support::app("openai/gpt-6-luna", &format!("{}/v1", upstream.uri())),
        "/v1/decisions",
        request(),
    )
    .await;

    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        support::json(response).await["error"]["message"],
        "slow down"
    );
}

#[rstest]
#[case::no_questions(json!({"model": "public/model", "input": "x", "questions": []}), "Invalid Decisions request: questions should have 1 to 128 items, got 0")]
#[case::extra_question_field(json!({"model": "public/model", "input": "x", "questions": [{"type": "predicate", "instructions": "q", "extra": 1}]}), "Invalid Decisions request: ")]
#[case::images_next_to_input(json!({"model": "public/model", "input": "x", "images": ["a"], "questions": [{"type": "predicate", "instructions": "q"}]}), "")]
#[case::box_unsupported(json!({"model": "public/model", "input": "x", "questions": [{"type": "box", "instructions": "the mug"}]}), "openai does not support parameters: [\"box questions\"]")]
#[case::unknown_model(json!({"model": "nope", "input": "x", "questions": [{"type": "predicate", "instructions": "q"}]}), "")]
#[tokio::test]
async fn bad_requests_are_400_in_the_openai_envelope(#[case] body: Value, #[case] prefix: &str) {
    let response = support::post(
        support::app("openai/gpt-6-luna", "http://127.0.0.1:9/v1"),
        "/v1/decisions",
        body,
    )
    .await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = support::json(response).await;
    assert_eq!(body["error"]["code"], 400);
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .starts_with(prefix),
        "{body}"
    );
}

#[rstest]
#[tokio::test]
async fn a_box_question_reaches_a_deployment_that_supports_it() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/decisions"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            r#"{"model":"m","answers":[{"type":"box","box":{"top":1,"bottom":2,"left":3,"right":4},"confidence":0.5}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}"#,
            "application/json",
        ))
        .expect(1)
        .mount(&upstream)
        .await;
    let app = support::app_with_deployment(Deployment {
        model: "openai/gpt-6-luna".into(),
        api_base: Some(format!("{}/v1", upstream.uri())),
        api_key: Some("test-key".into()),
        timeout: Some(Duration::from_secs(5)),
        supports_box_questions: Some(true),
        ..Default::default()
    });

    let response = support::post(
        app,
        "/v1/decisions",
        json!({"model": "public/model", "input": "x", "questions": [{"type": "box", "name": "mug", "instructions": "the mug"}]}),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        support::json(response).await["answers"][0],
        json!({"type": "box", "name": "mug", "box": {"top": 1.0, "bottom": 2.0, "left": 3.0, "right": 4.0}, "confidence": 0.5})
    );
}

#[rstest]
#[tokio::test]
async fn an_answer_missing_its_confidence_is_a_500() {
    let upstream = upstream(
        "/v1/decisions",
        ("authorization", "Bearer test-key"),
        200,
        r#"{"model":"m","answers":[{"type":"choice","choice":"high","probabilities":[]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}"#,
    )
    .await;

    let response = support::post(
        support::app("openai/gpt-6-luna", &format!("{}/v1", upstream.uri())),
        "/v1/decisions",
        request(),
    )
    .await;

    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body = support::json(response).await;
    assert!(
        body["error"]["message"].as_str().unwrap().starts_with(
            "OpenaiException - Decisions provider 'openai' returned an unexpected response"
        ),
        "{body}"
    );
}

#[rstest]
#[tokio::test]
async fn an_unported_provider_is_501() {
    let response = support::post(
        support::app("ollama/llama", "http://127.0.0.1:9"),
        "/v1/decisions",
        json!({"model": "public/model", "input": "x", "questions": [{"type": "predicate", "instructions": "q"}]}),
    )
    .await;

    assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
}
