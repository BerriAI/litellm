mod support;

use std::time::Duration;

use axum::{body::Body, http::Request};
use litellm_gateway_inference::Deployment;
use rstest::rstest;
use serde_json::{Value, json};
use tower::ServiceExt;
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

const MESSAGE: &str = r#"{"id":"msg_1","type":"message","role":"assistant","model":"test-model","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}"#;
const COMPLETION: &str = r#"{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"gpt-6-astra","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}"#;

#[rstest]
#[case::anthropic("anthropic/test-model", "", MESSAGE, "stop_sequences")]
#[case::openai("openai/gpt-6-astra", "/v1", COMPLETION, "stop")]
#[tokio::test]
async fn chat_fills_what_the_request_leaves_out_from_the_deployment_litellm_params(
    #[case] model: &str,
    #[case] base_path: &str,
    #[case] answer: &str,
    #[case] stop: &str,
) {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(answer, "application/json"))
        .expect(1)
        .mount(&upstream)
        .await;
    let api_base = format!("{}{base_path}", upstream.uri());
    let deployment = Deployment {
        model: model.into(),
        api_base: Some(api_base.clone()),
        api_key: Some("test-key".into()),
        timeout: Some(Duration::from_secs(5)),
        litellm_params: serde_json::from_value(json!({
            "model": model,
            "api_base": api_base,
            "temperature": 0.2,
            "top_k": 5,
            "stop": ["END"],
            "extra_headers": {"x-team": "team-1"},
            "input_cost_per_token": 0.000001,
            "cache_control_injection_points": [{"location": "message", "role": "system"}],
        }))
        .unwrap(),
        ..Default::default()
    };
    let request = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"model": "public/model", "max_tokens": 16, "temperature": 0.9,
                "messages": [{"role": "user", "content": "hi"}]})
            .to_string(),
        ))
        .unwrap();

    let response = support::app_with_deployment(deployment)
        .oneshot(request)
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let sent = &upstream.received_requests().await.unwrap()[0];
    assert_eq!(sent.headers["x-team"], "team-1");
    let body: Value = serde_json::from_slice(&sent.body).unwrap();
    assert_eq!(body["temperature"], json!(0.9), "{body}");
    assert_eq!(body["top_k"], json!(5), "{body}");
    assert_eq!(body[stop], json!(["END"]), "{body}");
    for internal in [
        "extra_headers",
        "input_cost_per_token",
        "cache_control_injection_points",
    ] {
        assert!(body.get(internal).is_none(), "{body}");
    }
}

#[tokio::test]
async fn chat_marks_the_messages_a_deployment_injection_point_names() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(MESSAGE, "application/json"))
        .expect(1)
        .mount(&upstream)
        .await;
    let model = "anthropic/test-model";
    let deployment = Deployment {
        model: model.into(),
        api_base: Some(upstream.uri()),
        api_key: Some("test-key".into()),
        timeout: Some(Duration::from_secs(5)),
        litellm_params: serde_json::from_value(json!({
            "model": model,
            "cache_control_injection_points": [{"location": "message", "role": "system"}],
        }))
        .unwrap(),
        ..Default::default()
    };
    let request = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"model": "public/model", "max_tokens": 16, "messages": [
                {"role": "system", "content": "rules"},
                {"role": "user", "content": "hi"},
            ]})
            .to_string(),
        ))
        .unwrap();

    let response = support::app_with_deployment(deployment)
        .oneshot(request)
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let sent = &upstream.received_requests().await.unwrap()[0];
    let body: Value = serde_json::from_slice(&sent.body).unwrap();
    assert_eq!(
        body["system"],
        json!([{"type": "text", "text": "rules", "cache_control": {"type": "ephemeral"}}]),
        "{body}"
    );
    assert!(
        body.get("cache_control_injection_points").is_none(),
        "{body}"
    );
}
