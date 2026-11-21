mod support;

use rstest::rstest;
use serde_json::{Value, json};
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

fn prompt() -> Value {
    json!({
        "model": "public/model",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hi"}],
    })
}

#[tokio::test]
async fn a_deployment_past_its_rpd_answers_429_until_its_daily_window_starts_over() {
    let upstream = MockServer::start().await;
    let app = support::app_with_router_settings(
        "anthropic/test-model",
        &upstream.uri(),
        json!({"rpm": 5, "rpd": 0}),
        json!({"optional_pre_call_checks": ["enforce_model_rate_limits"]}),
    );

    let refused = support::post(app, "/v1/messages", prompt()).await;

    let status = refused.status().as_u16();
    let retry_after = refused
        .headers()
        .get("retry-after")
        .and_then(|value| value.to_str().ok())
        .and_then(|seconds| seconds.parse::<u64>().ok());
    let body = support::json(refused).await;
    assert_eq!(
        (
            status,
            retry_after.map(|seconds| seconds <= 86_400),
            body.pointer("/error/message").and_then(Value::as_str),
            upstream.received_requests().await.unwrap().len(),
        ),
        (
            429,
            Some(true),
            Some("litellm.RateLimitError: Model rate limit exceeded. RPD limit=0, current usage=1"),
            0,
        )
    );
}

/// Python's proxy answers this refusal with the same message and no `retry-after`: only
/// `llm_provider-retry-after: 60` on chat, and nothing on `/v1/messages`.
#[rstest]
#[case::messages("/v1/messages")]
#[tokio::test]
async fn a_deployment_past_its_rpm_answers_429_with_retry_after_and_no_provider_call(
    #[case] path: &str,
) {
    let upstream = MockServer::start().await;
    let app = support::app_with_router_settings(
        "anthropic/test-model",
        &upstream.uri(),
        json!({"rpm": 0}),
        json!({"optional_pre_call_checks": ["enforce_model_rate_limits"]}),
    );

    let refused = support::post(app, path, prompt()).await;

    let status = refused.status().as_u16();
    let retry_after = refused.headers().get("retry-after").cloned();
    let body = support::json(refused).await;
    assert_eq!(
        (
            status,
            retry_after.as_ref().and_then(|value| value.to_str().ok()),
            body.pointer("/error/message").and_then(Value::as_str),
            upstream.received_requests().await.unwrap().len(),
        ),
        (
            429,
            Some("60"),
            Some("litellm.RateLimitError: Model rate limit exceeded. RPM limit=0, current usage=1"),
            0,
        )
    );
}

#[tokio::test]
async fn the_cache_tokens_of_a_messages_answer_count_toward_the_deployments_tpm() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_test", "type": "message", "role": "assistant", "model": "test-model",
            "content": [{"type": "text", "text": "hello"}], "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10, "output_tokens": 10,
                "cache_creation_input_tokens": 600, "cache_read_input_tokens": 400,
            },
        })))
        .mount(&upstream)
        .await;
    let app = support::app_with_router_settings(
        "anthropic/test-model",
        &upstream.uri(),
        json!({"tpm": 1000}),
        json!({"optional_pre_call_checks": ["enforce_model_rate_limits"]}),
    );

    let served = support::post(app.clone(), "/v1/messages", prompt()).await;
    let refused = support::post(app, "/v1/messages", prompt()).await;

    let statuses = (served.status().as_u16(), refused.status().as_u16());
    let body = support::json(refused).await;
    assert_eq!(
        (
            statuses,
            body.pointer("/error/message").and_then(Value::as_str),
        ),
        (
            (200, 429),
            Some(
                "litellm.RateLimitError: Model rate limit exceeded. TPM limit=1000, current usage=1020"
            ),
        )
    );
}
