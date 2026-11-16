mod support;

use rstest::rstest;
use serde_json::{Value, json};
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

async fn upstream() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"id": "resp_1", "model": "test-model", "output": []})),
        )
        .mount(&server)
        .await;
    server
}

fn prompt(words: usize) -> Value {
    json!({
        "model": "public/model",
        "instructions": "Answer briefly.",
        "input": [{"type": "message", "role": "user", "content": "word ".repeat(words)}],
    })
}

async fn reached(server: &MockServer) -> bool {
    !server.received_requests().await.unwrap().is_empty()
}

#[rstest]
#[case::fits_both(5, (true, true))]
#[case::fits_the_large_one(100, (false, true))]
#[tokio::test]
async fn a_prompt_reaches_only_the_deployments_whose_context_window_it_fits(
    #[case] words: usize,
    #[case] expected: (bool, bool),
) {
    let (small, large) = (upstream().await, upstream().await);
    let app = support::described_group_app(
        &[
            ("small", &small.uri(), json!({"max_input_tokens": 50})),
            ("large", &large.uri(), json!({"max_input_tokens": 1000})),
        ],
        json!({"enable_pre_call_checks": true}),
    );
    for _ in 0..30 {
        let response = support::post(app.clone(), "/v1/responses", prompt(words)).await;
        assert_eq!(response.status(), 200);
    }
    assert_eq!((reached(&small).await, reached(&large).await), expected);
}

#[tokio::test]
async fn a_prompt_no_deployment_fits_is_refused_before_any_provider_call() {
    let server = upstream().await;
    let app = support::described_group_app(
        &[("small", &server.uri(), json!({"max_input_tokens": 10}))],
        json!({"enable_pre_call_checks": true}),
    );
    let response = support::post(app, "/v1/responses", prompt(100)).await;
    assert_eq!(response.status(), 400);
    assert!(!reached(&server).await);
}

fn messages_prompt() -> Value {
    json!({
        "model": "public/model",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hi"}],
    })
}

/// Python's proxy answers this refusal with `litellm.RateLimitError: Model rate limit exceeded.
/// RPM limit=1, current usage=2`, and with no `retry-after`: only `llm_provider-retry-after: 60`
/// on chat, and nothing on `/v1/messages`.
#[rstest]
#[case::responses("/v1/responses", prompt(1))]
#[case::messages("/v1/messages", messages_prompt())]
#[tokio::test]
async fn a_deployment_past_its_rpm_answers_429_with_retry_after_without_a_provider_call(
    #[case] path: &str,
    #[case] body: Value,
) {
    let server = upstream().await;
    let app = support::described_group_app(
        &[("limited", &server.uri(), json!({"rpm": 1}))],
        json!({"optional_pre_call_checks": ["enforce_model_rate_limits"]}),
    );
    support::post(app.clone(), path, body.clone()).await;
    let calls_before = server.received_requests().await.unwrap().len();
    let refused = support::post(app, path, body).await;
    let status = refused.status().as_u16();
    let retry_after = refused.headers().get("retry-after").cloned();
    let error = support::json(refused).await;
    assert_eq!(
        (
            status,
            retry_after.as_ref().and_then(|value| value.to_str().ok()),
            error["error"]["message"].as_str(),
            server.received_requests().await.unwrap().len(),
        ),
        (
            429,
            Some("60"),
            Some("the attempt failed: the deployment call failed with RateLimitError"),
            calls_before,
        )
    );
}
