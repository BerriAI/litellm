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
