//! A key's own budget, served natively while `general_settings.disable_budget_reservation` is on:
//! `admit` admits the key, and the read-time check refuses a call over its budget with Python's
//! budget error before anything reaches the provider. Nothing is priced while the reservation is
//! disabled, so a model group the cost map does not name is served and checked as paid.

mod keys;

use std::sync::Arc;

use axum::{
    body::{Body, to_bytes},
    http::Request,
};
use litellm_config::{Config, Environment};
use litellm_gateway::{Admit, AppState, StoredConfig};
use rstest::rstest;
use serde_json::{Value, json};
use tower::ServiceExt;
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

const KEY: &str = "sk-budgeted";
const PRICED: &str = "anthropic/claude-haiku-5-5";
const UNKNOWN: &str = "anthropic/claude-unlisted-9";

/// Who answered a request: Python, over the reason `admit` gave, or the native app.
#[derive(Debug, PartialEq)]
enum Answered {
    Python(String),
    Native(u16, Value),
}

async fn provider() -> MockServer {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
            "content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        })))
        .mount(&upstream)
        .await;
    upstream
}

/// What a call from [`KEY`], holding `columns`, to the model group named for `model` gets from a
/// gateway in front of Python, and how many calls reached the provider.
async fn answered(columns: Value, model: &str, reservation_disabled: bool) -> (Answered, usize) {
    litellm_gateway::install_crypto_provider();
    let held = keys::Held::holding(KEY, columns).await;
    let upstream = provider().await;
    let group = model.rsplit_once('/').map_or(model, |(_, name)| name);
    let config = Config::from_yaml(&format!(
        "model_list:\n  - model_name: {group}\n    litellm_params:\n      model: {model}\n      api_base: {}\n      api_key: provider-key\ngeneral_settings:\n  master_key: gateway-key\n  database_url: {}\n  disable_budget_reservation: {reservation_disabled}\n",
        upstream.uri(),
        held.url(),
    ))
    .unwrap();
    let state = AppState::build(&config, &Environment::default())
        .await
        .unwrap();
    let admit = Arc::new(Admit::new(&config, &state, StoredConfig::Unused));
    let request = Request::post("/v1/messages")
        .header("authorization", format!("Bearer {KEY}"))
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"model": group, "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]})
                .to_string(),
        ))
        .unwrap();

    let answered = match admit.admit(request).await {
        Err((_, reason)) => Answered::Python(<&str>::from(reason.why).to_owned()),
        Ok(admitted) => {
            let response = litellm_gateway::router(state.clone(), &config)
                .oneshot(admitted)
                .await
                .unwrap();
            let status = response.status().as_u16();
            let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
            Answered::Native(status, serde_json::from_slice(&body).unwrap())
        }
    };
    let sent = upstream.received_requests().await.unwrap_or_default().len();
    state.close().await;
    (answered, sent)
}

/// Python's `BudgetExceededError` for a key over its budget, in the error envelope Python's
/// `/v1/messages` answers it in.
fn over_budget(spend: &str) -> Value {
    json!({"type": "error", "error": {
        "type": "invalid_request_error",
        "message": format!("Budget has been exceeded! Key=key (sk-...-key) Current cost: {spend}, Max budget: 10.0"),
        "litellm_error_type": "budget_exceeded",
    }})
}

#[rstest]
#[case::at_its_budget(json!({"max_budget": 10.0, "spend": 10.0}), PRICED, "10.0")]
#[case::over_its_budget(json!({"max_budget": 10.0, "spend": 12.5}), PRICED, "12.5")]
#[case::over_a_budget_that_resets(
    json!({"max_budget": 10.0, "spend": 12.5, "budget_duration": "30d", "budget_reset_at": "2100-01-01T00:00:00"}),
    PRICED,
    "12.5"
)]
#[case::over_its_budget_on_a_model_the_cost_map_does_not_name(
    json!({"max_budget": 10.0, "spend": 12.5}),
    UNKNOWN,
    "12.5"
)]
#[tokio::test(flavor = "multi_thread")]
async fn a_key_at_or_over_its_budget_is_refused_natively_with_pythons_error(
    #[case] columns: Value,
    #[case] model: &str,
    #[case] spend: &str,
) {
    assert_eq!(
        answered(columns, model, true).await,
        (Answered::Native(422, over_budget(spend)), 0)
    );
}

#[rstest]
#[case::with_a_max_budget(json!({"max_budget": 10.0, "spend": 9.5}), PRICED)]
#[case::with_a_budget_that_resets(
    json!({"max_budget": 10.0, "spend": 9.5, "budget_duration": "30d", "budget_reset_at": "2100-01-01T00:00:00"}),
    PRICED
)]
#[case::on_a_model_the_cost_map_does_not_name(json!({"max_budget": 10.0, "spend": 9.5}), UNKNOWN)]
#[tokio::test(flavor = "multi_thread")]
async fn a_key_under_its_budget_is_served_natively(#[case] columns: Value, #[case] model: &str) {
    let (answered, sent) = answered(columns, model, true).await;

    assert_eq!(
        (matches!(answered, Answered::Native(200, _)), sent),
        (true, 1),
        "{answered:?}"
    );
}

#[rstest]
#[case::under_its_budget(json!({"max_budget": 10.0, "spend": 9.5}))]
#[case::over_its_budget(json!({"max_budget": 10.0, "spend": 12.5}))]
#[tokio::test(flavor = "multi_thread")]
async fn with_the_reservation_on_a_key_with_a_budget_goes_to_python(#[case] columns: Value) {
    assert_eq!(
        answered(columns, PRICED, false).await,
        (Answered::Python("credential".to_owned()), 0)
    );
}
