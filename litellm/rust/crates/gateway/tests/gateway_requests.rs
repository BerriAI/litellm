//! The requests the gateway serves reach the daily gateway request tables: counted by the request
//! volume middleware around the real routes and written by the flush on shutdown.

use std::sync::Arc;

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use litellm_config::{Config, Environment};
use litellm_db_testing::MigratedPostgres;
use litellm_gateway::{AppState, middleware};
use litellm_license::Entitlements;
use rstest::rstest;
use serde_json::json;
use tower::ServiceExt;
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

async fn upstream() -> MockServer {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-unlisted-9",
            "content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        })))
        .mount(&upstream)
        .await;
    upstream
}

fn messages(model: &str) -> Request<Body> {
    Request::post("/v1/messages")
        .header("authorization", "Bearer gateway-key")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"model": model, "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]})
                .to_string(),
        ))
        .unwrap()
}

#[rstest]
#[tokio::test]
async fn served_requests_are_written_by_route_and_status_on_shutdown() {
    litellm_gateway::install_crypto_provider();
    let postgres = MigratedPostgres::start().await.unwrap();
    let upstream = upstream().await;
    let config = Config::from_yaml(&format!(
        "model_list:\n  - model_name: claude\n    litellm_params:\n      model: anthropic/claude-unlisted-9\n      api_base: {}\n      api_key: test-key\ngeneral_settings:\n  master_key: gateway-key\n  database_url: {}\n",
        upstream.uri(),
        postgres.url(),
    ))
    .unwrap();
    let state = AppState::build(&config, &Environment::default())
        .await
        .unwrap();
    let requests = state.gateway_requests.clone();
    let app = middleware::global(
        litellm_gateway::router(state.clone(), &config),
        &middleware::Settings::default(),
        Arc::new(Entitlements::Community),
        &middleware::InFlight::new(),
        requests,
    );

    let statuses = [
        app.clone()
            .oneshot(messages("claude"))
            .await
            .unwrap()
            .status(),
        app.clone()
            .oneshot(messages("claude"))
            .await
            .unwrap()
            .status(),
        app.oneshot(messages("unconfigured"))
            .await
            .unwrap()
            .status(),
    ];
    state.close().await;

    assert_eq!(
        statuses,
        [StatusCode::OK, StatusCode::OK, StatusCode::BAD_REQUEST]
    );
    #[expect(
        clippy::disallowed_methods,
        reason = "Fixture reads are outside the compile-checked query set"
    )]
    let totals: Vec<(String, String, i64, i64)> = sqlx::query_as(
        r#"SELECT category, route, successful_requests, failed_requests FROM "LiteLLM_DailyGatewayRequests""#,
    )
    .fetch_all(postgres.pool())
    .await
    .unwrap();
    #[expect(
        clippy::disallowed_methods,
        reason = "Fixture reads are outside the compile-checked query set"
    )]
    let failures: Vec<(String, i32, i64)> = sqlx::query_as(
        r#"SELECT route, status_code, failed_requests FROM "LiteLLM_DailyGatewayFailedRequests""#,
    )
    .fetch_all(postgres.pool())
    .await
    .unwrap();
    assert_eq!(
        (totals, failures),
        (
            vec![("llm".to_owned(), "/v1/messages".to_owned(), 2, 1)],
            vec![("/v1/messages".to_owned(), 400, 1)]
        )
    );
}
