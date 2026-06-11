use std::time::{Duration, Instant};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use guardrails_core::{
    GuardrailInput, GuardrailStatus, InputType, ProviderConfig, RequestContext, Verdict,
};
use guardrails_providers::build;

use crate::state::AppState;

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/providers", get(list_providers))
        .route("/v1/guardrails/apply", post(apply_guardrail))
        .with_state(state)
}

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

async fn list_providers() -> impl IntoResponse {
    Json(serde_json::json!({
        "providers": [
            "generic_guardrail_api",
            "openai_moderation",
            "azure/prompt_shield",
            "azure/text_moderations",
            "presidio",
            "lakera_v2",
            "bedrock",
        ]
    }))
}

#[derive(Deserialize)]
struct ApplyRequest {
    #[allow(dead_code)]
    guardrail_name: String,
    config: ProviderConfig,
    input: GuardrailInput,
    input_type: InputType,
    #[serde(default)]
    context: RequestContext,
    #[serde(default)]
    timeout_ms: Option<u64>,
}

#[derive(Serialize)]
struct ApplyResponse {
    verdict: Verdict,
    provider_response: serde_json::Value,
    guardrail_status: GuardrailStatus,
    duration_ms: u64,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Serialize)]
struct ErrorDetail {
    code: String,
    message: String,
}

async fn apply_guardrail(
    State(state): State<AppState>,
    Json(req): Json<ApplyRequest>,
) -> impl IntoResponse {
    let provider = match build(req.config, &state.http) {
        Ok(p) => p,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(
                    serde_json::to_value(ErrorResponse {
                        error: ErrorDetail {
                            code: "invalid_config".to_owned(),
                            message: e.to_string(),
                        },
                    })
                    .unwrap(),
                ),
            );
        }
    };

    let timeout = Duration::from_millis(req.timeout_ms.unwrap_or(5000));
    let start = Instant::now();

    let result = tokio::time::timeout(
        timeout,
        provider.apply(&req.input, req.input_type, &req.context),
    )
    .await;

    let duration_ms = start.elapsed().as_millis() as u64;

    let response = match result {
        Ok(Ok(outcome)) => ApplyResponse {
            guardrail_status: outcome.status(),
            verdict: outcome.verdict,
            provider_response: outcome.provider_response,
            duration_ms: outcome.duration_ms,
        },
        Ok(Err(provider_err)) => ApplyResponse {
            guardrail_status: GuardrailStatus::GuardrailFailedToRespond,
            verdict: Verdict::Pass,
            provider_response: serde_json::to_value(&provider_err).unwrap_or_default(),
            duration_ms,
        },
        Err(_elapsed) => ApplyResponse {
            guardrail_status: GuardrailStatus::GuardrailFailedToRespond,
            verdict: Verdict::Pass,
            provider_response: serde_json::json!({"kind": "timeout", "ms": duration_ms}),
            duration_ms,
        },
    };

    (
        StatusCode::OK,
        Json(serde_json::to_value(response).unwrap()),
    )
}
