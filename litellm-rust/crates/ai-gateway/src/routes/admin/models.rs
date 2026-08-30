//! `/v1/models` endpoint - list available models.

use axum::Router;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use serde_json::json;

use crate::auth::key_auth::RequireValidKey;
use crate::state::AppState;

/// This route's contribution to the app router.
pub fn router() -> Router<AppState> {
    Router::new().route("/v1/models", get(handle_list_models))
}

async fn handle_list_models(_auth: RequireValidKey, State(state): State<AppState>) -> Response {
    let deployments = state.router.deployments();

    let models: Vec<serde_json::Value> = deployments
        .iter()
        .map(|d| {
            json!({
                "id": d.model_name,
                "object": "model",
                "created": 0,
                "owned_by": d.litellm_params.model.split('/').next().unwrap_or("unknown"),
                "litellm_params": {
                    "model": d.litellm_params.model,
                    "api_base": d.litellm_params.api_base,
                }
            })
        })
        .collect();

    let response = json!({
        "object": "list",
        "data": models,
    });

    (StatusCode::OK, axum::Json(response)).into_response()
}
