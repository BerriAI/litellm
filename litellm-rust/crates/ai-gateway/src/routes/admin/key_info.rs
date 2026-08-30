//! `/key/info` endpoint - retrieve key information.

use axum::Router;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use serde::Deserialize;
use serde_json::json;

use crate::auth::key_auth::RequireValidKey;
use crate::state::AppState;

#[derive(Deserialize)]
struct KeyInfoParams {
    key: Option<String>,
}

/// This route's contribution to the app router.
pub fn router() -> Router<AppState> {
    Router::new().route("/key/info", get(handle_key_info))
}

async fn handle_key_info(
    auth: RequireValidKey,
    State(state): State<AppState>,
    Query(params): Query<KeyInfoParams>,
) -> Response {
    let hashed_token = params
        .key
        .as_deref()
        .unwrap_or(auth.hashed_token.as_hex_str());

    // Try to get key info from database
    if let Some(ref postgres) = state.postgres {
        match postgres.get_key_by_hashed_token(hashed_token).await {
            Ok(Some(key_data)) => {
                return (StatusCode::OK, axum::Json(key_data)).into_response();
            }
            Ok(None) => {
                return (
                    StatusCode::NOT_FOUND,
                    axum::Json(json!({
                        "error": {
                            "message": "Key not found",
                            "type": "not_found",
                        }
                    })),
                )
                    .into_response();
            }
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    axum::Json(json!({
                        "error": {
                            "message": format!("Database error: {}", e),
                            "type": "database_error",
                        }
                    })),
                )
                    .into_response();
            }
        }
    }

    // If no database, return basic info from the auth object
    let key_info = json!({
        "token": hashed_token,
        "key_name": auth.key_object.key_name,
        "key_alias": auth.key_object.key_alias,
        "user_id": auth.key_object.user_id,
        "team_id": auth.key_object.team_id,
        "org_id": auth.key_object.org_id,
        "max_budget": auth.key_object.max_budget,
        "spend": auth.key_object.spend,
        "models": auth.key_object.models,
        "tpm_limit": auth.key_object.tpm_limit,
        "rpm_limit": auth.key_object.rpm_limit,
    });

    (StatusCode::OK, axum::Json(key_info)).into_response()
}
