//! `/user/info` endpoint - retrieve user information.

use axum::Router;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use serde::Deserialize;
use serde_json::json;

use crate::auth::key_auth::RequireMasterKey;
use crate::state::AppState;

#[derive(Deserialize)]
struct UserInfoParams {
    user_id: String,
}

/// This route's contribution to the app router.
pub fn router() -> Router<AppState> {
    Router::new().route("/user/info", get(handle_user_info))
}

async fn handle_user_info(
    _auth: RequireMasterKey,
    State(state): State<AppState>,
    Query(params): Query<UserInfoParams>,
) -> Response {
    // Try to get user info from database
    if let Some(ref postgres) = state.postgres {
        match postgres.get_user_by_id(&params.user_id).await {
            Ok(Some(user_data)) => {
                return (StatusCode::OK, axum::Json(user_data)).into_response();
            }
            Ok(None) => {
                return (
                    StatusCode::NOT_FOUND,
                    axum::Json(json!({
                        "error": {
                            "message": "User not found",
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

    // If no database, return error
    (
        StatusCode::NOT_IMPLEMENTED,
        axum::Json(json!({
            "error": {
                "message": "Database not configured",
                "type": "not_implemented",
            }
        })),
    )
        .into_response()
}
