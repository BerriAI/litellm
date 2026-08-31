//! `/spend/logs` endpoint - retrieve spend logs with filtering.

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
struct SpendLogsParams {
    start_time: Option<String>,
    end_time: Option<String>,
    user_id: Option<String>,
    team_id: Option<String>,
    model: Option<String>,
    limit: Option<i64>,
    offset: Option<i64>,
}

/// This route's contribution to the app router.
pub fn router() -> Router<AppState> {
    Router::new().route("/spend/logs", get(handle_spend_logs))
}

async fn handle_spend_logs(
    _auth: RequireMasterKey,
    State(state): State<AppState>,
    Query(params): Query<SpendLogsParams>,
) -> Response {
    let limit = params.limit.unwrap_or(100).min(1000);
    let offset = params.offset.unwrap_or(0);

    // Try to get spend logs from database
    if let Some(ref postgres) = state.postgres {
        match postgres
            .get_spend_logs(
                params.start_time.as_deref(),
                params.end_time.as_deref(),
                params.user_id.as_deref(),
                params.team_id.as_deref(),
                params.model.as_deref(),
                limit,
                offset,
            )
            .await
        {
            Ok(logs) => {
                return (StatusCode::OK, axum::Json(json!({ "data": logs }))).into_response();
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

    // If no database, return empty list
    (StatusCode::OK, axum::Json(json!({ "data": [] }))).into_response()
}
