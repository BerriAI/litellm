use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use litellm_core::realtime::types::RealtimeEvent;
use serde::{Deserialize, Serialize};

use crate::state::AppState;

/// Request body for the minimal realtime invoke endpoint.
#[derive(Debug, Deserialize)]
pub struct RealtimeRequest {
    /// Public model alias to route on (e.g. `gpt-realtime`).
    pub model: String,
    /// Client events to send upstream, each a typed realtime event.
    pub input: Vec<RealtimeEvent>,
}

/// Response body: the backend events collected for this turn.
#[derive(Debug, Serialize)]
pub struct RealtimeResponse {
    pub events: Vec<RealtimeEvent>,
}

/// `POST /v1/realtime` — minimal: the router selects a deployment and invokes
/// the realtime route end to end, returning the collected backend events.
pub async fn invoke(
    State(state): State<AppState>,
    Json(request): Json<RealtimeRequest>,
) -> Result<Json<RealtimeResponse>, (StatusCode, String)> {
    let events = state
        .router
        .realtime(&request.model, request.input, None)
        .await
        .map_err(|err| (StatusCode::BAD_GATEWAY, err.to_string()))?;
    Ok(Json(RealtimeResponse { events }))
}
