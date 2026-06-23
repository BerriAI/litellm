use axum::extract::State;
use axum::http::header::AUTHORIZATION;
use axum::http::{HeaderMap, StatusCode};
use axum::Json;
use litellm_core::error::CoreError;
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

/// `POST /v1/realtime` — authenticate, select a deployment via the router, and
/// invoke the realtime route end to end, returning the collected backend events.
pub async fn invoke(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<RealtimeRequest>,
) -> Result<Json<RealtimeResponse>, (StatusCode, String)> {
    authorize(&state, &headers)?;

    let events = crate::dispatch::realtime(&state.router, &request.model, request.input, None)
        .await
        .map_err(|err| {
            // `Routing` means the model has no deployment — a client-side error
            // (bad model name / misconfig). Everything else is an upstream or
            // transport failure.
            let status = match &err {
                CoreError::Routing(_) => StatusCode::NOT_FOUND,
                _ => StatusCode::BAD_GATEWAY,
            };
            (status, err.to_string())
        })?;
    Ok(Json(RealtimeResponse { events }))
}

/// Require a bearer token matching the configured gateway key. Fails closed when
/// no key is configured.
///
/// This is an **interim** guard so the route is never an open, unauthenticated
/// provider proxy. Full per-key auth, budgets, and rate limits are delegated to
/// the Python proxy in a later phase (pull-from-Python).
fn authorize(state: &AppState, headers: &HeaderMap) -> Result<(), (StatusCode, String)> {
    let Some(expected) = state.gateway_key.as_deref() else {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "gateway auth not configured (set LITELLM_GATEWAY_KEY)".to_string(),
        ));
    };
    let provided = headers
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .map(str::trim);
    match provided {
        Some(token) if token == expected => Ok(()),
        _ => Err((
            StatusCode::UNAUTHORIZED,
            "missing or invalid bearer token".to_string(),
        )),
    }
}
