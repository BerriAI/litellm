//! HTTP/WS entry point. Thin: authenticate, validate input, hand off. No
//! business logic or transport details here.

use axum::extract::ws::WebSocketUpgrade;
use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use serde::Deserialize;

use crate::auth;
use crate::state::AppState;

#[derive(Debug, Deserialize)]
pub struct RealtimeQuery {
    pub model: String,
}

/// `GET /v1/realtime`. Authenticate and validate the model BEFORE the WebSocket
/// upgrade, so failures are clean HTTP responses (401 / 400 / 404) rather than a
/// socket that opens then immediately closes. Then hand the socket to transport.
pub async fn handle(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<RealtimeQuery>,
) -> Result<Response, (StatusCode, String)> {
    auth::authorize(&state, &headers)?;

    if query.model.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "missing 'model' query param".to_string(),
        ));
    }
    if !state.router.has_deployment(&query.model) {
        return Err((
            StatusCode::NOT_FOUND,
            format!("no deployment for model '{}'", query.model),
        ));
    }

    let router = state.router.clone();
    let model = query.model;
    Ok(ws.on_upgrade(move |socket| super::transport::bridge(socket, router, model)))
}
