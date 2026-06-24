use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Query, State};
use axum::http::header::AUTHORIZATION;
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use futures_util::{SinkExt, StreamExt};
use litellm_core::realtime::types::RealtimeEvent;
use serde::Deserialize;
use subtle::ConstantTimeEq;

use crate::state::AppState;

#[derive(Debug, Deserialize)]
pub struct RealtimeQuery {
    pub model: String,
}

/// `GET /v1/realtime` (WebSocket). Authenticate BEFORE the upgrade so a bad key
/// is a clean 401 (not a WS close), then splice the client socket to the provider.
pub async fn handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<RealtimeQuery>,
) -> Result<Response, (StatusCode, String)> {
    authorize(&state, &headers)?;
    if query.model.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "missing 'model' query param".to_string(),
        ));
    }
    let router = state.router.clone();
    let model = query.model;
    Ok(ws.on_upgrade(move |socket| bridge(socket, router, model)))
}

/// Adapt the axum socket to typed events and run the splice.
async fn bridge(
    socket: WebSocket,
    router: std::sync::Arc<litellm_core::router::Router>,
    model: String,
) {
    let (ws_sink, ws_stream) = socket.split();

    // client text frames -> typed events (non-text / parse failures are skipped)
    let client_in = ws_stream.filter_map(|message| async move {
        match message {
            Ok(Message::Text(text)) => serde_json::from_str::<RealtimeEvent>(&text).ok(),
            _ => None,
        }
    });

    // typed events -> client text frames
    let client_out = ws_sink.with(|event: RealtimeEvent| async move {
        Ok::<Message, axum::Error>(Message::Text(
            serde_json::to_string(&event).unwrap_or_default(),
        ))
    });

    futures_util::pin_mut!(client_in, client_out);
    let _ = crate::dispatch::realtime(&router, &model, None, client_in, client_out).await;
}

/// Require a bearer token matching the configured gateway key. Fails closed when
/// no key is configured.
///
/// This is an **interim** guard so the route is never an open, unauthenticated
/// provider proxy. Full per-key auth, budgets, and rate limits are delegated to
/// the Python proxy in a later phase (pull-from-Python).
fn authorize(state: &AppState, headers: &HeaderMap) -> Result<(), (StatusCode, String)> {
    let Some(expected) = state.gateway_key.as_deref() else {
        // Permanent server misconfiguration (no key set), not a transient outage —
        // 500 so clients don't treat it as retryable like a 503.
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "gateway auth not configured (set LITELLM_GATEWAY_KEY)".to_string(),
        ));
    };
    let provided = headers
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .map(str::trim);
    match provided {
        // Constant-time compare to avoid leaking the key via timing.
        Some(token) if bool::from(token.as_bytes().ct_eq(expected.as_bytes())) => Ok(()),
        _ => Err((
            StatusCode::UNAUTHORIZED,
            "missing or invalid bearer token".to_string(),
        )),
    }
}
