//! `GET /v1/realtime` (WebSocket).
//!
//! This file is the **axum surface**: `router()`, the handler, and the small
//! socket↔events adapter. The pure logic (no axum) lives in [`service`]. Auth is
//! the `RequireMasterKey` extractor, so the handler stays thin.

mod service;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::io::realtime_pool::RealtimePool;
use axum::Router;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::Response;
use axum::routing::get;
use futures_util::{SinkExt, StreamExt};
use litellm_core::realtime::types::RealtimeEvent;
use litellm_core::router::Router as ModelRouter;
use serde::Deserialize;

use crate::auth::RequireMasterKey;
use crate::state::AppState;
use litellm_core::integrations::custom_logger::CustomLogger;
use litellm_core::integrations::types::RequestMetadata;

/// Process-local monotonic counter, mixed into the per-session call id so two
/// sessions opened in the same nanosecond still get distinct ids.
static CALL_SEQ: AtomicU64 = AtomicU64::new(0);

/// Generate a per-connection `litellm_call_id`. No external uuid dep: epoch
/// nanos + a process-local sequence is unique enough for log correlation.
fn new_call_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let seq = CALL_SEQ.fetch_add(1, Ordering::Relaxed);
    format!("rt-{nanos:x}-{seq:x}")
}

/// This route's contribution to the app router.
pub fn router() -> Router<AppState> {
    Router::new().route("/v1/realtime", get(handle))
}

#[derive(Debug, Deserialize)]
struct RealtimeQuery {
    model: String,
}

/// Auth runs via the `RequireMasterKey` extractor. We validate the model BEFORE
/// the upgrade so failures are clean HTTP (400/404), not a socket that opens then
/// closes, then hand the socket to `bridge`.
async fn handle(
    _auth: RequireMasterKey,
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    Query(query): Query<RealtimeQuery>,
) -> Result<Response, (StatusCode, String)> {
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
    let pool = state.realtime_pool.clone();
    let loggers = state.loggers.clone();
    let master_key = state.master_key.clone();
    let model = query.model;
    Ok(ws.on_upgrade(move |socket| bridge(socket, router, pool, loggers, master_key, model)))
}

async fn bridge(
    socket: WebSocket,
    router: Arc<ModelRouter>,
    pool: Arc<RealtimePool>,
    loggers: Arc<Vec<Arc<dyn CustomLogger>>>,
    master_key: Option<Arc<str>>,
    model: String,
) {
    let (ws_sink, ws_stream) = socket.split();

    let metadata = RequestMetadata {
        user_api_key_hash: master_key.as_deref().map(crate::auth::hash_token),
        ..RequestMetadata::default()
    };

    let client_in = ws_stream.filter_map(|message| async move {
        match message {
            Ok(Message::Text(text)) => serde_json::from_str::<RealtimeEvent>(&text).ok(),
            _ => None,
        }
    });
    let client_out = ws_sink.with(|event: RealtimeEvent| async move {
        Ok::<Message, axum::Error>(Message::Text(
            serde_json::to_string(&event).unwrap_or_default(),
        ))
    });

    futures_util::pin_mut!(client_in, client_out);

    let _ = service::run(
        &router,
        &pool,
        &model,
        None,
        loggers,
        new_call_id(),
        metadata,
        client_in,
        client_out,
    )
    .await;
}
