//! `GET /v1/realtime` (WebSocket).
//!
//! This file is the **axum surface**: `router()`, the handler, and the small
//! socket↔events adapter. The pure logic (no axum) lives in [`service`]. Auth is
//! the `RequireMasterKey` extractor, so the handler stays thin.

mod service;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::Response;
use axum::routing::get;
use axum::Router;
use futures_util::{SinkExt, StreamExt};
use litellm_core::realtime::types::RealtimeEvent;
use litellm_core::router::Router as ModelRouter;
use serde::Deserialize;

use crate::auth::RequireMasterKey;
use crate::integrations::custom_logger::CustomLogger;
use crate::integrations::types::RequestMetadata;
use crate::realtime::streaming::{RealTimeStreaming, SessionStatus};
use crate::state::AppState;

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
    let loggers = state.loggers.clone();
    let master_key = state.master_key.clone();
    let model = query.model;
    Ok(ws.on_upgrade(move |socket| bridge(socket, router, loggers, master_key, model)))
}

/// Adapt the axum socket (text frames) to the typed-event `Stream`/`Sink` the
/// service wants, keeping axum types out of `service`.
///
/// This is also the realtime-logging seam: every upstream→client event (the
/// direction carrying `session.created` and `response.done` with usage) is fed
/// to a [`RealTimeStreaming`] collector as it passes through `client_out`. The
/// observe is O(1) and never buffers frames. When the splice returns (any of the
/// three break paths — client disconnect, upstream close, idle timeout), we flush
/// one logging payload to the registered callbacks.
async fn bridge(
    socket: WebSocket,
    router: Arc<ModelRouter>,
    loggers: Arc<Vec<Arc<dyn CustomLogger>>>,
    master_key: Option<Arc<str>>,
    model: String,
) {
    let (ws_sink, ws_stream) = socket.split();

    // Attribute the spend log to the key that authenticated this session (the
    // master key — the gateway is master-key auth). A non-null user_api_key_hash
    // is required for the Python spend logger to write a SpendLogs row.
    let metadata = RequestMetadata {
        user_api_key_hash: master_key.as_deref().map(str::to_string),
        ..RequestMetadata::default()
    };

    // Owned by THIS task only. The splice observes it via a synchronous `&mut`
    // callback (below), so there is no Arc/Mutex/atomic on the per-frame hot
    // path — just a monomorphized FnMut mutating stack-local fields. This is
    // what lets observe scale: 10K concurrent sessions = 10K independent
    // collectors, zero cross-task synchronization.
    let mut collector = RealTimeStreaming::new(
        loggers.as_ref().clone(),
        new_call_id(),
        model.clone(),
        metadata,
    );

    let client_in = ws_stream.filter_map(|message| async move {
        match message {
            Ok(Message::Text(text)) => serde_json::from_str::<RealtimeEvent>(&text).ok(),
            _ => None,
        }
    });
    // Plain forwarding sink — no observe here anymore.
    let client_out = ws_sink.with(|event: RealtimeEvent| async move {
        Ok::<Message, axum::Error>(Message::Text(
            serde_json::to_string(&event).unwrap_or_default(),
        ))
    });

    futures_util::pin_mut!(client_in, client_out);

    // The observe closure borrows `&mut collector` for the duration of the
    // splice; the borrow ends when `run` returns, freeing the collector for the
    // single post-session `log_messages` flush.
    let result = service::run(
        &router,
        &model,
        None,
        |event: &RealtimeEvent| collector.observe(event),
        client_in,
        client_out,
    )
    .await;

    let status = if result.is_ok() {
        SessionStatus::Success
    } else {
        SessionStatus::Failure
    };
    collector.log_messages(status);
}
