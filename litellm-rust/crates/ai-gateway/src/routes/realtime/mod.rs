//! `GET /v1/realtime` (WebSocket).
//!
//! This file is the **axum surface**: `router()`, the handler, and the small
//! socket↔events adapter. The pure logic (no axum) lives in [`service`].

mod service;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::io::realtime_pool::RealtimePool;
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

use crate::auth::UserApiKeyAuth;
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

fn request_metadata_for_auth(auth: &UserApiKeyAuth, master_key: Option<&str>) -> RequestMetadata {
    let user_api_key_hash = auth.api_key.clone().or_else(|| {
        auth.is_proxy_admin()
            .then(|| master_key.map(crate::auth::hash_token))
            .flatten()
    });

    RequestMetadata {
        user_api_key_hash,
        user_api_key_user_id: auth.user_id.clone(),
        user_api_key_team_id: auth.team_id.clone(),
        user_api_key_budget_reservation: auth.budget_reservation.clone(),
    }
}

/// Auth runs via the `UserApiKeyAuth` extractor (master key → admin, otherwise
/// cache then the swappable authenticator). We validate the model BEFORE the
/// upgrade so failures are clean HTTP (400/404), not a socket that opens then
/// closes, then hand the socket to `bridge`.
async fn handle(
    auth: UserApiKeyAuth,
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
    Ok(ws.on_upgrade(move |socket| bridge(socket, router, pool, loggers, auth, master_key, model)))
}

/// Adapt the axum socket (text frames) to the typed-event `Stream`/`Sink` the
/// service wants, keeping axum types out of `service`.
///
/// This is also the realtime-logging seam: every upstream→client event (the
/// direction carrying `session.created` and `response.done` with usage) is fed
/// to a [`RealTimeStreaming`] collector via the splice's `observe` callback. The
/// observe is O(1) and never buffers frames. When the splice returns (any of the
/// three break paths — client disconnect, upstream close, idle timeout), we flush
/// one logging payload to the registered callbacks.
async fn bridge(
    socket: WebSocket,
    router: Arc<ModelRouter>,
    pool: Arc<RealtimePool>,
    loggers: Arc<Vec<Arc<dyn CustomLogger>>>,
    auth: UserApiKeyAuth,
    master_key: Option<Arc<str>>,
    model: String,
) {
    let (ws_sink, ws_stream) = socket.split();

    // Attribute the spend log to the key that authenticated this session. For a
    // virtual key, the Python verifier returns the already-hashed key + user/team
    // attribution. For the local master-key fast path, hash the master key here.
    // A non-null user_api_key_hash is required for the Python spend logger to
    // write a SpendLogs row.
    let metadata = request_metadata_for_auth(&auth, master_key.as_deref());

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
    // single post-session `log_messages` flush. `run` picks a pooled (warm) or
    // fresh upstream — observe fires on the upstream arm either way.
    let result = service::run(
        &router,
        &pool,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_metadata_uses_virtual_key_identity() {
        let metadata = request_metadata_for_auth(
            &UserApiKeyAuth {
                api_key: Some("hashed-key".to_string()),
                user_id: Some("user-1".to_string()),
                team_id: Some("team-1".to_string()),
                budget_reservation: Some(serde_json::json!({
                    "reserved_cost": 0.5,
                    "entries": [{"counter_key": "spend:key:hashed-key"}],
                    "finalized": false,
                    "input_cost": 0.1
                })),
                ..UserApiKeyAuth::default()
            },
            Some("sk-master"),
        );

        assert_eq!(metadata.user_api_key_hash.as_deref(), Some("hashed-key"));
        assert_eq!(metadata.user_api_key_user_id.as_deref(), Some("user-1"));
        assert_eq!(metadata.user_api_key_team_id.as_deref(), Some("team-1"));
        assert_eq!(
            metadata
                .user_api_key_budget_reservation
                .as_ref()
                .and_then(|reservation| reservation.get("reserved_cost"))
                .and_then(serde_json::Value::as_f64),
            Some(0.5)
        );
    }

    #[test]
    fn request_metadata_hashes_master_key_for_local_admin_fast_path() {
        let metadata = request_metadata_for_auth(&UserApiKeyAuth::admin(), Some("sk-master"));
        let expected_hash = crate::auth::hash_token("sk-master");

        assert_eq!(
            metadata.user_api_key_hash.as_deref(),
            Some(expected_hash.as_str())
        );
    }
}
