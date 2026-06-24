//! `GET /v1/realtime` (WebSocket).
//!
//! This file is the **axum surface**: `router()`, the handler, and the small
//! socket↔events adapter. The pure logic (no axum) lives in [`service`].

mod service;

use std::sync::Arc;

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
use crate::state::AppState;

/// This route's contribution to the app router.
pub fn router() -> Router<AppState> {
    Router::new().route("/v1/realtime", get(handle))
}

#[derive(Debug, Deserialize)]
struct RealtimeQuery {
    model: String,
}

fn require_realtime_billing_safe_auth(auth: &UserApiKeyAuth) -> Result<(), (StatusCode, String)> {
    if auth.is_proxy_admin() {
        return Ok(());
    }
    Err((
        StatusCode::FORBIDDEN,
        "non-admin virtual-key realtime is disabled until realtime usage is reported to the control plane".to_string(),
    ))
}

/// Auth runs via the `UserApiKeyAuth` extractor (master key → admin, otherwise
/// cache then the swappable authenticator). Non-admin virtual keys are rejected
/// until realtime usage is reported back to the proxy; otherwise callers could
/// consume upstream realtime spend without that spend being charged to their key
/// or team. We validate the model BEFORE the upgrade so failures are clean HTTP
/// (400/404), not a socket that opens then closes, then hand the socket to
/// `bridge`.
async fn handle(
    auth: UserApiKeyAuth,
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    Query(query): Query<RealtimeQuery>,
) -> Result<Response, (StatusCode, String)> {
    require_realtime_billing_safe_auth(&auth)?;

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
    let model = query.model;
    Ok(ws.on_upgrade(move |socket| bridge(socket, router, pool, model)))
}

/// Adapt the axum socket (text frames) to the typed-event `Stream`/`Sink` the
/// service wants, keeping axum types out of `service`.
async fn bridge(
    socket: WebSocket,
    router: Arc<ModelRouter>,
    pool: Arc<RealtimePool>,
    model: String,
) {
    let (ws_sink, ws_stream) = socket.split();

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
    let _ = service::run(&router, &pool, &model, None, client_in, client_out).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn realtime_allows_proxy_admin_identity() {
        assert!(require_realtime_billing_safe_auth(&UserApiKeyAuth::admin()).is_ok());
    }

    #[test]
    fn realtime_rejects_non_admin_virtual_keys_until_spend_is_reported() {
        let err = require_realtime_billing_safe_auth(&UserApiKeyAuth {
            user_role: Some("internal_user".to_string()),
            ..UserApiKeyAuth::default()
        })
        .unwrap_err();

        assert_eq!(err.0, StatusCode::FORBIDDEN);
        assert!(err.1.contains("usage is reported"));
    }
}
