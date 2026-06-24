//! `GET /v1/realtime` (WebSocket).
//!
//! Template for a non-trivial route (see `routes/AGENTS.md`):
//! - [`handler`] — the thin HTTP/WS entry point (auth, validate, hand off).
//! - [`service`] — the business logic (select a deployment, call the provider).
//! - [`transport`] — adapts the axum socket to the typed events `service` wants.

mod handler;
mod service;
mod transport;

use axum::routing::get;
use axum::Router;

use crate::state::AppState;

/// This route's contribution to the app router.
pub fn router() -> Router<AppState> {
    Router::new().route("/v1/realtime", get(handler::handle))
}
