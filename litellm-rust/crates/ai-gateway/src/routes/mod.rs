//! HTTP routes.
//!
//! **Template:** every route module exposes `pub fn router() -> Router<AppState>`
//! that mounts its own paths; [`app`] merges them. A trivial route is a single
//! file (`health.rs`, `gil.rs`); a non-trivial one is a folder (`realtime/`) with
//! `handler` (entry) + `service` (logic) + `transport` (adapters). See AGENTS.md.

pub mod gil;
pub mod health;
pub mod realtime;
pub mod responses;

use axum::Router;

use crate::state::AppState;

/// Assemble the application router by merging every route module's `router()`.
pub fn app(state: AppState) -> Router {
    Router::new()
        .merge(health::router())
        .merge(gil::router())
        .merge(realtime::router())
        .merge(responses::router())
        .with_state(state)
}
