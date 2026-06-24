//! HTTP routes. One module per route so adding a route is a local change here.

pub mod gil;
pub mod health;
pub mod realtime;

use axum::routing::get;
use axum::Router as AxumRouter;

use crate::state::AppState;

/// Assemble the HTTP router from the per-route handlers.
pub fn app(state: AppState) -> AxumRouter {
    AxumRouter::new()
        .route("/health/liveness", get(health::liveness))
        .route("/health/readiness", get(health::readiness))
        .route("/health/gil", get(gil::status))
        .route("/v1/realtime", get(realtime::handler))
        .with_state(state)
}
