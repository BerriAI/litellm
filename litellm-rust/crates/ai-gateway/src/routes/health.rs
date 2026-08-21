//! Health probes. Simple-route template: a `router()` plus its handlers, in one file.

use axum::Router;
use axum::http::StatusCode;
use axum::routing::get;

use crate::state::AppState;

/// This route's contribution to the app router.
pub fn router() -> Router<AppState> {
    Router::new()
        .route("/health/liveness", get(liveness))
        .route("/health/readiness", get(readiness))
}

/// The process is up.
async fn liveness() -> StatusCode {
    StatusCode::OK
}

/// The server is ready to accept traffic.
async fn readiness() -> StatusCode {
    StatusCode::OK
}
