//! HTTP routes.
//!
//! **Template:** every route module exposes `pub fn router() -> Router<AppState>`
//! that mounts its own paths; [`app`] merges them. A trivial route is a single
//! file (`health.rs`, `gil.rs`); a non-trivial one is a folder (`realtime/`) with
//! `handler` (entry) + `service` (logic) + `transport` (adapters). See AGENTS.md.

pub mod admin;
pub mod audio;
pub mod chat_completions;
pub mod embeddings;
pub mod gil;
pub mod health;
pub mod images;
pub mod messages;
pub mod realtime;
pub mod responses;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::extract::State;
use axum::http::StatusCode;
use axum::middleware::{from_fn, from_fn_with_state};
use axum::response::IntoResponse;
use axum::routing::get;
use tower::limit::ConcurrencyLimitLayer;
use tower_http::timeout::TimeoutLayer;

use crate::middleware::alerting::alerting_middleware_from_app_state;
use crate::middleware::cors::cors_middleware_default;
use crate::middleware::csrf::csrf_middleware_from_app_state;
use crate::middleware::security_headers::security_headers_middleware;
use crate::middleware::tracing::tracing_middleware;
use crate::middleware::validation::validation_middleware;
use crate::state::AppState;

/// Maximum request body size: 10MB. Prevents abuse from oversized payloads.
const MAX_BODY_SIZE: usize = 10 * 1024 * 1024;

/// Assemble the application router by merging every route module's `router()`.
pub fn app(state: AppState) -> Router {
    let max_concurrent: usize = std::env::var("MAX_CONCURRENT_REQUESTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1000);

    // Slow loris protection: timeout for entire request processing
    let request_timeout = std::env::var("REQUEST_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300u64);

    Router::new()
        .merge(health::router())
        .merge(gil::router())
        .merge(chat_completions::router())
        .merge(messages::router())
        .merge(embeddings::router())
        .merge(images::router())
        .merge(audio::router())
        .merge(admin::router())
        .merge(realtime::router())
        .merge(responses::router())
        .route("/metrics", get(metrics))
        .layer(from_fn_with_state(
            state.clone(),
            csrf_middleware_from_app_state,
        ))
        .layer(from_fn(cors_middleware_default))
        .layer(from_fn(validation_middleware))
        .layer(from_fn(security_headers_middleware))
        .layer(from_fn(tracing_middleware))
        .layer(from_fn_with_state(
            state.clone(),
            alerting_middleware_from_app_state,
        ))
        .layer(DefaultBodyLimit::max(MAX_BODY_SIZE))
        .layer(ConcurrencyLimitLayer::new(max_concurrent))
        .layer(TimeoutLayer::new(std::time::Duration::from_secs(
            request_timeout,
        )))
        .with_state(state)
}

pub async fn metrics(State(state): State<AppState>) -> impl IntoResponse {
    let body = state.metrics.render();
    (
        StatusCode::OK,
        [("content-type", "text/plain; charset=utf-8")],
        body,
    )
}
