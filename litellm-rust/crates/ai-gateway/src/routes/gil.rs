//! `GET /health/gil` — poll to confirm Python is only touched at load time.
//! Simple-route template: a `router()` plus its handler, in one file.

use axum::routing::get;
use axum::{Json, Router};
use serde::Serialize;

use crate::gil;
use crate::state::AppState;

/// This route's contribution to the app router.
pub fn router() -> Router<AppState> {
    Router::new().route("/health/gil", get(status))
}

#[derive(Debug, Serialize)]
struct GilStatusResponse {
    gil_acquired_last_30s: bool,
    total_acquisitions: u64,
    seconds_since_last: Option<u64>,
}

async fn status() -> Json<GilStatusResponse> {
    let snapshot = gil::snapshot();
    Json(GilStatusResponse {
        gil_acquired_last_30s: snapshot.acquired_last_30s,
        total_acquisitions: snapshot.total_acquisitions,
        seconds_since_last: snapshot.seconds_since_last,
    })
}
