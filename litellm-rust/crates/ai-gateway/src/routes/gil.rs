use axum::Json;
use serde::Serialize;

use crate::gil;

/// Pollable GIL-activity status. `gil_acquired_last_30s` should be `false` during
/// steady realtime traffic — the GIL is only taken at load time (config read).
#[derive(Debug, Serialize)]
pub struct GilStatusResponse {
    pub gil_acquired_last_30s: bool,
    pub total_acquisitions: u64,
    pub seconds_since_last: Option<u64>,
}

/// `GET /health/gil` — report recent GIL activity for polling.
pub async fn status() -> Json<GilStatusResponse> {
    let snapshot = gil::snapshot();
    Json(GilStatusResponse {
        gil_acquired_last_30s: snapshot.acquired_last_30s,
        total_acquisitions: snapshot.total_acquisitions,
        seconds_since_last: snapshot.seconds_since_last,
    })
}
