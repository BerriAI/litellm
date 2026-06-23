use axum::http::StatusCode;

/// Liveness probe — the process is up.
pub async fn liveness() -> StatusCode {
    StatusCode::OK
}

/// Readiness probe — the server is ready to accept traffic.
pub async fn readiness() -> StatusCode {
    StatusCode::OK
}
