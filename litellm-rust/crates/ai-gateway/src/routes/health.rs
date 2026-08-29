//! Health probes. Simple-route template: a `router()` plus its handlers, in one file.

use axum::Router;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::get;
use serde_json::json;

use crate::state::AppState;

/// This route's contribution to the app router.
pub fn router() -> Router<AppState> {
    Router::new()
        .route("/health/liveness", get(liveness))
        .route("/health/readiness", get(readiness))
        .route("/health/deep", get(deep))
}

/// The process is up.
async fn liveness() -> StatusCode {
    StatusCode::OK
}

/// The server is ready to accept traffic.
async fn readiness() -> StatusCode {
    StatusCode::OK
}

/// Deep health check: verifies connectivity to Redis and PostgreSQL.
///
/// Returns 200 if all configured backends are reachable, 503 otherwise.
/// Response body includes per-component status.
async fn deep(State(state): State<AppState>) -> impl IntoResponse {
    let mut checks = serde_json::Map::new();
    let mut all_healthy = true;

    // Check Redis
    if let Some(ref redis) = state.redis {
        match redis.ping().await {
            Ok(()) => {
                checks.insert("redis".to_string(), json!("ok"));
            }
            Err(e) => {
                checks.insert("redis".to_string(), json!({ "error": e.to_string() }));
                all_healthy = false;
            }
        }
    } else {
        checks.insert("redis".to_string(), json!("not configured"));
    }

    // Check PostgreSQL
    if let Some(ref postgres) = state.postgres {
        match postgres.ping().await {
            Ok(()) => {
                checks.insert("postgres".to_string(), json!("ok"));
            }
            Err(e) => {
                checks.insert("postgres".to_string(), json!({ "error": e.to_string() }));
                all_healthy = false;
            }
        }
    } else {
        checks.insert("postgres".to_string(), json!("not configured"));
    }

    // Check router has deployments
    let deployment_count = state.router.deployments().len();
    checks.insert(
        "router".to_string(),
        json!({ "deployments": deployment_count }),
    );
    if deployment_count == 0 {
        all_healthy = false;
    }

    let body = json!({
        "status": if all_healthy { "healthy" } else { "unhealthy" },
        "checks": checks,
    });

    let status = if all_healthy {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    (status, axum::Json(body))
}
