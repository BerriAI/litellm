//! Unauthenticated load balancer probes, with the Python proxy's paths and bodies.
//!
//! Readiness answers 503 `{"status": "shutting_down"}` once draining starts, as Python's
//! `GracefulShutdownManager` makes it. Liveness keeps answering 200 while draining, unlike
//! Python: a drain can last many minutes, and a failing liveness probe would get the process
//! restarted before its open streams finish.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
};
use serde_json::json;

/// Set once shutdown starts; shared by the probes, the metrics and the drain loop.
#[derive(Clone, Default)]
pub struct Draining(Arc<AtomicBool>);

impl Draining {
    pub fn start(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    pub fn is_draining(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }

    pub fn flag(&self) -> Arc<AtomicBool> {
        self.0.clone()
    }
}

pub fn router(draining: Draining) -> Router {
    Router::new()
        .route("/health/liveliness", get(liveliness))
        .route("/health/liveness", get(liveliness))
        .route("/health/readiness", get(readiness))
        .with_state(draining)
}

async fn liveliness() -> Json<&'static str> {
    Json("I'm alive!")
}

async fn readiness(State(draining): State<Draining>) -> Response {
    if draining.is_draining() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"status": "shutting_down"})),
        )
            .into_response();
    }
    // The gateway has no database yet; Python reports "Not connected" when none is configured.
    Json(json!({"status": "healthy", "db": "Not connected"})).into_response()
}

#[cfg(test)]
mod tests {
    use axum::body::{Body, to_bytes};
    use axum::extract::Request;
    use rstest::rstest;
    use serde_json::Value;
    use tower::ServiceExt;

    use super::*;

    #[rstest]
    #[case::liveliness_serving("/health/liveliness", false, StatusCode::OK, json!("I'm alive!"))]
    #[case::liveness_serving("/health/liveness", false, StatusCode::OK, json!("I'm alive!"))]
    #[case::liveliness_draining("/health/liveliness", true, StatusCode::OK, json!("I'm alive!"))]
    #[case::readiness_serving(
        "/health/readiness",
        false,
        StatusCode::OK,
        json!({"status": "healthy", "db": "Not connected"}),
    )]
    #[case::readiness_draining(
        "/health/readiness",
        true,
        StatusCode::SERVICE_UNAVAILABLE,
        json!({"status": "shutting_down"}),
    )]
    #[tokio::test]
    async fn probes_answer_with_python_bodies(
        #[case] path: &str,
        #[case] draining: bool,
        #[case] status: StatusCode,
        #[case] expected: Value,
    ) {
        let state = Draining::default();
        if draining {
            state.start();
        }

        let response = router(state)
            .oneshot(Request::get(path).body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), status);
        let body = to_bytes(response.into_body(), 1024).await.unwrap();
        assert_eq!(serde_json::from_slice::<Value>(&body).unwrap(), expected);
    }
}
