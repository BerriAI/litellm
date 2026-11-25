use std::time::Duration;

use axum::{Json, Router, extract::State, routing::get};
use litellm_db::Pools;
use serde_json::{Value, json};

use crate::AppState;

const DATABASE_CHECK_TIMEOUT: Duration = Duration::from_secs(2);

pub(crate) fn router(state: AppState) -> Router {
    Router::new()
        .route("/health/readiness", get(readiness))
        .with_state(state)
}

async fn readiness(State(database): State<Option<Pools>>) -> Json<Value> {
    let db = match &database {
        None => "Not connected",
        Some(pools) => {
            let check = litellm_db::health_check(pools.writer());
            match litellm_db::within(Some(DATABASE_CHECK_TIMEOUT), check).await {
                Ok(()) => "connected",
                Err(_) => "disconnected",
            }
        }
    };
    Json(json!({"status": "healthy", "db": db}))
}
