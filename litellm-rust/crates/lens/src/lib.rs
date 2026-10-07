pub mod activity;
pub mod agent;
pub mod auth;
pub mod config;
pub mod control;
mod error;
pub mod evidence;
pub mod grouping;
mod ingest;
pub mod journal;
pub mod model;
pub mod pipeline;
pub mod sandbox;
mod storage;
pub mod worker;

use axum::{
    Json, Router,
    body::{Body, to_bytes},
    extract::State as AppState,
    http::{HeaderMap, StatusCode},
    routing::{get, post},
};
pub use error::Error;
use litellm_traces_clickhouse::InsertTable;
use serde_json::Value;
use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};
pub use storage::Storage;

#[allow(
    dead_code,
    reason = "the schema generator emits default helpers shared across contracts"
)]
#[allow(
    clippy::derivable_impls,
    clippy::type_complexity,
    reason = "typify generates explicit defaults and contract tuple types"
)]
pub mod wire {
    include!(concat!(env!("OUT_DIR"), "/wire.rs"));
}
use tokio::sync::Semaphore;

pub struct State {
    pub credentials: Arc<auth::Credentials>,
    pub storage: Storage,
    pub schema_ready: AtomicBool,
    service_token: String,
    ingest_slots: Arc<Semaphore>,
    read_slots: Arc<Semaphore>,
    export_slots: Arc<Semaphore>,
}

impl State {
    pub fn new(storage: Storage, service_token: String) -> Self {
        Self {
            credentials: Arc::new(auth::Credentials::default()),
            storage,
            schema_ready: AtomicBool::new(false),
            service_token,
            ingest_slots: Arc::new(Semaphore::new(2)),
            read_slots: Arc::new(Semaphore::new(8)),
            export_slots: Arc::new(Semaphore::new(2)),
        }
    }

    fn require_storage(&self) -> Result<(), Error> {
        if self.schema_ready.load(Ordering::Acquire) {
            Ok(())
        } else {
            Err(Error::Unavailable)
        }
    }
}

pub fn router(state: Arc<State>) -> Router {
    Router::new()
        .route("/health/live", get(|| async { StatusCode::OK }))
        .route("/health/ready", get(ready))
        .route("/v1/traces", post(traces))
        .route("/v1/logs", post(logs))
        .route("/internal/read", post(read))
        .route("/internal/spend", post(spend))
        .with_state(state)
}

async fn ready(AppState(state): AppState<Arc<State>>) -> StatusCode {
    if state.schema_ready.load(Ordering::Acquire) && state.credentials.ready() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

async fn traces(
    AppState(state): AppState<Arc<State>>,
    headers: HeaderMap,
    body: Body,
) -> axum::response::Response {
    ingest::receive(state, headers, body, false).await
}

async fn logs(
    AppState(state): AppState<Arc<State>>,
    headers: HeaderMap,
    body: Body,
) -> axum::response::Response {
    ingest::receive(state, headers, body, true).await
}

async fn read(
    AppState(state): AppState<Arc<State>>,
    headers: HeaderMap,
    body: Body,
) -> Result<Json<Value>, Error> {
    auth::authorize_service(&headers, &state.service_token)?;
    state.require_storage()?;
    let permit = state
        .read_slots
        .clone()
        .try_acquire_owned()
        .map_err(|_| Error::Unavailable)?;
    let body = tokio::time::timeout(Duration::from_secs(10), to_bytes(body, 1024 * 1024))
        .await
        .map_err(|_| Error::Unavailable)?
        .map_err(|_| Error::TooLarge)?;
    let request = serde_json::from_slice(&body).map_err(|_| Error::InvalidRequest)?;
    tokio::spawn(async move {
        let _permit = permit;
        state.storage.read(request).await.map(Json)
    })
    .await
    .map_err(|_| Error::Unavailable)?
}

async fn spend(
    AppState(state): AppState<Arc<State>>,
    headers: HeaderMap,
    body: Body,
) -> Result<StatusCode, Error> {
    auth::authorize_service(&headers, &state.service_token)?;
    state.require_storage()?;
    let permit = state
        .export_slots
        .clone()
        .try_acquire_owned()
        .map_err(|_| Error::Unavailable)?;
    let body = tokio::time::timeout(Duration::from_secs(10), to_bytes(body, 8 * 1024 * 1024))
        .await
        .map_err(|_| Error::Unavailable)?
        .map_err(|_| Error::TooLarge)?;
    tokio::spawn(async move {
        let _permit = permit;
        let rows: Vec<BTreeMap<String, Value>> =
            serde_json::from_slice(&body).map_err(|_| Error::InvalidRequest)?;
        if rows.len() > 1000 {
            return Err(Error::TooLarge);
        }
        litellm_traces_clickhouse::insert_rows(
            &state.storage.client,
            state.storage.config.storage().writer(),
            state.storage.config.storage().database(),
            InsertTable::SpendLogs,
            rows,
        )
        .await?;
        Ok(StatusCode::NO_CONTENT)
    })
    .await
    .map_err(|_| Error::Unavailable)?
}

pub async fn provision(state: Arc<State>) {
    loop {
        let ready = if state.schema_ready.load(Ordering::Acquire) {
            tokio::time::timeout(Duration::from_secs(5), state.storage.ping())
                .await
                .is_ok_and(|r| r.is_ok())
        } else {
            tokio::time::timeout(Duration::from_secs(30), state.storage.ensure_schema())
                .await
                .is_ok_and(|r| r.is_ok())
        };
        state.schema_ready.store(ready, Ordering::Release);
        if !ready {
            tracing::warn!("Lens storage unavailable; retrying");
        }
        tokio::time::sleep(Duration::from_secs(10)).await;
    }
}
