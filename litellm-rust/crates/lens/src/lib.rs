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
    future::Future,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};
pub use storage::Storage;
use tokio::sync::Semaphore;

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

const READ_QUEUE_WAIT: Duration = Duration::from_secs(10);

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

async fn wait_for_read_slot<P>(
    acquire: impl Future<Output = Result<P, tokio::sync::AcquireError>>,
) -> Result<P, Error> {
    tokio::time::timeout(READ_QUEUE_WAIT, acquire)
        .await
        .map_err(|_| Error::Unavailable)?
        .map_err(|_| Error::Unavailable)
}

pub fn router(state: Arc<State>) -> Router {
    router_with_root_path(state, "")
}

pub fn router_with_root_path(state: Arc<State>, server_root_path: &str) -> Router {
    let public = Router::new()
        .route("/health/live", get(|| async { StatusCode::OK }))
        .route("/health/ready", get(ready))
        .route("/v1/traces", post(traces))
        .route("/v1/logs", post(logs))
        .route("/v1/traces/receipt", post(receipt))
        .layer(
            tower_http::cors::CorsLayer::new()
                .allow_origin(tower_http::cors::Any)
                .allow_methods([http::Method::POST, http::Method::GET])
                .allow_headers([
                    http::header::AUTHORIZATION,
                    http::header::CONTENT_TYPE,
                    http::header::CONTENT_ENCODING,
                ]),
        );
    public
        .clone()
        .nest(&format!("{server_root_path}/lens-ingest"), public)
        .merge(
            Router::new()
                .route("/internal/read", post(read))
                .route("/internal/spend", post(spend))
                .route("/internal/feedback", post(feedback))
                .route("/internal/credentials", post(credentials))
                .route("/internal/status", get(status)),
        )
        .with_state(state)
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ReceiptRequest {
    trace_id: String,
    #[serde(default)]
    span_ids: Vec<String>,
}

async fn receipt(
    AppState(state): AppState<Arc<State>>,
    headers: HeaderMap,
    body: Body,
) -> Result<Json<Value>, Error> {
    let tenant = state.credentials.tenant(&headers)?;
    state.require_storage()?;
    let _permit = wait_for_read_slot(state.read_slots.acquire()).await?;
    let body = tokio::time::timeout(Duration::from_secs(5), to_bytes(body, 64 * 1024))
        .await
        .map_err(|_| Error::Unavailable)?
        .map_err(|_| Error::TooLarge)?;
    let request: ReceiptRequest =
        serde_json::from_slice(&body).map_err(|_| Error::InvalidRequest)?;
    let received = litellm_traces_clickhouse::trace_received(
        &state.storage.client,
        state.storage.config.storage().reader(),
        &tenant,
        &request.trace_id,
        &request.span_ids,
    )
    .await?;
    Ok(Json(serde_json::json!({"received": received})))
}

async fn status(
    AppState(state): AppState<Arc<State>>,
    headers: HeaderMap,
) -> Result<Json<Value>, Error> {
    auth::authorize_service(&headers, &state.service_token)?;
    Ok(Json(serde_json::json!({
        "storage_ready": state.schema_ready.load(Ordering::Acquire),
        "credentials_ready": state.credentials.ready(),
        "release": std::env::var("LITELLM_RELEASE_TAG").unwrap_or_default(),
        "protocol_version": wire::PROTOCOL_VERSION,
    })))
}

async fn credentials(
    AppState(state): AppState<Arc<State>>,
    headers: HeaderMap,
    body: Body,
) -> Result<StatusCode, Error> {
    auth::authorize_service(&headers, &state.service_token)?;
    let body = tokio::time::timeout(Duration::from_secs(5), to_bytes(body, 8 * 1024 * 1024))
        .await
        .map_err(|_| Error::Unavailable)?
        .map_err(|_| Error::TooLarge)?;
    state
        .credentials
        .replace(serde_json::from_slice(&body).map_err(|_| Error::InvalidRequest)?)?;
    Ok(StatusCode::NO_CONTENT)
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
    let permit = wait_for_read_slot(state.read_slots.clone().acquire_owned()).await?;
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
    insert(state, headers, body, InsertTable::SpendLogs).await
}

async fn feedback(
    AppState(state): AppState<Arc<State>>,
    headers: HeaderMap,
    body: Body,
) -> Result<StatusCode, Error> {
    insert(state, headers, body, InsertTable::LensFeedback).await
}

async fn insert(
    state: Arc<State>,
    headers: HeaderMap,
    body: Body,
    table: InsertTable,
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
            table,
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

#[cfg(test)]
mod tests {
    use super::{Error, READ_QUEUE_WAIT, wait_for_read_slot};
    use std::sync::Arc;
    use tokio::sync::Semaphore;

    #[tokio::test]
    async fn ninth_read_waits_for_a_permit_and_succeeds() {
        let slots = Arc::new(Semaphore::new(8));
        let permits = (0..8)
            .map(|_| slots.clone().try_acquire_owned().expect("available permit"))
            .collect::<Vec<_>>();
        let waiting_slots = slots.clone();
        let waiting =
            tokio::spawn(async move { wait_for_read_slot(waiting_slots.acquire_owned()).await });

        tokio::task::yield_now().await;
        assert!(!waiting.is_finished());
        drop(permits);
        assert!(waiting.await.expect("joined read").is_ok());
    }

    #[tokio::test(start_paused = true)]
    async fn read_queue_timeout_returns_unavailable() {
        let slots = Arc::new(Semaphore::new(0));
        let waiting = tokio::spawn(wait_for_read_slot(slots.acquire_owned()));

        tokio::task::yield_now().await;
        tokio::time::advance(READ_QUEUE_WAIT).await;
        assert!(matches!(
            waiting.await.expect("joined read"),
            Err(Error::Unavailable)
        ));
    }
}
