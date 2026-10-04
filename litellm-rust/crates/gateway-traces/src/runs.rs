use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    Extension, Json,
    extract::{Path, Query, State},
};
use litellm_traces::{
    QueryScope, TracePage,
    search::{RunField, RunFilter, RunSearch, RunValues, TraceHistogram},
};
use litellm_traces_cache::{PageRequest, TraceStore};
use serde::Deserialize;

use crate::{ReadFailure, Traces};

const DAY_MS: i64 = 24 * 60 * 60 * 1000;
const PAGE_SIZE: u32 = 50;

#[derive(Deserialize)]
pub(crate) struct Runs {
    start_ms: Option<i64>,
    end_ms: Option<i64>,
    #[serde(default)]
    q: String,
}

impl Runs {
    fn filter(&self) -> RunFilter {
        let now_ms = now_ms();
        RunFilter {
            start_ms: self.start_ms.unwrap_or(now_ms - DAY_MS),
            end_ms: self.end_ms.unwrap_or(now_ms),
            search: RunSearch::parse(&self.q),
        }
    }
}

#[derive(Deserialize)]
pub(crate) struct Page {
    cursor: Option<String>,
}

pub(crate) async fn list<S: TraceStore>(
    State(traces): State<Arc<Traces<S>>>,
    Extension(access): Extension<QueryScope>,
    Query(runs): Query<Runs>,
    Query(Page { cursor }): Query<Page>,
) -> Result<Json<TracePage>, ReadFailure> {
    let page = PageRequest {
        cursor,
        limit: PAGE_SIZE,
    };
    Ok(Json(
        traces
            .reader
            .list_traces(&traces.store, &access, &runs.filter(), &page)
            .await?,
    ))
}

#[derive(Deserialize)]
pub(crate) struct Buckets {
    #[serde(default = "default_buckets")]
    buckets: u32,
}

fn default_buckets() -> u32 {
    60
}

pub(crate) async fn histogram<S: TraceStore>(
    State(traces): State<Arc<Traces<S>>>,
    Extension(access): Extension<QueryScope>,
    Query(runs): Query<Runs>,
    Query(Buckets { buckets }): Query<Buckets>,
) -> Result<Json<TraceHistogram>, ReadFailure> {
    Ok(Json(
        traces
            .reader
            .histogram(&traces.store, &access, &runs.filter(), buckets)
            .await?,
    ))
}

#[derive(Deserialize)]
pub(crate) struct Values {
    #[serde(default)]
    contains: String,
    #[serde(default = "default_values")]
    limit: u32,
}

fn default_values() -> u32 {
    20
}

pub(crate) async fn values<S: TraceStore>(
    State(traces): State<Arc<Traces<S>>>,
    Extension(access): Extension<QueryScope>,
    Path(field): Path<RunField>,
    Query(runs): Query<Runs>,
    Query(values): Query<Values>,
) -> Result<Json<RunValues>, ReadFailure> {
    Ok(Json(
        traces
            .reader
            .values(
                &traces.store,
                &access,
                &runs.filter(),
                field,
                &values.contains,
                values.limit,
            )
            .await?,
    ))
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_millis() as i64)
}
