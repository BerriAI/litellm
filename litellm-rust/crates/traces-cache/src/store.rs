use std::future::Future;

use litellm_traces::query::named::{
    ListTracesParams, ListTracesRow, SpanDetailParams, SpanDetailRow, SpanErrorParams,
    SpanErrorRow, SpendByResponseIdsParams, SpendByResponseIdsRow, TraceIdentityParams,
    TracePageSpansParams, TraceSpansParams, TraceSpansRow,
};

#[derive(Debug, thiserror::Error)]
pub enum StoreError<E> {
    #[error("trace read exceeds the storage read budget")]
    TooLarge,
    #[error(transparent)]
    Failed(E),
}

pub trait TraceStore: Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    /// Identifies the backing storage for snapshot cache keys.
    fn source(&self) -> &str;

    fn trace_refs(
        &self,
        params: &TraceIdentityParams,
    ) -> impl Future<Output = Result<Vec<String>, StoreError<Self::Error>>> + Send;

    /// Returns `TooLarge` when the response exceeds the storage limit so the reader can halve `limit`.
    fn list_runs(
        &self,
        params: &ListTracesParams,
    ) -> impl Future<Output = Result<Vec<ListTracesRow>, StoreError<Self::Error>>> + Send;

    /// Returns spans visible at `snapshot_ms`, sorted by `start_ns`, or `TooLarge` past `MAX_GRAPH_BYTES`/`MAX_GRAPH_SPANS`.
    fn trace_spans(
        &self,
        params: &TraceSpansParams,
        snapshot_ms: u64,
    ) -> impl Future<Output = Result<Vec<TraceSpansRow>, StoreError<Self::Error>>> + Send;

    /// Returns spans visible at `snapshot_ms`, sorted by `start_ns`, or `TooLarge` past `MAX_GRAPH_BYTES`/`MAX_GRAPH_SPANS`.
    fn run_spans(
        &self,
        params: &TracePageSpansParams,
        snapshot_ms: u64,
    ) -> impl Future<Output = Result<Vec<TraceSpansRow>, StoreError<Self::Error>>> + Send;

    fn spend(
        &self,
        params: &SpendByResponseIdsParams,
    ) -> impl Future<Output = Result<Vec<SpendByResponseIdsRow>, StoreError<Self::Error>>> + Send;

    fn span_detail(
        &self,
        params: &SpanDetailParams,
    ) -> impl Future<Output = Result<Option<SpanDetailRow>, StoreError<Self::Error>>> + Send;

    fn span_error(
        &self,
        params: &SpanErrorParams,
    ) -> impl Future<Output = Result<Option<SpanErrorRow>, StoreError<Self::Error>>> + Send;
}
