use std::future::Future;

use litellm_traces::{
    QueryScope,
    store::{
        CallQuery, CallRow, RunCount, RunCountQuery, RunQuery, RunRow, SpanQuery, SpanRow,
        SpanText, SpanTextQuery,
    },
};

#[derive(Debug, thiserror::Error)]
pub enum StoreError<E> {
    /// The answer would exceed what storage returns in one response; a smaller `limit` may fit.
    #[error("trace read exceeds the storage read budget")]
    TooLarge,
    #[error(transparent)]
    Failed(E),
}

pub type StoreResult<T, E> = Result<T, StoreError<E>>;

/// Every read trace storage serves. Each call returns at most `limit` rows from one round trip;
/// paging, budgets and caching live above it.
pub trait TraceStore: Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    /// Identifies the backing storage for snapshot cache keys.
    fn source(&self) -> &str;

    fn runs(
        &self,
        access: &QueryScope,
        query: &RunQuery,
    ) -> impl Future<Output = StoreResult<Vec<RunRow>, Self::Error>> + Send;

    fn run_counts(
        &self,
        access: &QueryScope,
        query: &RunCountQuery,
    ) -> impl Future<Output = StoreResult<Vec<RunCount>, Self::Error>> + Send;

    fn spans(
        &self,
        access: &QueryScope,
        query: &SpanQuery,
    ) -> impl Future<Output = StoreResult<Vec<SpanRow>, Self::Error>> + Send;

    fn span_text(
        &self,
        access: &QueryScope,
        query: &SpanTextQuery,
    ) -> impl Future<Output = StoreResult<Vec<SpanText>, Self::Error>> + Send;

    fn calls(
        &self,
        access: &QueryScope,
        query: &CallQuery,
    ) -> impl Future<Output = StoreResult<Vec<CallRow>, Self::Error>> + Send;
}
