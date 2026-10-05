use std::ops::Range;

use litellm_traces::{
    QueryScope, SpendLookup,
    store::{CallQuery, CallRow, SpanRow},
};

use crate::{TraceStore, pages::read_all};

const NANOS_PER_MS: i64 = 1_000_000;
const SPEND_WINDOW_MS: i64 = 30 * 60 * 1000;

pub(super) fn spend_window(rows: &[SpanRow]) -> Option<Range<i64>> {
    let start_ns = rows.iter().map(|row| row.start_ns).min()?;
    let end_ns = rows
        .iter()
        .map(|row| row.start_ns.saturating_add_unsigned(row.duration_ns))
        .max()?;
    Some(
        start_ns.div_euclid(NANOS_PER_MS) - SPEND_WINDOW_MS
            ..end_ns.div_euclid(NANOS_PER_MS) + SPEND_WINDOW_MS,
    )
}

pub(super) fn spend_within(spend: &[CallRow], window: Range<i64>) -> &[CallRow] {
    let first = spend.partition_point(|row| row.start_ms < window.start);
    let end = spend.partition_point(|row| row.start_ms < window.end);
    &spend[first..end.max(first)]
}

/// Spend rows sorted by `start_ms`, or `None` when the lookup failed and spend is unknown.
pub(super) async fn spend<S: TraceStore>(
    store: &S,
    access: &QueryScope,
    rows: &[SpanRow],
    as_of_ms: u64,
) -> Option<Vec<CallRow>> {
    let lookup = SpendLookup::new(rows);
    let Some(window) = spend_window(rows) else {
        return Some(Vec::new());
    };
    if lookup.is_empty() {
        return Some(Vec::new());
    }
    let calls = read_all(|after, limit| {
        let query = CallQuery {
            as_of_ms,
            window: window.clone(),
            response_ids: lookup.response_ids.clone(),
            request_ids: lookup.request_ids.clone(),
            trace_ids: lookup.trace_ids.clone(),
            after,
            limit,
        };
        async move { store.calls(access, &query).await }
    })
    .await;
    match calls {
        Ok(rows) => {
            let mut rows = rows;
            rows.sort_by_key(|row| row.start_ms);
            Some(rows)
        }
        Err(error) => {
            tracing::warn!(%error, "trace spend lookup unavailable");
            None
        }
    }
}
