use std::ops::Range;

use crate::TraceStore;
use litellm_traces::{
    SpendLookup,
    query::named::{
        ReadAccessParams, SpendByResponseIdsParams, SpendByResponseIdsRow, TraceSpansRow,
    },
};

const NANOS_PER_MS: i64 = 1_000_000;
const SPEND_WINDOW_MS: i64 = 30 * 60 * 1000;

pub(super) fn spend_window(rows: &[TraceSpansRow]) -> Option<Range<i64>> {
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

pub(super) fn spend_within(
    spend: &[SpendByResponseIdsRow],
    window: Range<i64>,
) -> &[SpendByResponseIdsRow] {
    let first = spend.partition_point(|row| row.start_ms < window.start);
    let end = spend.partition_point(|row| row.start_ms < window.end);
    &spend[first..end.max(first)]
}

pub(super) async fn spend<S: TraceStore>(
    store: &S,
    access: &ReadAccessParams,
    rows: &[TraceSpansRow],
) -> Vec<SpendByResponseIdsRow> {
    let lookup = SpendLookup::new(rows);
    let Some(window) = spend_window(rows) else {
        return Vec::new();
    };
    if lookup.is_empty() {
        return Vec::new();
    }
    let params = SpendByResponseIdsParams {
        access: access.clone(),
        response_ids: lookup.response_ids,
        request_ids: lookup.request_ids,
        trace_ids: lookup.trace_ids,
        start_ms: window.start,
        end_ms: window.end,
    };
    match store.spend(&params).await {
        Ok(rows) => {
            let mut rows = rows;
            rows.sort_by_key(|row| row.start_ms);
            rows
        }
        Err(error) => {
            tracing::warn!(%error, "trace spend lookup unavailable");
            Vec::new()
        }
    }
}
