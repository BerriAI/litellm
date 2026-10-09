use std::ops::Range;

use crate::TraceStore;
use litellm_traces::{
    SpendLookup,
    query::named::{
        ReadAccessParams, SpendByResponseIdsParams, SpendByResponseIdsRow, TraceSpansRow,
    },
};

const NANOS_PER_MS: i64 = 1_000_000;
const SPEND_SLACK_MS: i64 = 2 * 60 * 1000;

pub(super) fn spend_window(rows: &[TraceSpansRow]) -> Option<Range<i64>> {
    let start_ns = rows.iter().map(|row| row.start_ns).min()?;
    let end_ns = rows
        .iter()
        .map(|row| row.start_ns.saturating_add_unsigned(row.duration_ns))
        .max()?;
    Some(
        start_ns.div_euclid(NANOS_PER_MS) - SPEND_SLACK_MS
            ..end_ns.div_euclid(NANOS_PER_MS) + SPEND_SLACK_MS,
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

/// Spend rows sorted by `start_ms`, or `None` when the lookup failed and spend is unknown.
pub(super) async fn spend<S: TraceStore>(
    store: &S,
    access: &ReadAccessParams,
    rows: &[TraceSpansRow],
) -> Option<Vec<SpendByResponseIdsRow>> {
    let lookup = SpendLookup::new(rows);
    let Some(window) = spend_window(rows) else {
        return Some(Vec::new());
    };
    if lookup.is_empty() {
        return Some(Vec::new());
    }
    let params = SpendByResponseIdsParams {
        access: access.clone(),
        response_ids: lookup.response_ids,
        provider_request_ids: lookup.provider_request_ids,
        request_ids: lookup.request_ids,
        trace_ids: lookup.trace_ids,
        start_ms: window.start,
        end_ms: window.end,
    };
    match store.spend(&params).await {
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

#[cfg(test)]
mod tests {
    use litellm_traces::{ObservationType, SpanStatus};
    use rstest::rstest;

    use super::*;

    fn span(start_ns: i64, duration_ns: u64) -> TraceSpansRow {
        TraceSpansRow {
            trace_id: "trace".into(),
            original_trace_id: String::new(),
            span_id: start_ns.to_string(),
            parent_span_id: String::new(),
            name: "call".into(),
            kind: ObservationType::Llm,
            wrapper_candidate: false,
            agent: String::new(),
            framework: String::new(),
            status: SpanStatus::Ok,
            status_message: String::new(),
            error_truncated: false,
            start_ns,
            duration_ns,
            service: String::new(),
            input_preview: String::new(),
            model: String::new(),
            input_tokens: 0,
            output_tokens: 0,
            litellm_request_id: String::new(),
            call_keys: Vec::new(),
            call_evidence: None,
            tool_call_id: String::new(),
            source_type: String::new(),
            source_url: String::new(),
            source_title: String::new(),
            team_id: String::new(),
            api_key_hash: String::new(),
            user_id: String::new(),
        }
    }

    #[rstest]
    #[case::single_span(
        vec![span(1_790_000_000_000_000_000, 4_000_000_000)],
        1_790_000_000_000,
        1_790_000_004_000
    )]
    #[case::latest_end_wins_over_latest_start(
        vec![
            span(1_790_000_000_000_000_000, 90_000_000_000),
            span(1_790_000_010_000_000_000, 1_000_000_000),
        ],
        1_790_000_000_000,
        1_790_000_090_000
    )]
    #[case::sub_millisecond_floors(
        vec![span(1_790_000_000_000_999_999, 1)],
        1_790_000_000_000,
        1_790_000_000_001
    )]
    fn window_covers_the_trace_plus_a_short_slack(
        #[case] rows: Vec<TraceSpansRow>,
        #[case] first_ms: i64,
        #[case] last_ms: i64,
    ) {
        let window = spend_window(&rows).expect("non-empty trace has a window");
        assert_eq!(window.start, first_ms - SPEND_SLACK_MS);
        assert_eq!(window.end, last_ms + SPEND_SLACK_MS);
        assert!((60_000..=5 * 60_000).contains(&SPEND_SLACK_MS));
    }

    #[rstest]
    fn empty_trace_has_no_window() {
        assert_eq!(spend_window(&[]), None);
    }
}
