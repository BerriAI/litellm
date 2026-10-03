use std::collections::HashMap;

use crate::{
    ReadError, TraceReader, TraceStore, TraceSummary, listed_summary,
    query::named::{ListTracesRow, ReadAccessParams, TracePageSpansParams, TraceSpansRow},
    resolve_trace,
};

use super::{
    map_store_error,
    spend::{spend, spend_window, spend_within},
    store::StoreError,
};

const RUNS_PER_SPAN_READ: usize = 16;

fn run_key(team_id: &str, api_key_hash: &str, trace_id: &str) -> (String, String, String) {
    (
        team_id.to_owned(),
        api_key_hash.to_owned(),
        trace_id.to_owned(),
    )
}

pub(super) async fn list_summaries<S: TraceStore>(
    reader: &TraceReader,
    store: &S,
    access: &ReadAccessParams,
    runs: &[ListTracesRow],
) -> Result<Vec<TraceSummary>, ReadError<S::Error>> {
    let (Some(start_ms), Some(end_ms)) = (
        runs.iter().map(|row| row.start_ms).min(),
        runs.iter()
            .map(|row| row.start_ms.saturating_add(row.duration_ms))
            .max(),
    ) else {
        return Ok(Vec::new());
    };
    let params = TracePageSpansParams {
        access: access.clone(),
        trace_refs: runs.iter().map(|row| row.trace_ref.clone()).collect(),
        start_ms,
        end_ms: end_ms.saturating_add(1),
    };
    let spans = match store.run_spans(&params, super::now_ms()).await {
        Ok(spans) => spans,
        Err(StoreError::TooLarge) => {
            let mut summaries = Vec::with_capacity(runs.len());
            for row in runs {
                summaries.push(run_summary(reader, store, access, row).await?);
            }
            return Ok(summaries);
        }
        Err(error) => return Err(map_store_error(error)),
    };
    let spend_rows = spend(store, access, &spans).await;
    let mut spans = spans;
    spans.sort_by(|left, right| {
        run_key(&left.team_id, &left.api_key_hash, &left.trace_id)
            .cmp(&run_key(
                &right.team_id,
                &right.api_key_hash,
                &right.trace_id,
            ))
            .then(left.start_ns.cmp(&right.start_ns))
    });
    let by_run: HashMap<_, &[TraceSpansRow]> = spans
        .chunk_by(|left, right| {
            (&left.team_id, &left.api_key_hash, &left.trace_id)
                == (&right.team_id, &right.api_key_hash, &right.trace_id)
        })
        .map(|run| {
            (
                run_key(&run[0].team_id, &run[0].api_key_hash, &run[0].trace_id),
                run,
            )
        })
        .collect();
    Ok(runs
        .iter()
        .map(|row| {
            let spans = by_run
                .get(&run_key(&row.team_id, &row.api_key_hash, &row.trace_id))
                .copied()
                .unwrap_or_default();
            let spend =
                spend_window(spans).map_or(&[][..], |window| spend_within(&spend_rows, window));
            resolve_trace(&row.trace_id, &row.trace_ref, spans, spend)
                .map_or_else(|| listed_summary(row), |trace| trace.summary)
        })
        .collect())
}

async fn run_summary<S: TraceStore>(
    reader: &TraceReader,
    store: &S,
    access: &ReadAccessParams,
    row: &ListTracesRow,
) -> Result<TraceSummary, ReadError<S::Error>> {
    match reader
        .get_trace(store, access, &row.trace_id, &row.trace_ref)
        .await
    {
        Ok(trace) => Ok(trace.map_or_else(|| listed_summary(row), |trace| trace.summary)),
        Err(ReadError::TooLarge) => Ok(listed_summary(row)),
        Err(error) => Err(error),
    }
}

pub(super) fn run_batches<T>(runs: &[T]) -> impl Iterator<Item = &[T]> + '_ {
    runs.chunks(RUNS_PER_SPAN_READ)
}
