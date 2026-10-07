use std::collections::HashMap;

use crate::{
    ReadError, SnapshotKey, TraceReader, TraceStore,
    cache::{Freshness, ListedRun},
    estimates::ResolutionInput,
    reader::{map_store_error, now_ms},
    spend::{spend, spend_window, spend_within},
    store::StoreError,
};
use litellm_traces::{
    TraceSummary, listed_summary,
    query::named::{ListTracesRow, ReadAccessParams, TracePageSpansParams, TraceSpansRow},
};

const RUNS_PER_SPAN_READ: usize = 16;

fn run_key(team_id: &str, api_key_hash: &str, trace_id: &str) -> (String, String, String) {
    (
        team_id.to_owned(),
        api_key_hash.to_owned(),
        trace_id.to_owned(),
    )
}

fn cache_key<E>(
    source: &str,
    access: &ReadAccessParams,
    row: &ListTracesRow,
) -> Result<SnapshotKey, ReadError<E>> {
    Ok(SnapshotKey::run(
        source,
        access,
        (
            &row.team_id,
            &row.api_key_hash,
            &row.trace_id,
            &row.trace_ref,
        ),
    )?)
}

fn summary(row: &ListTracesRow, listed: Option<&ListedRun>) -> TraceSummary {
    match listed {
        Some(ListedRun::Resolved(summary, _)) => (**summary).clone(),
        Some(ListedRun::Limited) | None => listed_summary(row),
    }
}

/// Summaries for one batch of listed runs. Runs resolved within their freshness window come
/// from the cache; only the rest are read from storage, with one span and one spend read.
pub(super) async fn list_summaries<S: TraceStore>(
    reader: &TraceReader,
    store: &S,
    access: &ReadAccessParams,
    runs: &[ListTracesRow],
) -> Result<Vec<TraceSummary>, ReadError<S::Error>> {
    let mut keys = Vec::with_capacity(runs.len());
    let mut listed = Vec::with_capacity(runs.len());
    for row in runs {
        let key = cache_key(store.source(), access, row)?;
        listed.push(reader.lists.runs.get(&key).await);
        keys.push(key);
    }
    let misses: Vec<&ListTracesRow> = runs
        .iter()
        .zip(&listed)
        .filter_map(|(row, listed)| listed.is_none().then_some(row))
        .collect();
    let mut resolved = resolve_runs(reader, store, access, &misses)
        .await?
        .into_iter();
    let mut summaries = Vec::with_capacity(runs.len());
    for ((row, key), cached) in runs.iter().zip(keys).zip(listed) {
        let listed = match cached {
            Some(listed) => Some(listed),
            None => {
                let listed = resolved.next().flatten();
                if let Some(listed) = &listed {
                    reader.lists.runs.insert(key, listed.clone()).await;
                }
                listed
            }
        };
        summaries.push(summary(row, listed.as_ref()));
    }
    Ok(summaries)
}

/// One entry per run; `None` means the run could not be resolved and keeps its listed summary
/// without being cached.
async fn resolve_runs<S: TraceStore>(
    reader: &TraceReader,
    store: &S,
    access: &ReadAccessParams,
    runs: &[&ListTracesRow],
) -> Result<Vec<Option<ListedRun>>, ReadError<S::Error>> {
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
    let snapshot_ms = now_ms();
    let spans = match store.run_spans(&params, snapshot_ms).await {
        Ok(spans) => spans,
        Err(StoreError::TooLarge) => {
            let mut resolved = Vec::with_capacity(runs.len());
            for row in runs {
                resolved.push(resolve_run(reader, store, access, row).await?);
            }
            return Ok(resolved);
        }
        Err(error) => return Err(map_store_error(error)),
    };
    let Some(spend_rows) = spend(store, access, &spans).await else {
        // The batch's combined spend read failed; a run's own narrower window may still
        // resolve, so fall back per run instead of leaving every run in the batch costless.
        let mut resolved = Vec::with_capacity(runs.len());
        for row in runs {
            resolved.push(resolve_run(reader, store, access, row).await?);
        }
        return Ok(resolved);
    };
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
    let inputs: Vec<_> = runs
        .iter()
        .map(|row| {
            let spans = by_run
                .get(&run_key(&row.team_id, &row.api_key_hash, &row.trace_id))
                .copied()
                .unwrap_or_default();
            ResolutionInput {
                trace_id: &row.trace_id,
                trace_ref: &row.trace_ref,
                rows: spans,
                spend: spend_window(spans)
                    .map_or(&[][..], |window| spend_within(&spend_rows, window)),
            }
        })
        .collect();
    Ok(reader
        .resolve_batch(&inputs)
        .await
        .into_iter()
        .zip(inputs)
        .map(|(trace, input)| {
            trace.map(|trace| {
                let freshness = Freshness::of(input.rows, &trace, snapshot_ms);
                ListedRun::Resolved(Box::new(trace.summary), freshness)
            })
        })
        .collect())
}

async fn resolve_run<S: TraceStore>(
    reader: &TraceReader,
    store: &S,
    access: &ReadAccessParams,
    row: &ListTracesRow,
) -> Result<Option<ListedRun>, ReadError<S::Error>> {
    match reader
        .current(store, access, &row.trace_id, &row.trace_ref)
        .await
    {
        Ok(snapshot) => Ok(snapshot.map(|snapshot| {
            ListedRun::Resolved(
                Box::new(snapshot.trace().summary.clone()),
                snapshot.freshness(),
            )
        })),
        Err(ReadError::TooLarge) => Ok(Some(ListedRun::Limited)),
        Err(error) => Err(error),
    }
}

pub(super) fn run_batches<T>(runs: &[T]) -> impl Iterator<Item = &[T]> + '_ {
    runs.chunks(RUNS_PER_SPAN_READ)
}
