//! Scoped trace reads: the trace list, one trace resolved with its spend, and span payloads.

use std::sync::LazyLock;
use std::time::Duration;

use base64::{Engine, engine::general_purpose::URL_SAFE};
use futures_util::{StreamExt, TryStreamExt, stream};
use itertools::Itertools;
use litellm_http::Client;
use litellm_storage_clickhouse::{Query, fetch};
use litellm_traces::{
    SpanDetail, SpanErrorPage, SpendLookup, Trace, TracePage, listed_summary,
    query::named as contracts, resolve_trace, to_ui_content,
};
use litellm_traces_cache::{SnapshotCache, SnapshotKey};
use serde::{Deserialize, Serialize};

use crate::{
    Connection, Error,
    query::named::{
        ListTracesParams, ListTracesRow, ReadAccessParams, SpanDetail as SpanDetailQuery,
        SpanDetailParams, SpanError, SpanErrorParams, SpendByResponseIdsParams, TraceIdentity,
        TraceIdentityParams, TracePageSpansParams, TraceSpansParams,
    },
};

struct RunCandidates;

impl Query for RunCandidates {
    type Params = ListTracesParams;
    type Row = ListTracesRow;
    const SQL: &'static str = concat!(
        "SELECT * EXCEPT (request_ids), [] AS request_ids FROM (",
        include_str!("../query/list_traces.sql"),
        ") ORDER BY start_ms DESC, trace_ref DESC"
    );
}

// Cursor pages share a bounded snapshot so advancing does not resolve the whole graph again.
static TRACE_SNAPSHOTS: LazyLock<SnapshotCache> = LazyLock::new(|| {
    SnapshotCache::new(
        crate::span_batches::MAX_GRAPH_BYTES,
        Duration::from_secs(120),
    )
});

const NANOS_PER_MS: i64 = 1_000_000;
const SPEND_WINDOW_MS: i64 = 30 * 60 * 1000;
const SPEND_CONCURRENCY: usize = 4;

fn encode_cursor<T: Serialize>(position: &T) -> String {
    URL_SAFE.encode(serde_json::to_vec(position).unwrap_or_default())
}

fn decode_cursor<T: for<'de> Deserialize<'de>>(
    cursor: &str,
    kind: &'static str,
) -> Result<T, Error> {
    URL_SAFE
        .decode(cursor)
        .ok()
        .and_then(|json| serde_json::from_slice(&json).ok())
        .ok_or(Error::InvalidCursor(kind))
}

fn trace_position(cursor: Option<&str>) -> Result<(i64, String), Error> {
    let Some(cursor) = cursor.filter(|cursor| !cursor.is_empty()) else {
        return Ok((0, String::new()));
    };
    match decode_cursor::<(i64, String)>(cursor, "trace")? {
        (start_ms, trace_ref) if start_ms > 0 && !trace_ref.is_empty() => Ok((start_ms, trace_ref)),
        _ => Err(Error::InvalidCursor("trace")),
    }
}

#[derive(Deserialize, Serialize)]
struct ErrorPosition {
    offset: u64,
    version: String,
}

fn error_position(cursor: Option<&str>) -> Result<Option<ErrorPosition>, Error> {
    let Some(cursor) = cursor else {
        return Ok(None);
    };
    let position = decode_cursor::<ErrorPosition>(cursor, "diagnostic")?;
    let valid_version = position.version.len() == 64
        && position
            .version
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'A'..=b'F').contains(&byte));
    if i64::try_from(position.offset).is_err() || !valid_version {
        return Err(Error::InvalidCursor("diagnostic"));
    }
    Ok(Some(position))
}

/// The stored run a trace id names for this caller; ids can repeat across tenants and runs.
async fn reference(
    client: &Client,
    connection: &Connection,
    access: &ReadAccessParams,
    trace_id: &str,
    trace_ref: &str,
) -> Result<Option<String>, Error> {
    if !trace_ref.is_empty() {
        return Ok(Some(trace_ref.to_owned()));
    }
    let params = TraceIdentityParams {
        access: access.clone(),
        trace_id: trace_id.to_owned(),
    };
    let mut identities = fetch::<TraceIdentity>(client, connection, &params).await?;
    if identities.len() > 1 {
        return Err(Error::AmbiguousTrace);
    }
    Ok(identities.pop().map(|identity| identity.trace_ref))
}

/// Spend records behind the spans' calls. A failed lookup leaves cost unknown instead of failing
/// the read.
async fn spend(
    client: &Client,
    connection: &Connection,
    access: &ReadAccessParams,
    rows: &[contracts::TraceSpansRow],
) -> Vec<contracts::SpendByResponseIdsRow> {
    let lookup = SpendLookup::new(rows);
    let (Some(start_ns), Some(end_ns)) = (
        rows.iter().map(|row| row.start_ns).min(),
        rows.iter()
            .map(|row| row.start_ns.saturating_add_unsigned(row.duration_ns))
            .max(),
    ) else {
        return Vec::new();
    };
    if lookup.is_empty() {
        return Vec::new();
    }
    let params = SpendByResponseIdsParams::from(contracts::SpendByResponseIdsParams {
        access: access.clone(),
        response_ids: lookup.response_ids,
        request_ids: lookup.request_ids,
        trace_ids: lookup.trace_ids,
        start_ms: start_ns.div_euclid(NANOS_PER_MS) - SPEND_WINDOW_MS,
        end_ms: end_ns.div_euclid(NANOS_PER_MS) + SPEND_WINDOW_MS,
    });
    match crate::span_batches::read_spend(client, connection, params).await {
        Ok(rows) => rows,
        Err(error) => {
            tracing::warn!(%error, "trace spend lookup unavailable");
            Vec::new()
        }
    }
}

pub async fn list_traces(
    client: &Client,
    connection: &Connection,
    access: &ReadAccessParams,
    start_ms: i64,
    end_ms: i64,
    cursor: Option<&str>,
    limit: u32,
) -> Result<TracePage, Error> {
    if limit == 0 {
        return Err(Error::InvalidParameters);
    }
    let (cursor_ms, cursor_trace_id) = trace_position(cursor)?;
    let mut params = ListTracesParams::from(contracts::ListTracesParams {
        access: access.clone(),
        start_ms,
        end_ms,
        cursor_ms,
        cursor_trace_id,
        limit: limit.min(500),
    });
    let page: Vec<contracts::ListTracesRow> = loop {
        match fetch::<RunCandidates>(client, connection, &params).await {
            Err(litellm_storage_clickhouse::Error::ResponseTooLarge) if params.0.limit > 1 => {
                params.0.limit /= 2;
            }
            Err(litellm_storage_clickhouse::Error::ResponseTooLarge) => {
                return Err(Error::ReadTooLarge);
            }
            result => break result?.into_iter().map(|row| row.0).collect(),
        }
    };
    let next_cursor = page
        .last()
        .filter(|_| page.len() == params.0.limit as usize)
        .map(|last| encode_cursor(&(last.start_ms, &last.trace_ref)));
    let data = stream::iter(page.chunks(16))
        .then(|batch| list_summaries(client, connection, access, batch))
        .try_collect::<Vec<_>>()
        .await?
        .into_iter()
        .flatten()
        .collect();
    Ok(TracePage { data, next_cursor })
}

async fn list_summaries(
    client: &Client,
    connection: &Connection,
    access: &ReadAccessParams,
    runs: &[contracts::ListTracesRow],
) -> Result<Vec<litellm_traces::TraceSummary>, Error> {
    let (Some(start_ms), Some(end_ms)) = (
        runs.iter().map(|row| row.start_ms).min(),
        runs.iter()
            .map(|row| row.start_ms.saturating_add(row.duration_ms))
            .max(),
    ) else {
        return Ok(Vec::new());
    };
    let params = TracePageSpansParams::from(contracts::TracePageSpansParams {
        access: access.clone(),
        trace_refs: runs.iter().map(|row| row.trace_ref.clone()).collect(),
        start_ms,
        end_ms: end_ms.saturating_add(1),
    });
    let spans = match crate::span_batches::read_list_spans(client, connection, params).await {
        Ok(spans) => spans,
        Err(Error::ReadTooLarge) => {
            return stream::iter(runs)
                .then(|row| async move {
                    match get_trace(client, connection, access, &row.trace_id, &row.trace_ref).await
                    {
                        Ok(trace) => {
                            Ok(trace.map_or_else(|| listed_summary(row), |trace| trace.summary))
                        }
                        Err(Error::ReadTooLarge) => Ok(listed_summary(row)),
                        Err(error) => Err(error),
                    }
                })
                .try_collect()
                .await;
        }
        Err(error) => return Err(error),
    };
    let by_trace = spans.into_iter().into_group_map_by(|span| {
        (
            span.team_id.clone(),
            span.api_key_hash.clone(),
            span.trace_id.clone(),
        )
    });
    let summaries = runs
        .iter()
        .map(|row| {
            let spans = by_trace
                .get(&(
                    row.team_id.clone(),
                    row.api_key_hash.clone(),
                    row.trace_id.clone(),
                ))
                .map(Vec::as_slice)
                .unwrap_or_default();
            async move {
                let spend_rows = spend(client, connection, access, spans).await;
                resolve_trace(&row.trace_id, &row.trace_ref, spans, &spend_rows)
                    .map_or_else(|| listed_summary(row), |trace| trace.summary)
            }
        })
        .collect::<Vec<_>>();
    Ok(stream::iter(summaries)
        .buffered(SPEND_CONCURRENCY)
        .collect()
        .await)
}

pub async fn get_trace(
    client: &Client,
    connection: &Connection,
    access: &ReadAccessParams,
    trace_id: &str,
    trace_ref: &str,
) -> Result<Option<Trace>, Error> {
    let Some(trace_ref) = reference(client, connection, access, trace_id, trace_ref).await? else {
        return Ok(None);
    };
    let params = TraceSpansParams {
        access: access.clone(),
        trace_id: trace_id.to_owned(),
        trace_ref: trace_ref.clone(),
    };
    let rows = crate::span_batches::read_spans(client, connection, params, u64::MAX).await?;
    if rows.is_empty() {
        return Ok(None);
    }
    let spend_rows = spend(client, connection, access, &rows).await;
    Ok(resolve_trace(trace_id, &trace_ref, &rows, &spend_rows))
}

#[derive(Deserialize, Serialize)]
struct SpanPosition {
    trace_ref: String,
    snapshot_ms: u64,
    offset: usize,
    version: String,
}

pub async fn get_trace_page(
    client: &Client,
    connection: &Connection,
    access: &ReadAccessParams,
    trace_id: &str,
    trace_ref: &str,
    cursor: Option<&str>,
    page_size: u32,
) -> Result<Option<Trace>, Error> {
    if !(1..=500).contains(&page_size) {
        return Err(Error::InvalidParameters);
    }
    let Some(trace_ref) = reference(client, connection, access, trace_id, trace_ref).await? else {
        return Ok(None);
    };
    let position = match cursor {
        Some(cursor) => {
            let position: SpanPosition = decode_cursor(cursor, "span")?;
            if position.trace_ref != trace_ref || position.snapshot_ms == 0 {
                return Err(Error::InvalidCursor("span"));
            }
            position
        }
        None => SpanPosition {
            trace_ref: trace_ref.clone(),
            snapshot_ms: (time::OffsetDateTime::now_utc().unix_timestamp_nanos() / 1_000_000)
                as u64,
            offset: 0,
            version: String::new(),
        },
    };
    let key = SnapshotKey::new(
        connection.url().as_str(),
        access,
        trace_id,
        &trace_ref,
        position.snapshot_ms,
    )
    .map_err(|_| Error::InvalidParameters)?;
    let snapshot = match TRACE_SNAPSHOTS.get(&key).await {
        Some(snapshot) => snapshot,
        None => {
            let params = TraceSpansParams {
                access: access.clone(),
                trace_id: trace_id.to_owned(),
                trace_ref: trace_ref.clone(),
            };
            let rows =
                crate::span_batches::read_spans(client, connection, params, position.snapshot_ms)
                    .await?;
            let spend_rows = spend(client, connection, access, &rows).await;
            let Some(trace) = resolve_trace(trace_id, &trace_ref, &rows, &spend_rows) else {
                return Ok(None);
            };
            TRACE_SNAPSHOTS.insert(key, trace).await?
        }
    };
    let spans = &snapshot.trace().spans;
    if cursor.is_some() && position.version != snapshot.version() {
        return Err(Error::TraceChanged);
    }
    let mut trace = Trace {
        summary: snapshot.trace().summary.clone(),
        agents: snapshot.trace().agents.clone(),
        spans: Vec::new(),
        next_cursor: None,
    };
    if position.offset > spans.len() {
        return Err(Error::InvalidCursor("span"));
    }
    let end = position
        .offset
        .saturating_add(page_size as usize)
        .min(spans.len());
    trace.next_cursor = (end < spans.len()).then(|| {
        encode_cursor(&SpanPosition {
            offset: end,
            version: snapshot.version().to_owned(),
            ..position
        })
    });
    trace.spans = spans[position.offset..end].to_vec();
    while serde_json::to_vec(&trace)
        .map_err(|_| Error::InvalidResponse)?
        .len()
        > litellm_storage_clickhouse::READ_LIMITS.response_bytes
    {
        if trace.spans.len() <= 1 {
            return Err(Error::ReadTooLarge);
        }
        trace.spans.truncate(trace.spans.len() / 2);
        trace.next_cursor = Some(encode_cursor(&SpanPosition {
            trace_ref: trace_ref.clone(),
            snapshot_ms: position.snapshot_ms,
            offset: position.offset + trace.spans.len(),
            version: snapshot.version().to_owned(),
        }));
    }
    Ok(Some(trace))
}

pub async fn get_span(
    client: &Client,
    connection: &Connection,
    access: &ReadAccessParams,
    trace_id: &str,
    span_id: &str,
    trace_ref: &str,
) -> Result<Option<SpanDetail>, Error> {
    let Some(trace_ref) = reference(client, connection, access, trace_id, trace_ref).await? else {
        return Ok(None);
    };
    let params = SpanDetailParams {
        access: access.clone(),
        trace_id: trace_id.to_owned(),
        trace_ref,
        span_id: span_id.to_owned(),
    };
    let row = fetch::<SpanDetailQuery>(client, connection, &params)
        .await?
        .into_iter()
        .next();
    Ok(row.map(|row| SpanDetail {
        input_ui: to_ui_content(&row.input),
        output_ui: to_ui_content(&row.output),
        span_id: row.span_id,
        input: row.input,
        output: row.output,
        attributes: row.attributes,
    }))
}

pub async fn get_span_error(
    client: &Client,
    connection: &Connection,
    access: &ReadAccessParams,
    trace_id: &str,
    span_id: &str,
    trace_ref: &str,
    cursor: Option<&str>,
) -> Result<Option<SpanErrorPage>, Error> {
    let position = error_position(cursor)?;
    let Some(trace_ref) = reference(client, connection, access, trace_id, trace_ref).await? else {
        return Ok(None);
    };
    let offset = position.as_ref().map_or(0, |position| position.offset);
    let params = SpanErrorParams::from(contracts::SpanErrorParams {
        access: access.clone(),
        trace_id: trace_id.to_owned(),
        trace_ref,
        span_id: span_id.to_owned(),
        error_offset: offset,
        error_version: position
            .map(|position| position.version)
            .unwrap_or_default(),
    });
    let Some(row) = fetch::<SpanError>(client, connection, &params)
        .await?
        .into_iter()
        .next()
    else {
        return Ok(None);
    };
    let row = row.0;
    let next_offset = offset + row.message.chars().count() as u64;
    let next_cursor = (next_offset < row.total_chars).then(|| {
        encode_cursor(&ErrorPosition {
            offset: next_offset,
            version: row.version,
        })
    });
    Ok(Some(SpanErrorPage {
        span_id: row.span_id,
        message: row.message,
        total_chars: row.total_chars,
        next_cursor,
    }))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    fn trace_cursor_round_trips_the_last_listed_run() {
        let cursor = encode_cursor(&(1_790_742_989_377_i64, "4bad42b84e9de3ba46fc870185f8f023"));
        assert_eq!(
            trace_position(Some(&cursor)).unwrap(),
            (
                1_790_742_989_377,
                "4bad42b84e9de3ba46fc870185f8f023".to_owned()
            )
        );
        assert_eq!(trace_position(None).unwrap(), (0, String::new()));
        assert_eq!(trace_position(Some("")).unwrap(), (0, String::new()));
    }

    #[rstest]
    #[case::not_base64("abc")]
    #[case::not_json("bm90LWpzb24=")]
    #[case::numeric_reference("WzEsIDJd")]
    #[case::zero_start("WzAsICJ0Il0=")]
    fn malformed_trace_cursors_are_rejected(#[case] cursor: &str) {
        assert!(matches!(
            trace_position(Some(cursor)),
            Err(Error::InvalidCursor("trace"))
        ));
    }

    #[rstest]
    #[case::not_base64("garbage")]
    #[case::missing_fields("e30=")]
    #[case::not_an_object("WzEsMl0=")]
    fn malformed_diagnostic_cursors_are_rejected(#[case] cursor: &str) {
        assert!(matches!(
            error_position(Some(cursor)),
            Err(Error::InvalidCursor("diagnostic"))
        ));
    }

    #[rstest]
    #[case::lowercase_version("a".repeat(64))]
    #[case::short_version("A".repeat(63))]
    fn diagnostic_cursor_requires_a_content_version(#[case] version: String) {
        let cursor = encode_cursor(&ErrorPosition { offset: 1, version });
        assert!(matches!(
            error_position(Some(&cursor)),
            Err(Error::InvalidCursor("diagnostic"))
        ));
    }
}
