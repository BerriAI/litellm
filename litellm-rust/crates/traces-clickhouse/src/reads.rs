//! Scoped trace reads: the trace list, one trace resolved with its spend, and span payloads.

use std::collections::HashMap;

use base64::{Engine, engine::general_purpose::URL_SAFE};
use litellm_http::Client;
use litellm_storage_clickhouse::fetch;
use litellm_traces::{
    SpanDetail, SpanErrorPage, SpendLookup, Trace, TracePage, listed_summary,
    query::named as contracts, resolve_trace, to_ui_content,
};
use serde::{Deserialize, Serialize};

use crate::{
    Connection, Error,
    query::named::{
        ListTraces, ListTracesParams, ReadAccessParams, SpanDetail as SpanDetailQuery,
        SpanDetailParams, SpanError, SpanErrorParams, SpendByResponseIds, SpendByResponseIdsParams,
        TraceIdentity, TraceIdentityParams, TracePageSpans, TracePageSpansParams, TraceSpans,
        TraceSpansParams,
    },
};

const NANOS_PER_MS: i64 = 1_000_000;
const SPEND_WINDOW_MS: i64 = 30 * 60 * 1000;

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
    match fetch::<SpendByResponseIds>(client, connection, &params).await {
        Ok(rows) => rows.into_iter().map(|row| row.0).collect(),
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
    let (cursor_ms, cursor_trace_id) = trace_position(cursor)?;
    let params = ListTracesParams::from(contracts::ListTracesParams {
        access: access.clone(),
        start_ms,
        end_ms,
        cursor_ms,
        cursor_trace_id,
        limit,
    });
    let page: Vec<contracts::ListTracesRow> = fetch::<ListTraces>(client, connection, &params)
        .await?
        .into_iter()
        .map(|row| row.0)
        .collect();
    let next_cursor = page
        .last()
        .filter(|_| page.len() == limit as usize)
        .map(|last| encode_cursor(&(last.start_ms, &last.trace_ref)));
    let (Some(page_start), Some(page_end)) = (
        page.iter().map(|row| row.start_ms).min(),
        page.iter().map(|row| row.start_ms + row.duration_ms).max(),
    ) else {
        return Ok(TracePage {
            data: Vec::new(),
            next_cursor,
        });
    };
    let span_params = TracePageSpansParams::from(contracts::TracePageSpansParams {
        access: access.clone(),
        trace_refs: page.iter().map(|row| row.trace_ref.clone()).collect(),
        start_ms: page_start,
        end_ms: page_end + 1,
    });
    let span_rows: Vec<contracts::TraceSpansRow> =
        fetch::<TracePageSpans>(client, connection, &span_params)
            .await?
            .into_iter()
            .map(|row| row.0)
            .collect();
    let spend_rows = spend(client, connection, access, &span_rows).await;
    let mut by_trace: HashMap<(String, String, String), Vec<contracts::TraceSpansRow>> =
        HashMap::new();
    for span in span_rows {
        let key = (
            span.team_id.clone(),
            span.api_key_hash.clone(),
            span.trace_id.clone(),
        );
        by_trace.entry(key).or_default().push(span);
    }
    let data = page
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
            resolve_trace(&row.trace_id, &row.trace_ref, spans, &spend_rows)
                .map_or_else(|| listed_summary(row), |trace| trace.summary)
        })
        .collect();
    Ok(TracePage { data, next_cursor })
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
    let rows: Vec<contracts::TraceSpansRow> = fetch::<TraceSpans>(client, connection, &params)
        .await?
        .into_iter()
        .map(|row| row.0)
        .collect();
    if rows.is_empty() {
        return Ok(None);
    }
    let spend_rows = spend(client, connection, access, &rows).await;
    Ok(resolve_trace(trace_id, &trace_ref, &rows, &spend_rows))
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
