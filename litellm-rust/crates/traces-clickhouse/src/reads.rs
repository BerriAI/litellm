//! Scoped trace reads: the trace list, one trace resolved with its spend, and span payloads.

use std::collections::BTreeMap;
use std::sync::LazyLock;
use std::time::Duration;

use futures_util::{StreamExt, TryStreamExt, stream};
use itertools::Itertools;
use litellm_http::Client;
use litellm_pagination::{Binding, Cursor, KeyRing, Page, Traversal};
use litellm_storage_clickhouse::{Query, fetch};
use litellm_traces::{
    SpanDetail, SpanErrorPage, SpendLookup, Trace, TracePage, listed_summary,
    query::named as contracts, resolve_trace, to_ui_content,
};
use litellm_traces_cache::{SnapshotCache, SnapshotKey};
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

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
const LIST_RESOURCE: &str = "traces:list";
const DETAIL_RESOURCE: &str = "traces:detail";
const DIAGNOSTIC_RESOURCE: &str = "traces:diagnostic";
const LIST_LIMIT: u32 = 500;
const CURSOR_TTL: time::Duration = time::Duration::minutes(30);
const RESPONSE_BYTES: usize = litellm_storage_clickhouse::READ_LIMITS.response_bytes;

fn now_ms(now: OffsetDateTime) -> u64 {
    (now.unix_timestamp_nanos() / i128::from(NANOS_PER_MS)) as u64
}

fn scope_binding(
    resource: &str,
    access: &ReadAccessParams,
    query: &impl Serialize,
) -> Result<Binding, Error> {
    Ok(Binding::new(
        resource,
        &(&access.user_id, &access.team_ids, access.all_teams),
        query,
    )?)
}

/// A continuation that starts a fresh traversal when no cursor was supplied, or verifies and
/// resumes the pinned one.
fn open_cursor<P: for<'de> Deserialize<'de>>(
    keys: &KeyRing,
    binding: &Binding,
    cursor: Option<&str>,
    first: P,
    now: OffsetDateTime,
) -> Result<Cursor<P>, Error> {
    match cursor.filter(|cursor| !cursor.is_empty()) {
        None => Ok(Cursor {
            position: first,
            revision: String::new(),
            published_ms: now_ms(now),
            expires_at: now + CURSOR_TTL,
        }),
        Some(cursor) => Ok(keys.decode(binding, cursor, now)?),
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct ListPosition {
    start_ms: i64,
    trace_ref: String,
}

/// Where a list traversal starts and which publication instant it is pinned to.
struct ListTraversal {
    cursor: Cursor<ListPosition>,
    binding: Binding,
}

impl ListTraversal {
    fn open(
        keys: &KeyRing,
        access: &ReadAccessParams,
        window: (i64, i64),
        cursor: Option<&str>,
        now: OffsetDateTime,
    ) -> Result<Self, Error> {
        let binding = scope_binding(LIST_RESOURCE, access, &window)?;
        let first = ListPosition {
            start_ms: 0,
            trace_ref: String::new(),
        };
        let resumed = cursor.is_some_and(|cursor| !cursor.is_empty());
        let cursor = open_cursor(keys, &binding, cursor, first, now)?;
        if resumed && (cursor.position.start_ms <= 0 || cursor.position.trace_ref.is_empty()) {
            return Err(litellm_pagination::Error::InvalidCursor.into());
        }
        Ok(Self { cursor, binding })
    }

    fn traversal(&self) -> Traversal {
        Traversal::new(
            &self.binding,
            self.cursor.published_ms,
            self.cursor.expires_at,
        )
    }

    fn continue_after(
        &self,
        keys: &KeyRing,
        starts: &BTreeMap<String, i64>,
        last: &litellm_traces::TraceSummary,
    ) -> Result<String, litellm_pagination::Error> {
        let position = ListPosition {
            start_ms: *starts
                .get(&last.trace_ref)
                .ok_or(litellm_pagination::Error::InvalidCursor)?,
            trace_ref: last.trace_ref.clone(),
        };
        keys.encode(&self.binding, &self.cursor.advance(position))
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct SpanPosition {
    offset: usize,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct ErrorPosition {
    offset: u64,
}

fn valid_error_version(version: &str) -> bool {
    version.len() == 64
        && version
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'A'..=b'F').contains(&byte))
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

#[expect(
    clippy::too_many_arguments,
    reason = "one argument per public read parameter"
)]
pub async fn list_traces(
    client: &Client,
    connection: &Connection,
    keys: &KeyRing,
    access: &ReadAccessParams,
    start_ms: i64,
    end_ms: i64,
    cursor: Option<&str>,
    limit: u32,
) -> Result<TracePage, Error> {
    if limit == 0 || start_ms >= end_ms {
        return Err(Error::InvalidParameters);
    }
    let traversal = ListTraversal::open(
        keys,
        access,
        (start_ms, end_ms),
        cursor,
        OffsetDateTime::now_utc(),
    )?;
    let snapshot_ms = traversal.cursor.published_ms;
    let mut params = ListTracesParams::from(contracts::ListTracesParams {
        access: access.clone(),
        start_ms,
        end_ms,
        cursor_ms: traversal.cursor.position.start_ms,
        cursor_trace_id: traversal.cursor.position.trace_ref.clone(),
        limit: limit.min(LIST_LIMIT),
        snapshot_ms,
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
    let exhausted = page.len() < params.0.limit as usize;
    let starts: BTreeMap<String, i64> = page
        .iter()
        .map(|row| (row.trace_ref.clone(), row.start_ms))
        .collect();
    let items: Vec<litellm_traces::TraceSummary> = stream::iter(page.chunks(16))
        .then(|batch| list_summaries(client, connection, access, batch, snapshot_ms))
        .try_collect::<Vec<_>>()
        .await?
        .into_iter()
        .flatten()
        .collect();
    let next_cursor = items
        .last()
        .filter(|_| !exhausted)
        .map(|last| traversal.continue_after(keys, &starts, last))
        .transpose()?;
    let page = Page {
        items,
        next_cursor,
        traversal: traversal.traversal(),
    };
    Ok(page.bounded(RESPONSE_BYTES, |last| {
        traversal.continue_after(keys, &starts, last)
    })?)
}

async fn list_summaries(
    client: &Client,
    connection: &Connection,
    access: &ReadAccessParams,
    runs: &[contracts::ListTracesRow],
    snapshot_ms: u64,
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
    let spans = match crate::span_batches::read_list_spans(client, connection, params, snapshot_ms)
        .await
    {
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

#[expect(
    clippy::too_many_arguments,
    reason = "one argument per public read parameter"
)]
pub async fn get_trace_page(
    client: &Client,
    connection: &Connection,
    keys: &KeyRing,
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
    let binding = scope_binding(DETAIL_RESOURCE, access, &(trace_id, &trace_ref))?;
    let position = open_cursor(
        keys,
        &binding,
        cursor,
        SpanPosition { offset: 0 },
        OffsetDateTime::now_utc(),
    )?;
    let key = SnapshotKey::new(
        connection.url().as_str(),
        access,
        trace_id,
        &trace_ref,
        position.published_ms,
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
                crate::span_batches::read_spans(client, connection, params, position.published_ms)
                    .await?;
            let spend_rows = spend(client, connection, access, &rows).await;
            let Some(trace) = resolve_trace(trace_id, &trace_ref, &rows, &spend_rows) else {
                return Ok(None);
            };
            TRACE_SNAPSHOTS.insert(key, trace).await?
        }
    };
    if !position.revision.is_empty() {
        position.require_revision(snapshot.version())?;
    }
    let spans = &snapshot.trace().spans;
    if position.position.offset > spans.len() {
        return Err(litellm_pagination::Error::InvalidCursor.into());
    }
    let pinned = Cursor {
        position: position.position,
        revision: snapshot.version().to_owned(),
        published_ms: position.published_ms,
        expires_at: position.expires_at,
    };
    let continue_at = |offset: usize| -> Result<Option<String>, Error> {
        (offset < spans.len())
            .then(|| keys.encode(&binding, &pinned.advance(SpanPosition { offset })))
            .transpose()
            .map_err(Error::from)
    };
    let start = pinned.position.offset;
    let end = start.saturating_add(page_size as usize).min(spans.len());
    let mut trace = Trace {
        summary: snapshot.trace().summary.clone(),
        agents: snapshot.trace().agents.clone(),
        spans: spans[start..end].to_vec(),
        next_cursor: continue_at(end)?,
        traversal: Some(Traversal::new(
            &binding,
            pinned.published_ms,
            pinned.expires_at,
        )),
    };
    while serde_json::to_vec(&trace)
        .map_err(|_| Error::InvalidResponse)?
        .len()
        > RESPONSE_BYTES
    {
        if trace.spans.len() <= 1 {
            return Err(Error::ReadTooLarge);
        }
        trace.spans.truncate(trace.spans.len() / 2);
        trace.next_cursor = continue_at(start + trace.spans.len())?;
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

#[expect(
    clippy::too_many_arguments,
    reason = "one argument per public read parameter"
)]
pub async fn get_span_error(
    client: &Client,
    connection: &Connection,
    keys: &KeyRing,
    access: &ReadAccessParams,
    trace_id: &str,
    span_id: &str,
    trace_ref: &str,
    cursor: Option<&str>,
) -> Result<Option<SpanErrorPage>, Error> {
    let Some(trace_ref) = reference(client, connection, access, trace_id, trace_ref).await? else {
        return Ok(None);
    };
    let binding = scope_binding(
        DIAGNOSTIC_RESOURCE,
        access,
        &(trace_id, span_id, &trace_ref),
    )?;
    let position = open_cursor(
        keys,
        &binding,
        cursor,
        ErrorPosition { offset: 0 },
        OffsetDateTime::now_utc(),
    )?;
    let resumed = !position.revision.is_empty();
    if resumed && !valid_error_version(&position.revision)
        || i64::try_from(position.position.offset).is_err()
    {
        return Err(litellm_pagination::Error::InvalidCursor.into());
    }
    let offset = position.position.offset;
    let params = SpanErrorParams::from(contracts::SpanErrorParams {
        access: access.clone(),
        trace_id: trace_id.to_owned(),
        trace_ref,
        span_id: span_id.to_owned(),
        error_offset: offset,
        error_version: position.revision.clone(),
    });
    let Some(row) = fetch::<SpanError>(client, connection, &params)
        .await?
        .into_iter()
        .next()
    else {
        return Ok(None);
    };
    let row = row.0;
    if resumed {
        position.require_revision(&row.version)?;
    }
    let next_offset = offset + row.message.chars().count() as u64;
    let next_cursor = (next_offset < row.total_chars)
        .then(|| {
            keys.encode(
                &binding,
                &Cursor {
                    position: ErrorPosition {
                        offset: next_offset,
                    },
                    revision: row.version,
                    published_ms: position.published_ms,
                    expires_at: position.expires_at,
                },
            )
        })
        .transpose()?;
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

    fn keys() -> KeyRing {
        KeyRing::new(["list-secret"]).unwrap()
    }

    fn access() -> ReadAccessParams {
        ReadAccessParams {
            all_teams: false,
            user_id: "user-a".into(),
            team_ids: vec!["team-a".into()],
        }
    }

    fn now() -> OffsetDateTime {
        OffsetDateTime::from_unix_timestamp(1_790_000_000).unwrap()
    }

    #[rstest]
    fn list_traversal_pins_publication_and_continues_after_the_last_run() {
        let keys = keys();
        let first = ListTraversal::open(&keys, &access(), (0, 100), None, now()).unwrap();
        assert_eq!(first.cursor.published_ms, 1_790_000_000_000);
        let last = listed_summary(&contracts::ListTracesRow {
            trace_id: "t".into(),
            trace_ref: "4bad42b84e9de3ba46fc870185f8f023".into(),
            team_id: "team-a".into(),
            api_key_hash: String::new(),
            user_id: "user-a".into(),
            name: String::new(),
            service: String::new(),
            input_preview: String::new(),
            status: litellm_traces::SpanStatus::Ok,
            start_ms: 1_790_742_989_377,
            duration_ms: 1,
            span_count: 1,
            agent_count: 0,
            agent_invocations: 0,
            llm_calls: 0,
            tool_calls: 0,
            input_tokens: 0,
            output_tokens: 0,
            models: Vec::new(),
            agent_names: Vec::new(),
            frameworks: Vec::new(),
            error_count: 0,
            request_ids: Vec::new(),
        });
        let starts = BTreeMap::from([(last.trace_ref.clone(), 1_790_742_989_377_i64)]);
        let token = first.continue_after(&keys, &starts, &last).unwrap();
        let next = ListTraversal::open(
            &keys,
            &access(),
            (0, 100),
            Some(&token),
            now() + time::Duration::minutes(5),
        )
        .unwrap();
        assert_eq!(next.cursor.position.start_ms, 1_790_742_989_377);
        assert_eq!(next.cursor.position.trace_ref, last.trace_ref);
        assert_eq!(next.cursor.published_ms, first.cursor.published_ms);
        assert_eq!(next.traversal(), first.traversal());
    }

    #[rstest]
    #[case::other_window((0, 200))]
    fn list_cursors_are_bound_to_their_window(#[case] window: (i64, i64)) {
        let keys = keys();
        let first = ListTraversal::open(&keys, &access(), (0, 100), None, now()).unwrap();
        let token = keys
            .encode(
                &first.binding,
                &first.cursor.advance(ListPosition {
                    start_ms: 5,
                    trace_ref: "r".into(),
                }),
            )
            .unwrap();
        assert!(matches!(
            ListTraversal::open(&keys, &access(), window, Some(&token), now()),
            Err(Error::Pagination(litellm_pagination::Error::InvalidCursor))
        ));
        let other_access = ReadAccessParams {
            team_ids: vec!["team-b".into()],
            ..access()
        };
        assert!(matches!(
            ListTraversal::open(&keys, &other_access, (0, 100), Some(&token), now()),
            Err(Error::Pagination(litellm_pagination::Error::InvalidCursor))
        ));
        assert!(matches!(
            ListTraversal::open(&keys, &access(), (0, 100), Some(&token), now() + CURSOR_TTL),
            Err(Error::Pagination(
                litellm_pagination::Error::TraversalExpired
            ))
        ));
    }

    #[rstest]
    #[case::not_a_token("abc")]
    #[case::legacy_base64_json("WzEsICJ0Il0=")]
    fn malformed_list_cursors_are_invalid(#[case] cursor: &str) {
        assert!(matches!(
            ListTraversal::open(&keys(), &access(), (0, 100), Some(cursor), now()),
            Err(Error::Pagination(litellm_pagination::Error::InvalidCursor))
        ));
    }

    #[rstest]
    #[case::not_a_token("garbage")]
    #[case::legacy_base64_json("e30=")]
    fn malformed_diagnostic_cursors_are_rejected(#[case] cursor: &str) {
        let binding = scope_binding(DIAGNOSTIC_RESOURCE, &access(), &("t", "s", "r")).unwrap();
        assert!(matches!(
            open_cursor::<ErrorPosition>(
                &keys(),
                &binding,
                Some(cursor),
                ErrorPosition { offset: 0 },
                now()
            ),
            Err(Error::Pagination(litellm_pagination::Error::InvalidCursor))
        ));
    }

    #[rstest]
    #[case::lowercase_hex("a".repeat(64), false)]
    #[case::short("A".repeat(63), false)]
    #[case::uppercase_hex("A".repeat(64), true)]
    fn diagnostic_versions_are_uppercase_sha256_hex(#[case] version: String, #[case] valid: bool) {
        assert_eq!(valid_error_version(&version), valid);
    }

    #[rstest]
    fn detail_cursors_are_bound_to_the_trace_and_its_content() {
        let keys = keys();
        let binding = scope_binding(DETAIL_RESOURCE, &access(), &("t", "ref")).unwrap();
        let first = open_cursor(&keys, &binding, None, SpanPosition { offset: 0 }, now()).unwrap();
        let token = keys
            .encode(
                &binding,
                &Cursor {
                    revision: "content-v1".into(),
                    ..first.advance(SpanPosition { offset: 200 })
                },
            )
            .unwrap();
        let resumed: Cursor<SpanPosition> = open_cursor(
            &keys,
            &binding,
            Some(&token),
            SpanPosition { offset: 0 },
            now(),
        )
        .unwrap();
        assert_eq!(resumed.position.offset, 200);
        assert_eq!(resumed.published_ms, first.published_ms);
        assert!(resumed.require_revision("content-v1").is_ok());
        assert!(matches!(
            resumed.require_revision("content-v2"),
            Err(litellm_pagination::Error::TraversalChanged)
        ));
        let other_run = scope_binding(DETAIL_RESOURCE, &access(), &("t", "other-ref")).unwrap();
        assert!(matches!(
            open_cursor::<SpanPosition>(
                &keys,
                &other_run,
                Some(&token),
                SpanPosition { offset: 0 },
                now()
            ),
            Err(Error::Pagination(litellm_pagination::Error::InvalidCursor))
        ));
    }
}
