use futures_util::{TryStreamExt, stream};
use itertools::Itertools;
use litellm_http::Client;
use litellm_storage_clickhouse::{Query, fetch};
use litellm_traces::query::named as contracts;
use serde::Serialize;

use crate::{Connection, Error, query::named::TraceSpansRow};

const PAGE_SIZE: u32 = 256;
pub(crate) const MAX_GRAPH_BYTES: usize = 64 * 1024 * 1024;
const MAX_GRAPH_SPANS: usize = 100_000;

#[derive(Default)]
struct ReadBudget {
    bytes: usize,
    rows: usize,
}

impl ReadBudget {
    fn checked_add(&self, bytes: usize, rows: usize) -> Result<Self, Error> {
        let next = Self {
            bytes: self.bytes.saturating_add(bytes),
            rows: self.rows.saturating_add(rows),
        };
        if next.bytes > MAX_GRAPH_BYTES || next.rows > MAX_GRAPH_SPANS {
            return Err(Error::ReadTooLarge);
        }
        Ok(next)
    }

    fn record(&mut self, row: &impl Serialize) -> Result<(), Error> {
        let bytes = serde_json::to_vec(row).map_err(|_| Error::InvalidResponse)?;
        *self = self.checked_add(bytes.len(), 1)?;
        Ok(())
    }
}

#[derive(Serialize)]
struct Parameters {
    #[serde(flatten)]
    trace: contracts::TraceSpansParams,
    after_span_id: String,
    page_size: u32,
    snapshot_ms: u64,
}

struct SpanBatch;

impl Query for SpanBatch {
    type Params = Parameters;
    type Row = TraceSpansRow;

    const SQL: &'static str = include_str!("../query/trace_span_batch.sql");
}

pub(crate) async fn read_spans(
    client: &Client,
    connection: &Connection,
    trace: contracts::TraceSpansParams,
    snapshot_ms: u64,
) -> Result<Vec<contracts::TraceSpansRow>, Error> {
    let mut parameters = Parameters {
        trace,
        after_span_id: String::new(),
        page_size: PAGE_SIZE,
        snapshot_ms,
    };
    let mut spans = Vec::new();
    let mut budget = ReadBudget::default();
    loop {
        let page = match fetch::<SpanBatch>(client, connection, &parameters).await {
            Err(litellm_storage_clickhouse::Error::ResponseTooLarge)
                if parameters.page_size > 1 =>
            {
                parameters.page_size /= 2;
                continue;
            }
            Err(litellm_storage_clickhouse::Error::ResponseTooLarge) => {
                return Err(Error::ReadTooLarge);
            }
            result => result?,
        };
        let complete = page.len() < parameters.page_size as usize;
        if let Some(last) = page.last() {
            parameters.after_span_id.clone_from(&last.0.span_id);
        }
        for row in page {
            budget.record(&row)?;
            spans.push(row.0);
        }
        if complete {
            spans.sort_by_key(|row| row.start_ns);
            return Ok(spans);
        }
        parameters.page_size = (parameters.page_size * 2).min(PAGE_SIZE);
    }
}

#[derive(Serialize)]
struct ListParameters {
    #[serde(flatten)]
    runs: crate::query::named::TracePageSpansParams,
    after_team: String,
    after_key: String,
    after_trace: String,
    after_span: String,
    page_size: u32,
    snapshot_ms: u64,
}

struct ListSpanBatch;

impl Query for ListSpanBatch {
    type Params = ListParameters;
    type Row = TraceSpansRow;

    const SQL: &'static str = include_str!("../query/trace_list_span_batch.sql");
}

pub(crate) async fn read_list_spans(
    client: &Client,
    connection: &Connection,
    runs: crate::query::named::TracePageSpansParams,
) -> Result<Vec<contracts::TraceSpansRow>, Error> {
    let parameters = ListParameters {
        runs,
        after_team: String::new(),
        after_key: String::new(),
        after_trace: String::new(),
        after_span: String::new(),
        page_size: PAGE_SIZE,
        snapshot_ms: (time::OffsetDateTime::now_utc().unix_timestamp_nanos() / 1_000_000) as u64,
    };
    let pages = stream::try_unfold(
        (Some(parameters), ReadBudget::default()),
        |(parameters, budget)| async move {
            let Some(parameters) = parameters else {
                return Ok(None);
            };
            let page = match fetch::<ListSpanBatch>(client, connection, &parameters).await {
                Err(litellm_storage_clickhouse::Error::ResponseTooLarge)
                    if parameters.page_size > 1 =>
                {
                    let retry = ListParameters {
                        page_size: parameters.page_size / 2,
                        ..parameters
                    };
                    return Ok(Some((Vec::new(), (Some(retry), budget))));
                }
                Err(litellm_storage_clickhouse::Error::ResponseTooLarge) => {
                    return Err(Error::ReadTooLarge);
                }
                result => result?,
            };
            let next = page
                .last()
                .filter(|_| page.len() == parameters.page_size as usize)
                .map(|last| ListParameters {
                    after_team: last.0.team_id.clone(),
                    after_key: last.0.api_key_hash.clone(),
                    after_trace: last.0.trace_id.clone(),
                    after_span: last.0.span_id.clone(),
                    page_size: (parameters.page_size * 2).min(PAGE_SIZE),
                    ..parameters
                });
            let next_budget = page.iter().try_fold(budget, |budget, row| {
                let bytes = serde_json::to_vec(row).map_err(|_| Error::InvalidResponse)?;
                budget.checked_add(bytes.len(), 1)
            })?;
            Ok(Some((page, (next, next_budget))))
        },
    )
    .try_collect::<Vec<_>>()
    .await?;
    Ok(pages
        .into_iter()
        .flatten()
        .map(|row| row.0)
        .sorted_by_key(|row| row.start_ns)
        .collect())
}

#[derive(Serialize)]
struct SpendParameters {
    #[serde(flatten)]
    lookup: crate::query::named::SpendByResponseIdsParams,
    has_cursor: u8,
    after_team: String,
    after_ms: i64,
    after_id: String,
    page_size: u32,
}

struct SpendBatch;

impl Query for SpendBatch {
    type Params = SpendParameters;
    type Row = crate::query::named::SpendByResponseIdsRow;

    const SQL: &'static str = include_str!("../query/spend_batch.sql");
}

pub(crate) async fn read_spend(
    client: &Client,
    connection: &Connection,
    lookup: crate::query::named::SpendByResponseIdsParams,
) -> Result<Vec<contracts::SpendByResponseIdsRow>, Error> {
    let mut parameters = SpendParameters {
        lookup,
        has_cursor: 0,
        after_team: String::new(),
        after_ms: 0,
        after_id: String::new(),
        page_size: PAGE_SIZE,
    };
    let mut rows = Vec::new();
    let mut budget = ReadBudget::default();
    loop {
        let page = match fetch::<SpendBatch>(client, connection, &parameters).await {
            Err(litellm_storage_clickhouse::Error::ResponseTooLarge)
                if parameters.page_size > 1 =>
            {
                parameters.page_size /= 2;
                continue;
            }
            Err(litellm_storage_clickhouse::Error::ResponseTooLarge) => {
                return Err(Error::ReadTooLarge);
            }
            result => result?,
        };
        let complete = page.len() < parameters.page_size as usize;
        if let Some(last) = page.last() {
            parameters.has_cursor = 1;
            parameters.after_team.clone_from(&last.0.team_id);
            parameters.after_ms = last.0.start_ms;
            parameters.after_id.clone_from(&last.0.request_id);
        }
        for row in page {
            budget.record(&row)?;
            rows.push(row.0);
        }
        if complete {
            return Ok(rows);
        }
        parameters.page_size = (parameters.page_size * 2).min(PAGE_SIZE);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::byte_boundary(MAX_GRAPH_BYTES - 1, 0, 1, false)]
    #[case::byte_overflow(MAX_GRAPH_BYTES - 1, 0, 2, true)]
    #[case::integer_overflow(MAX_GRAPH_BYTES, 0, usize::MAX, true)]
    #[case::row_boundary(0, MAX_GRAPH_SPANS - 1, 1, false)]
    #[case::row_overflow(0, MAX_GRAPH_SPANS, 1, true)]
    fn accumulation_stops_at_the_graph_budget(
        #[case] bytes: usize,
        #[case] rows: usize,
        #[case] next: usize,
        #[case] rejected: bool,
    ) {
        let budget = ReadBudget { bytes, rows };
        assert_eq!(budget.checked_add(next, 1).is_err(), rejected);
    }
}
