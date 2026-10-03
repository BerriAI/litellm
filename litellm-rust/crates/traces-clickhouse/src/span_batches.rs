use litellm_http::Client;
use litellm_storage_clickhouse::{Query, fetch};
use litellm_traces::query::named as contracts;
use serde::Serialize;

use crate::{Connection, Error, query::named::TraceSpansRow};

const PAGE_SIZE: u32 = 256;
const MAX_GRAPH_BYTES: usize = 64 * 1024 * 1024;
const MAX_GRAPH_SPANS: usize = 100_000;

#[derive(Default)]
struct ReadBudget {
    bytes: usize,
    rows: usize,
}

impl ReadBudget {
    fn reserve(&mut self, bytes: usize) -> Result<(), Error> {
        self.bytes = self.bytes.saturating_add(bytes);
        if self.bytes > MAX_GRAPH_BYTES || self.rows == MAX_GRAPH_SPANS {
            return Err(Error::ReadTooLarge);
        }
        self.rows += 1;
        Ok(())
    }

    fn record(&mut self, row: &impl Serialize) -> Result<(), Error> {
        let bytes = serde_json::to_vec(row).map_err(|_| Error::InvalidResponse)?;
        self.reserve(bytes.len())
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
        let mut budget = ReadBudget { bytes, rows };
        assert_eq!(budget.reserve(next).is_err(), rejected);
    }
}
