use std::ops::Range;

use litellm_http::Client;
use litellm_storage_clickhouse::{Error as StorageError, Query, fetch};
use litellm_traces::{
    QueryScope,
    store::{
        CallQuery, CallRow, RunCount, RunCountQuery, RunQuery, RunRow, RunSelection, RunSortKey,
        SpanQuery, SpanRow, SpanText, SpanTextQuery,
    },
};
use litellm_traces_cache::{StoreError, StoreResult, TraceStore};

use crate::{
    Connection, Error,
    query::named::{
        Calls, CallsParams, RunAttributeCounts, RunCounts, RunCountsParams, RunSpans, RunsPage,
        RunsParams, SpanTextParams, SpanTexts, SpansParams, TraceSpans,
    },
};

/// ClickHouse codes for a read that hit a server-side limit: 158 TOO_MANY_ROWS, 159
/// TIMEOUT_EXCEEDED, 160 TOO_SLOW, 241 MEMORY_LIMIT_EXCEEDED, 307 TOO_MANY_BYTES and 396
/// TOO_MANY_ROWS_OR_BYTES.
const LIMIT_EXCEEDED: [u32; 6] = [158, 159, 160, 241, 307, 396];

const FIRST_RANGE_MS: i64 = 60 * 60 * 1000;

pub struct ClickHouseTraces {
    client: Client,
    connection: Connection,
}

impl ClickHouseTraces {
    pub fn new(client: Client, connection: Connection) -> Self {
        Self { client, connection }
    }

    async fn fetch<Q: Query>(&self, params: &Q::Params) -> StoreResult<Vec<Q::Row>, Error> {
        fetch::<Q>(&self.client, &self.connection, params)
            .await
            .map_err(|error| match error {
                StorageError::ResponseTooLarge => StoreError::TooLarge,
                StorageError::QueryFailed(failure)
                    if failure.code.is_some_and(|code| LIMIT_EXCEEDED.contains(&code)) =>
                {
                    StoreError::TooLarge
                }
                error => StoreError::Failed(Error::Storage(error)),
            })
    }
}

impl ClickHouseTraces {
    async fn runs_page(&self, params: &RunsParams) -> StoreResult<Vec<RunRow>, Error> {
        Ok(self
            .fetch::<RunsPage>(params)
            .await?
            .into_iter()
            .map(|row| row.0)
            .collect())
    }

    /// Reads runs by start time from the window edge the order starts at, over ranges that double
    /// until the page fills, so a page scans the rollups of about as many runs as it returns.
    async fn runs_by_start(
        &self,
        params: &RunsParams,
        query: &RunQuery,
        window: Range<i64>,
    ) -> StoreResult<Vec<RunRow>, Error> {
        let cursor = query.after.as_ref().map(|after| after.value);
        let mut rows = Vec::new();
        let mut range = first_range(&window, cursor, query.order.descending);
        let mut width = FIRST_RANGE_MS;
        while !range.is_empty() && rows.len() < query.limit as usize {
            let remaining = query.limit - rows.len() as u32;
            rows.extend(self.runs_page(&params.within(range.clone(), remaining)).await?);
            width = width.saturating_mul(2);
            range = if query.order.descending {
                window.start.max(range.start.saturating_sub(width))..range.start
            } else {
                range.end..window.end.min(range.end.saturating_add(width))
            };
        }
        Ok(rows)
    }
}

fn first_range(window: &Range<i64>, cursor: Option<i64>, descending: bool) -> Range<i64> {
    if descending {
        let end = cursor.map_or(window.end, |value| value.saturating_add(1).min(window.end));
        window.start.max(end.saturating_sub(FIRST_RANGE_MS))..end
    } else {
        let start = cursor.map_or(window.start, |value| value.max(window.start));
        start..window.end.min(start.saturating_add(FIRST_RANGE_MS))
    }
}

impl TraceStore for ClickHouseTraces {
    type Error = Error;

    fn source(&self) -> &str {
        self.connection.url().as_str()
    }

    async fn runs(&self, access: &QueryScope, query: &RunQuery) -> StoreResult<Vec<RunRow>, Error> {
        let params = RunsParams::new(access, query);
        let mut rows = match &query.selection {
            RunSelection::Matching(filter) if query.order.key == RunSortKey::StartMs => {
                self.runs_by_start(&params, query, filter.start_ms..filter.end_ms)
                    .await?
            }
            _ => self.runs_page(&params).await?,
        };
        rows.sort_by(|left, right| query.order.compare(left, right));
        Ok(rows)
    }

    async fn run_counts(
        &self,
        access: &QueryScope,
        query: &RunCountQuery,
    ) -> StoreResult<Vec<RunCount>, Error> {
        let params = RunCountsParams::new(access, query);
        let rows = if params.counts_attributes() {
            self.fetch::<RunAttributeCounts>(&params).await?
        } else {
            self.fetch::<RunCounts>(&params).await?
        };
        Ok(rows.into_iter().map(|row| row.0).collect())
    }

    async fn spans(
        &self,
        access: &QueryScope,
        query: &SpanQuery,
    ) -> StoreResult<Vec<SpanRow>, Error> {
        let rows = match SpansParams::new(access, query) {
            SpansParams::Trace(params) => self.fetch::<TraceSpans>(&params).await?,
            SpansParams::Runs(params) => self.fetch::<RunSpans>(&params).await?,
        };
        Ok(rows.into_iter().map(|row| row.0).collect())
    }

    async fn span_text(
        &self,
        access: &QueryScope,
        query: &SpanTextQuery,
    ) -> StoreResult<Vec<SpanText>, Error> {
        let rows = self
            .fetch::<SpanTexts>(&SpanTextParams::new(access, query))
            .await?;
        Ok(rows.into_iter().map(|row| row.0).collect())
    }

    async fn calls(
        &self,
        access: &QueryScope,
        query: &CallQuery,
    ) -> StoreResult<Vec<CallRow>, Error> {
        let rows = self
            .fetch::<Calls>(&CallsParams::new(access, query))
            .await?;
        Ok(rows.into_iter().map(|row| row.0).collect())
    }
}
