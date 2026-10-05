//! Keyset-paged reads that shrink their page when ClickHouse rejects a response as too large and
//! stop accumulating once a graph exceeds the interactive budget.

use std::{future::Future, marker::PhantomData};

use litellm_http::Client;
use litellm_storage_clickhouse::{Query, fetch};
use litellm_traces::query::named as contracts;
use litellm_traces_cache::{MAX_GRAPH_BYTES, MAX_GRAPH_SPANS, StoreError};
use serde::{Serialize, de::DeserializeOwned};

use crate::{
    Connection, Error,
    query::named::{SpendByResponseIdsParams, SpendByResponseIdsRow, TraceSpansRow},
};

const PAGE_SIZE: u32 = 256;

#[derive(Default)]
struct ReadBudget {
    bytes: usize,
    rows: usize,
}

impl ReadBudget {
    fn reserve(&mut self, bytes: usize) -> Result<(), StoreError<Error>> {
        self.bytes = self.bytes.saturating_add(bytes);
        if self.bytes > MAX_GRAPH_BYTES || self.rows == MAX_GRAPH_SPANS {
            return Err(StoreError::TooLarge);
        }
        self.rows += 1;
        Ok(())
    }

    fn record(&mut self, row: &impl Serialize) -> Result<(), StoreError<Error>> {
        let bytes =
            serde_json::to_vec(row).map_err(|_| StoreError::Failed(Error::InvalidResponse))?;
        self.reserve(bytes.len())
    }
}

/// One keyset position in a paged query: the SQL reads the cursor fields of `Self` plus the
/// `page_size` that [`Batch`] adds.
trait Keyset: Serialize + Sized + Send + Sync {
    type Row: Serialize + DeserializeOwned + Send;
    const SQL: &'static str;

    fn after(self, last: &Self::Row) -> Self;
}

#[derive(Serialize)]
struct Batch<K> {
    #[serde(flatten)]
    keyset: K,
    page_size: u32,
}

trait PageSource<K: Keyset> {
    fn page(
        &self,
        batch: &Batch<K>,
    ) -> impl Future<Output = Result<Vec<K::Row>, litellm_storage_clickhouse::Error>> + Send;
}

struct Paged<K>(PhantomData<K>);

impl<K: Keyset> Query for Paged<K> {
    type Params = Batch<K>;
    type Row = K::Row;
    const SQL: &'static str = K::SQL;
}

/// Reads every row after `keyset`. A page ClickHouse rejects as too large is retried at half the
/// size, and the smaller page is kept for the rest of the read because row sizes within one graph
/// rarely shrink again. Halving a one-row page means a single row exceeds the response limit.
async fn read_all<K: Keyset, S: PageSource<K>>(
    source: &S,
    keyset: K,
) -> Result<Vec<K::Row>, StoreError<Error>> {
    let mut batch = Batch {
        keyset,
        page_size: PAGE_SIZE,
    };
    let mut rows = Vec::new();
    let mut budget = ReadBudget::default();
    loop {
        let page = match source.page(&batch).await {
            Err(litellm_storage_clickhouse::Error::ResponseTooLarge) if batch.page_size > 1 => {
                batch.page_size /= 2;
                continue;
            }
            Err(litellm_storage_clickhouse::Error::ResponseTooLarge) => {
                return Err(StoreError::TooLarge);
            }
            result => result.map_err(|error| StoreError::Failed(Error::Storage(error)))?,
        };
        let complete = page.len() < batch.page_size as usize;
        for row in &page {
            budget.record(row)?;
        }
        if let Some(last) = page.last() {
            batch.keyset = batch.keyset.after(last);
        }
        rows.extend(page);
        if complete {
            return Ok(rows);
        }
    }
}

struct ClickHouse<'a> {
    client: &'a Client,
    connection: &'a Connection,
}

impl<K: Keyset> PageSource<K> for ClickHouse<'_> {
    fn page(
        &self,
        batch: &Batch<K>,
    ) -> impl Future<Output = Result<Vec<K::Row>, litellm_storage_clickhouse::Error>> + Send {
        fetch::<Paged<K>>(self.client, self.connection, batch)
    }
}

async fn read_paged<K: Keyset>(
    client: &Client,
    connection: &Connection,
    keyset: K,
) -> Result<Vec<K::Row>, StoreError<Error>> {
    let source = ClickHouse { client, connection };
    read_all(&source, keyset).await
}

fn by_start(mut rows: Vec<contracts::TraceSpansRow>) -> Vec<contracts::TraceSpansRow> {
    rows.sort_by_key(|row| row.start_ns);
    rows
}

#[derive(Serialize)]
struct SpanKeyset {
    #[serde(flatten)]
    trace: contracts::TraceSpansParams,
    after_span_id: String,
    snapshot_ms: u64,
}

impl Keyset for SpanKeyset {
    type Row = TraceSpansRow;
    const SQL: &'static str = include_str!("../query/trace_span_batch.sql");

    fn after(self, last: &TraceSpansRow) -> Self {
        Self {
            after_span_id: last.0.span_id.clone(),
            ..self
        }
    }
}

pub(crate) async fn read_spans(
    client: &Client,
    connection: &Connection,
    trace: contracts::TraceSpansParams,
    snapshot_ms: u64,
) -> Result<Vec<contracts::TraceSpansRow>, StoreError<Error>> {
    let keyset = SpanKeyset {
        trace,
        after_span_id: String::new(),
        snapshot_ms,
    };
    let rows = read_paged(client, connection, keyset).await?;
    Ok(by_start(rows.into_iter().map(|row| row.0).collect()))
}

#[derive(Serialize)]
struct ListSpanKeyset {
    #[serde(flatten)]
    runs: crate::query::named::TracePageSpansParams,
    after_team: String,
    after_key: String,
    after_trace: String,
    after_span: String,
    snapshot_ms: u64,
}

impl Keyset for ListSpanKeyset {
    type Row = TraceSpansRow;
    const SQL: &'static str = include_str!("../query/trace_list_span_batch.sql");

    fn after(self, last: &TraceSpansRow) -> Self {
        Self {
            after_team: last.0.team_id.clone(),
            after_key: last.0.api_key_hash.clone(),
            after_trace: last.0.trace_id.clone(),
            after_span: last.0.span_id.clone(),
            ..self
        }
    }
}

pub(crate) async fn read_list_spans(
    client: &Client,
    connection: &Connection,
    runs: crate::query::named::TracePageSpansParams,
    snapshot_ms: u64,
) -> Result<Vec<contracts::TraceSpansRow>, StoreError<Error>> {
    let keyset = ListSpanKeyset {
        runs,
        after_team: String::new(),
        after_key: String::new(),
        after_trace: String::new(),
        after_span: String::new(),
        snapshot_ms,
    };
    let rows = read_paged(client, connection, keyset).await?;
    Ok(by_start(rows.into_iter().map(|row| row.0).collect()))
}

#[derive(Serialize)]
struct SpendKeyset {
    #[serde(flatten)]
    lookup: SpendByResponseIdsParams,
    has_cursor: u8,
    after_team: String,
    after_ms: i64,
    after_id: String,
}

impl Keyset for SpendKeyset {
    type Row = SpendByResponseIdsRow;
    const SQL: &'static str = include_str!("../query/spend_batch.sql");

    fn after(self, last: &SpendByResponseIdsRow) -> Self {
        Self {
            has_cursor: 1,
            after_team: last.0.team_id.clone(),
            after_ms: last.0.start_ms,
            after_id: last.0.request_id.clone(),
            ..self
        }
    }
}

pub(crate) async fn read_spend(
    client: &Client,
    connection: &Connection,
    lookup: SpendByResponseIdsParams,
) -> Result<Vec<contracts::SpendByResponseIdsRow>, StoreError<Error>> {
    let keyset = SpendKeyset {
        lookup,
        has_cursor: 0,
        after_team: String::new(),
        after_ms: 0,
        after_id: String::new(),
    };
    let rows = read_paged(client, connection, keyset).await?;
    Ok(rows.into_iter().map(|row| row.0).collect())
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use rstest::rstest;

    use super::*;

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

    #[derive(Serialize)]
    struct Numbers {
        after: u32,
    }

    impl Keyset for Numbers {
        type Row = u32;
        const SQL: &'static str = "";

        fn after(self, last: &u32) -> Self {
            Self { after: *last }
        }
    }

    /// A table of `total` rows whose transport rejects any page larger than `largest_page`.
    struct Table {
        total: u32,
        largest_page: u32,
        requests: Mutex<Vec<u32>>,
    }

    impl PageSource<Numbers> for Table {
        async fn page(
            &self,
            batch: &Batch<Numbers>,
        ) -> Result<Vec<u32>, litellm_storage_clickhouse::Error> {
            self.requests.lock().unwrap().push(batch.page_size);
            if batch.page_size > self.largest_page {
                return Err(litellm_storage_clickhouse::Error::ResponseTooLarge);
            }
            let end = (batch.keyset.after + batch.page_size).min(self.total);
            Ok((batch.keyset.after + 1..=end).collect())
        }
    }

    #[rstest]
    #[case::fits(1000, PAGE_SIZE, &[256, 256, 256, 256])]
    #[case::uniform_large_rows(1000, 100, &[256, 128, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64])]
    #[tokio::test]
    async fn a_rejected_page_size_is_not_retried(
        #[case] total: u32,
        #[case] largest_page: u32,
        #[case] requests: &[u32],
    ) {
        let table = Table {
            total,
            largest_page,
            requests: Mutex::new(Vec::new()),
        };
        let rows = read_all(&table, Numbers { after: 0 }).await.unwrap();
        assert_eq!(rows, (1..=total).collect::<Vec<_>>());
        assert_eq!(table.requests.lock().unwrap().as_slice(), requests);
    }

    #[rstest]
    #[tokio::test]
    async fn a_single_oversized_row_fails_the_read() {
        let table = Table {
            total: 10,
            largest_page: 0,
            requests: Mutex::new(Vec::new()),
        };
        let result = read_all(&table, Numbers { after: 0 }).await;
        assert!(matches!(result, Err(StoreError::TooLarge)), "{result:?}");
        assert_eq!(
            table.requests.lock().unwrap().as_slice(),
            &[256, 128, 64, 32, 16, 8, 4, 2, 1]
        );
    }
}
