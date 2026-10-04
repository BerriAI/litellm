use litellm_http::Client;
use litellm_storage_clickhouse::{Error as StorageError, Query, fetch};
use litellm_traces::query::named as contracts;
use litellm_traces_cache::{StoreError, TraceStore};

use crate::{
    Connection, Error,
    query::named::{
        ListTracesParams, ListTracesRow, SpanDetail as SpanDetailQuery, SpanError, SpanErrorParams,
        SpendByResponseIdsParams, TraceIdentity, TracePageSpansParams,
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

pub struct ClickHouseTraces {
    client: Client,
    connection: Connection,
}

impl ClickHouseTraces {
    pub fn new(client: Client, connection: Connection) -> Self {
        Self { client, connection }
    }
}

impl TraceStore for ClickHouseTraces {
    type Error = Error;

    fn source(&self) -> &str {
        self.connection.url().as_str()
    }

    async fn trace_refs(
        &self,
        params: &contracts::TraceIdentityParams,
    ) -> Result<Vec<String>, StoreError<Self::Error>> {
        fetch::<TraceIdentity>(&self.client, &self.connection, params)
            .await
            .map(|rows| rows.into_iter().map(|row| row.trace_ref).collect())
            .map_err(failed)
    }

    async fn list_runs(
        &self,
        params: &contracts::ListTracesParams,
    ) -> Result<Vec<contracts::ListTracesRow>, StoreError<Self::Error>> {
        let storage_params = ListTracesParams::from(params.clone());
        match fetch::<RunCandidates>(&self.client, &self.connection, &storage_params).await {
            Ok(rows) => Ok(rows.into_iter().map(|row| row.0).collect()),
            Err(StorageError::ResponseTooLarge) => Err(StoreError::TooLarge),
            Err(error) => Err(failed(error)),
        }
    }

    async fn trace_spans(
        &self,
        params: &contracts::TraceSpansParams,
        snapshot_ms: u64,
    ) -> Result<Vec<contracts::TraceSpansRow>, StoreError<Self::Error>> {
        crate::span_batches::read_spans(&self.client, &self.connection, params.clone(), snapshot_ms)
            .await
    }

    async fn run_spans(
        &self,
        params: &contracts::TracePageSpansParams,
        snapshot_ms: u64,
    ) -> Result<Vec<contracts::TraceSpansRow>, StoreError<Self::Error>> {
        crate::span_batches::read_list_spans(
            &self.client,
            &self.connection,
            TracePageSpansParams::from(params.clone()),
            snapshot_ms,
        )
        .await
    }

    async fn spend(
        &self,
        params: &contracts::SpendByResponseIdsParams,
    ) -> Result<Vec<contracts::SpendByResponseIdsRow>, StoreError<Self::Error>> {
        crate::span_batches::read_spend(
            &self.client,
            &self.connection,
            SpendByResponseIdsParams::from(params.clone()),
        )
        .await
    }

    async fn span_detail(
        &self,
        params: &contracts::SpanDetailParams,
    ) -> Result<Option<contracts::SpanDetailRow>, StoreError<Self::Error>> {
        match fetch::<SpanDetailQuery>(&self.client, &self.connection, params).await {
            Ok(rows) => Ok(rows.into_iter().next()),
            Err(error) => Err(failed(error)),
        }
    }

    async fn span_error(
        &self,
        params: &contracts::SpanErrorParams,
    ) -> Result<Option<contracts::SpanErrorRow>, StoreError<Self::Error>> {
        let storage_params = SpanErrorParams::from(params.clone());
        match fetch::<SpanError>(&self.client, &self.connection, &storage_params).await {
            Ok(rows) => Ok(rows.into_iter().next().map(|row| row.0)),
            Err(error) => Err(failed(error)),
        }
    }
}

fn failed(error: StorageError) -> StoreError<Error> {
    StoreError::Failed(Error::Storage(error))
}
