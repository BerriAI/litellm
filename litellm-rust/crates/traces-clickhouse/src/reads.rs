use litellm_http::Client;
use litellm_storage_clickhouse::{Error as StorageError, Query, fetch};
use litellm_traces::{
    QueryScope,
    store::{
        CallQuery, CallRow, RunCount, RunCountQuery, RunQuery, RunRow, SpanQuery, SpanRow,
        SpanText, SpanTextQuery,
    },
};
use litellm_traces_cache::{StoreError, StoreResult, TraceStore};

use crate::{
    Connection, Error,
    query::named::{
        Calls, CallsParams, RunCounts, RunCountsParams, RunSpans, Runs, RunsParams, SpanTextParams,
        SpanTexts, SpansParams, TraceSpans,
    },
};

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
                error => StoreError::Failed(Error::Storage(error)),
            })
    }
}

impl TraceStore for ClickHouseTraces {
    type Error = Error;

    fn source(&self) -> &str {
        self.connection.url().as_str()
    }

    async fn runs(&self, access: &QueryScope, query: &RunQuery) -> StoreResult<Vec<RunRow>, Error> {
        let rows = self.fetch::<Runs>(&RunsParams::new(access, query)).await?;
        Ok(rows.into_iter().map(|row| row.0).collect())
    }

    async fn run_counts(
        &self,
        access: &QueryScope,
        query: &RunCountQuery,
    ) -> StoreResult<Vec<RunCount>, Error> {
        let rows = self
            .fetch::<RunCounts>(&RunCountsParams::new(access, query))
            .await?;
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
    ) -> StoreResult<Option<SpanText>, Error> {
        let rows = self
            .fetch::<SpanTexts>(&SpanTextParams::new(access, query))
            .await?;
        Ok(rows.into_iter().next().map(|row| row.0))
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
