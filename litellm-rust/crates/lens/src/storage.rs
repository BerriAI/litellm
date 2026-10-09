use crate::Error;
use litellm_http::Client;
use litellm_traces::{QueryScope, ReadQuery, query::named::ReadAccessParams};
use litellm_traces_cache::TraceReader;
use litellm_traces_clickhouse::{ClickHouseTraces, Config, Parameter, QueryReaders};
use serde::Deserialize;
use serde_json::Value;
use std::{collections::BTreeMap, sync::Arc};

pub struct Storage {
    pub config: Config,
    pub client: Client,
    reader: Arc<TraceReader>,
    query_readers: QueryReaders,
    query_secret: String,
}

#[derive(Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub enum Read {
    List {
        scope: ReadAccessParams,
        start_ms: i64,
        end_ms: i64,
        cursor: Option<String>,
        limit: u32,
    },
    Trace {
        scope: ReadAccessParams,
        trace_id: String,
        trace_ref: String,
        cursor: Option<String>,
        page_size: Option<u32>,
    },
    Span {
        scope: ReadAccessParams,
        trace_id: String,
        trace_ref: String,
        span_id: String,
    },
    SpanError {
        scope: ReadAccessParams,
        trace_id: String,
        trace_ref: String,
        span_id: String,
        cursor: Option<String>,
    },
    Query {
        name: String,
        parameters: BTreeMap<String, Parameter>,
    },
    Sql {
        sql: String,
        scope: QueryScope,
    },
    Help {
        scope: QueryScope,
    },
}

fn encode(value: impl serde::Serialize) -> Result<Value, Error> {
    serde_json::to_value(value).map_err(|_| Error::Unavailable)
}

impl Storage {
    pub async fn ping(&self) -> Result<(), Error> {
        litellm_storage_clickhouse::execute_read(
            &self.client,
            self.config.storage().reader(),
            "SELECT 1",
            &BTreeMap::new(),
        )
        .await
        .map_err(litellm_traces_clickhouse::Error::from)?;
        Ok(())
    }

    pub fn new(config: Config, client: Client, query_secret: String) -> Self {
        Self {
            query_readers: QueryReaders::new(
                config.storage().writer().clone(),
                config.storage().database().to_owned(),
            ),
            reader: Arc::new(TraceReader::new(
                litellm_storage_clickhouse::READ_LIMITS.response_bytes,
            )),
            config,
            client,
            query_secret,
        }
    }

    pub async fn ensure_schema(&self) -> Result<(), Error> {
        Ok(litellm_traces_clickhouse::ensure_schema(
            &self.client,
            self.config.storage().writer(),
            self.config.storage().database(),
            self.config.retention_days(),
        )
        .await?)
    }

    pub async fn read(&self, request: Read) -> Result<Value, Error> {
        let store =
            ClickHouseTraces::new(self.client.clone(), self.config.storage().reader().clone());
        match request {
            Read::List {
                scope,
                start_ms,
                end_ms,
                cursor,
                limit,
            } => encode(
                self.reader
                    .list_traces(&store, &scope, start_ms, end_ms, cursor.as_deref(), limit)
                    .await?,
            ),
            Read::Trace {
                scope,
                trace_id,
                trace_ref,
                cursor,
                page_size,
            } => {
                if let Some(page_size) = page_size {
                    return encode(
                        self.reader
                            .get_trace_page(
                                &store,
                                &scope,
                                &trace_id,
                                &trace_ref,
                                cursor.as_deref(),
                                page_size,
                            )
                            .await?,
                    );
                }
                if cursor.is_some() {
                    return Err(Error::InvalidRequest);
                }
                encode(
                    self.reader
                        .get_trace(&store, &scope, &trace_id, &trace_ref)
                        .await?,
                )
            }
            Read::Span {
                scope,
                trace_id,
                trace_ref,
                span_id,
            } => encode(
                self.reader
                    .get_span(&store, &scope, &trace_id, &span_id, &trace_ref)
                    .await?,
            ),
            Read::SpanError {
                scope,
                trace_id,
                trace_ref,
                span_id,
                cursor,
            } => encode(
                self.reader
                    .get_span_error(
                        &store,
                        &scope,
                        &trace_id,
                        &span_id,
                        &trace_ref,
                        cursor.as_deref(),
                    )
                    .await?,
            ),
            Read::Query { name, parameters } => {
                let query = ReadQuery::parse(&name).map_err(|_| Error::InvalidRequest)?;
                let result = litellm_traces_clickhouse::execute_named_read(
                    &self.client,
                    self.config.storage().reader(),
                    query,
                    &parameters,
                )
                .await?;
                serde_json::from_str(&result).map_err(|_| Error::Unavailable)
            }
            Read::Sql { sql, scope } => {
                let _permit = self.query_readers.acquire()?;
                let connection = self
                    .query_readers
                    .connection(&self.client, &scope, &self.query_secret)
                    .await?;
                let result =
                    litellm_traces_clickhouse::query_sql(&self.client, &connection, &sql).await?;
                serde_json::from_str(&result).map_err(|_| Error::Unavailable)
            }
            Read::Help { scope } => {
                let _permit = self.query_readers.acquire()?;
                let connection = self
                    .query_readers
                    .connection(&self.client, &scope, &self.query_secret)
                    .await?;
                encode(litellm_traces_clickhouse::query_help(&self.client, &connection).await?)
            }
        }
    }
}
