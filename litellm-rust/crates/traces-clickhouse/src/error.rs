#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid ClickHouse insert row")]
    InvalidRow,
    #[error("{0} must be a positive integer")]
    InvalidLimit(&'static str),
    #[error("invalid ClickHouse insert table")]
    InvalidTable,
    #[error("database must be a nonempty SQL identifier and retention must be positive")]
    InvalidSchema,
    #[error("unknown ClickHouse read query")]
    InvalidQuery,
    #[error("invalid ClickHouse query parameters")]
    InvalidParameters,
    #[error("ClickHouse returned an invalid or failed JSON query response")]
    InvalidResponse,
    #[error("ClickHouse insert exceeds the encoded size limit")]
    InsertTooLarge,
    #[error("Trace exceeds the interactive read budget; use a filtered trace query")]
    ReadTooLarge,
    #[error("ClickHouse schema setup failed with HTTP status {0}")]
    SchemaFailed(u16),
    #[error("ClickHouse schema setup transport failed")]
    SchemaTransport,
    #[error("trace SQL queries require a configured proxy master key")]
    MissingSecret,
    #[error("invalid trace query scope")]
    InvalidScope,
    #[error("trace SQL query concurrency limit exceeded")]
    Busy,
    #[error(
        "ClickHouse reader provisioning failed with HTTP status {0}; the configured connection must be allowed to manage users, row policies, and SELECT grants on the trace tables"
    )]
    ProvisionFailed(u16),
    #[error("ClickHouse reader provisioning transport failed")]
    ProvisionTransport,
    #[error("Invalid {0} cursor")]
    InvalidCursor(&'static str),
    #[error("Multiple traces have this ID; provide trace_ref")]
    AmbiguousTrace,
    #[error("Trace changed while paging; refresh the trace to continue")]
    TraceChanged,
    #[error(transparent)]
    Decode(#[from] litellm_traces::Error),
    #[error("trace ingestion task failed")]
    Task,
    #[error(transparent)]
    Storage(#[from] litellm_storage_clickhouse::Error),
    #[error(transparent)]
    Cached(#[from] std::sync::Arc<Error>),
}

impl From<litellm_traces_cache::Error> for Error {
    fn from(error: litellm_traces_cache::Error) -> Self {
        match error {
            litellm_traces_cache::Error::Serialization(_) => Self::InvalidResponse,
            litellm_traces_cache::Error::ReadTooLarge => Self::ReadTooLarge,
        }
    }
}
