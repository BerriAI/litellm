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
    #[error("Multiple traces have this ID; provide trace_ref")]
    AmbiguousTrace,
    #[error(transparent)]
    Pagination(#[from] litellm_pagination::Error),
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

impl Error {
    /// The stable public failure code for a read that failed with this error.
    pub fn failure_code(&self) -> litellm_pagination::FailureCode {
        use litellm_pagination::FailureCode;
        use litellm_storage_clickhouse::Error as StorageError;

        match self {
            Self::Pagination(error) => error.code(),
            Self::Cached(source) => source.failure_code(),
            Self::ReadTooLarge
            | Self::InsertTooLarge
            | Self::Decode(litellm_traces::Error::TooLarge)
            | Self::Storage(StorageError::InsertTooLarge | StorageError::ResponseTooLarge) => {
                FailureCode::ResourceTooLarge
            }
            Self::Busy => FailureCode::Busy,
            Self::InvalidRow
            | Self::InvalidLimit(_)
            | Self::InvalidTable
            | Self::InvalidSchema
            | Self::InvalidQuery
            | Self::InvalidParameters
            | Self::InvalidScope
            | Self::AmbiguousTrace
            | Self::Decode(_)
            | Self::Storage(
                StorageError::InvalidRow
                | StorageError::InvalidLimit(_)
                | StorageError::InvalidTable
                | StorageError::InvalidSchema
                | StorageError::EmptySql
                | StorageError::InvalidParameters
                | StorageError::InvalidQuery,
            ) => FailureCode::InvalidRequest,
            Self::InvalidResponse
            | Self::SchemaFailed(_)
            | Self::SchemaTransport
            | Self::MissingSecret
            | Self::ProvisionFailed(_)
            | Self::ProvisionTransport
            | Self::Task
            | Self::Storage(
                StorageError::InvalidUrl
                | StorageError::QueryFailed(_)
                | StorageError::InsertFailed(_)
                | StorageError::SchemaFailed(_)
                | StorageError::InvalidResponse
                | StorageError::Transport,
            ) => FailureCode::Unavailable,
        }
    }
}
