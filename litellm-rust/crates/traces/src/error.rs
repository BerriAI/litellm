#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("invalid OTLP trace payload")]
    InvalidPayload,
    #[error("OTLP trace payload exceeds the decoding budget")]
    TooLarge,
    #[error("OTLP token count is outside the storage range")]
    TokenCountOutOfRange,
}

#[derive(Debug, thiserror::Error)]
pub enum QueryAccessError {
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
    #[error(transparent)]
    Storage(#[from] litellm_storage_clickhouse::Error),
    #[error(transparent)]
    Cached(#[from] std::sync::Arc<QueryAccessError>),
}
