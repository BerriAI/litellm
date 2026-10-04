#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid ClickHouse insert row")]
    InvalidRow,
    #[error("{0} must be a positive integer")]
    InvalidLimit(&'static str),
    #[error("invalid ClickHouse insert table")]
    InvalidTable,
    #[error("invalid ClickHouse HTTP URL")]
    InvalidUrl,
    #[error("database must be a nonempty SQL identifier and retention must be positive")]
    InvalidSchema,
    #[error("SQL query must not be empty")]
    EmptySql,
    #[error("invalid ClickHouse query parameters")]
    InvalidParameters,
    #[error("unknown ClickHouse read query")]
    InvalidQuery,
    #[error("ClickHouse query failed with HTTP status {0}")]
    QueryFailed(u16),
    #[error("ClickHouse insert failed with HTTP status {0}")]
    InsertFailed(u16),
    #[error("ClickHouse insert exceeds the encoded size limit")]
    InsertTooLarge,
    #[error("ClickHouse schema setup failed with HTTP status {0}")]
    SchemaFailed(u16),
    #[error("ClickHouse query exceeded the response size limit")]
    ResponseTooLarge,
    #[error("ClickHouse returned an invalid or failed JSON query response")]
    InvalidResponse,
    #[error("ClickHouse query transport failed")]
    Transport,
}
