#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid ClickHouse insert row")]
    InvalidRow,
    #[error("invalid ClickHouse HTTP URL")]
    InvalidUrl,
    #[error("database must be a nonempty SQL identifier and retention must be positive")]
    InvalidSchema,
    #[error("SQL query must not be empty")]
    EmptySql,
    #[error("ClickHouse query failed with HTTP status {0}")]
    QueryFailed(u16),
    #[error("ClickHouse query exceeded the response size limit")]
    ResponseTooLarge,
    #[error("ClickHouse returned an invalid or failed JSON query response")]
    InvalidResponse,
    #[error("ClickHouse query transport failed")]
    Transport,
}
