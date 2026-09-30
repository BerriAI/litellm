#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid ClickHouse HTTP URL")]
    InvalidUrl,
    #[error("trace query requires start before end and a limit from 1 to 100")]
    InvalidListQuery,
    #[error("trace query requires nonempty identifiers")]
    InvalidIdentifier,
    #[error("trace queries require the ClickHouse table schema")]
    SchemaPending,
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
