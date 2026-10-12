#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid ClickHouse insert row")]
    InvalidRow,
    #[error("{0} must be a positive integer")]
    InvalidLimit(&'static str),
    #[error("database must be a nonempty SQL identifier and retention must be positive")]
    InvalidSchema,
    #[error("ClickHouse insert exceeds the encoded size limit")]
    InsertTooLarge,
    #[error(transparent)]
    Storage(#[from] litellm_storage_clickhouse::Error),
    #[error(transparent)]
    Migration(#[from] sqlx::migrate::MigrateError),
}
