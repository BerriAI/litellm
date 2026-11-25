use litellm_error::Source;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{field} must be an integer millisecond timestamp")]
    NonIntegerTimestamp { field: &'static str },
    #[error("{field} is outside the supported timestamp range")]
    TimestampOutOfRange { field: &'static str, source: Source },
    #[error(transparent)]
    Storage(#[from] litellm_storage_clickhouse::Error),
    #[error(transparent)]
    Migration(#[from] sqlx::migrate::MigrateError),
}

impl Error {
    pub fn is_request(&self) -> bool {
        match self {
            Self::NonIntegerTimestamp { .. } | Self::TimestampOutOfRange { .. } => true,
            Self::Storage(error) => error.is_request(),
            Self::Migration(_) => false,
        }
    }

    pub fn is_insert_too_large(&self) -> bool {
        match self {
            Self::Storage(error) => error.is_insert_too_large(),
            Self::NonIntegerTimestamp { .. }
            | Self::TimestampOutOfRange { .. }
            | Self::Migration(_) => false,
        }
    }
}
