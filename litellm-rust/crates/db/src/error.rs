#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("database query failed")]
    Query(#[from] sqlx::Error),
}

impl Error {
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Query(sqlx::Error::Database(error)) => {
                matches!(error.code().as_deref(), Some("40P01" | "55P03" | "40001"))
            }
            Self::Query(
                sqlx::Error::Io(_) | sqlx::Error::PoolTimedOut | sqlx::Error::PoolClosed,
            ) => true,
            Self::Query(_) => false,
        }
    }
}
