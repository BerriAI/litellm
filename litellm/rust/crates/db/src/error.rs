#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("database query failed")]
    Query(#[from] sqlx::Error),
    #[error("spend batch has {ids} ids and {costs} costs")]
    InvalidSpendBatch { ids: usize, costs: usize },
    #[error("invalid spend timestamp")]
    InvalidTimestamp(#[from] chrono::ParseError),
    #[error("invalid pool duration")]
    InvalidDuration(#[from] std::time::TryFromFloatSecsError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum ErrorKind {
    Unreachable,
    Capacity,
    Deadlock,
    SerializationFailure,
    LockTimeout,
    ReadOnly,
    StalePlan,
    DataRejected,
    Other,
}

impl Error {
    pub fn kind(&self) -> ErrorKind {
        match self {
            Self::Query(sqlx::Error::Io(_)) => ErrorKind::Unreachable,
            Self::Query(sqlx::Error::PoolTimedOut) => ErrorKind::Capacity,
            Self::Query(sqlx::Error::Database(error)) => {
                sqlstate_kind(error.code().as_deref(), error.message())
            }
            _ => ErrorKind::Other,
        }
    }

    pub fn is_retryable(&self) -> bool {
        matches!(
            self.kind(),
            ErrorKind::Unreachable
                | ErrorKind::Capacity
                | ErrorKind::Deadlock
                | ErrorKind::SerializationFailure
                | ErrorKind::LockTimeout
                | ErrorKind::StalePlan
        )
    }

    pub fn needs_fresh_connections(&self) -> bool {
        matches!(self.kind(), ErrorKind::ReadOnly | ErrorKind::StalePlan)
    }

    pub fn is_unavailable(&self) -> bool {
        matches!(self.kind(), ErrorKind::Unreachable | ErrorKind::Capacity)
    }

    pub fn is_requeue_safe(&self) -> bool {
        match self {
            Self::Query(sqlx::Error::PoolTimedOut | sqlx::Error::PoolClosed) => true,
            Self::Query(sqlx::Error::Database(_)) => self.kind() != ErrorKind::DataRejected,
            _ => false,
        }
    }
}

fn sqlstate_kind(code: Option<&str>, message: &str) -> ErrorKind {
    match code {
        Some("0A000") if message.contains("cached plan must not change result type") => {
            ErrorKind::StalePlan
        }
        Some("53300") => ErrorKind::Capacity,
        Some("40P01") => ErrorKind::Deadlock,
        Some("40001") => ErrorKind::SerializationFailure,
        Some("55P03") => ErrorKind::LockTimeout,
        Some("25006") => ErrorKind::ReadOnly,
        Some(code) if code.starts_with("22") || code.starts_with("23") => ErrorKind::DataRejected,
        _ => ErrorKind::Other,
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use rstest::rstest;
    use sqlx::PgPool;

    use super::{Error, ErrorKind};
    use crate::test_support::database_pool;

    #[rstest]
    #[case::closed_pool(sqlx::Error::PoolClosed, ErrorKind::Other, false, false, true)]
    #[case::pool_timeout(sqlx::Error::PoolTimedOut, ErrorKind::Capacity, true, true, true)]
    #[case::io(
        sqlx::Error::Io(std::io::ErrorKind::ConnectionReset.into()),
        ErrorKind::Unreachable,
        true,
        true,
        false
    )]
    #[case::missing_row(sqlx::Error::RowNotFound, ErrorKind::Other, false, false, false)]
    fn classifies_client_errors(
        #[case] error: sqlx::Error,
        #[case] kind: ErrorKind,
        #[case] retryable: bool,
        #[case] unavailable: bool,
        #[case] requeue_safe: bool,
    ) {
        let error = Error::from(error);

        assert_eq!(error.kind(), kind);
        assert_eq!(error.is_retryable(), retryable);
        assert_eq!(error.is_unavailable(), unavailable);
        assert_eq!(error.is_requeue_safe(), requeue_safe);
    }

    #[rstest]
    #[case::connection_capacity("53300", ErrorKind::Capacity, true, true, true)]
    #[case::deadlock("40P01", ErrorKind::Deadlock, true, false, true)]
    #[case::lock_unavailable("55P03", ErrorKind::LockTimeout, true, false, true)]
    #[case::serialization_failure("40001", ErrorKind::SerializationFailure, true, false, true)]
    #[case::read_only_after_failover("25006", ErrorKind::ReadOnly, false, false, true)]
    #[case::unique_violation("23505", ErrorKind::DataRejected, false, false, false)]
    #[case::numeric_overflow("22003", ErrorKind::DataRejected, false, false, false)]
    #[case::undefined_table("42P01", ErrorKind::Other, false, false, true)]
    #[case::feature_not_supported_without_a_stale_plan(
        "0A000",
        ErrorKind::Other,
        false,
        false,
        true
    )]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn classifies_server_errors(
        #[case] code: &str,
        #[case] kind: ErrorKind,
        #[case] retryable: bool,
        #[case] unavailable: bool,
        #[case] requeue_safe: bool,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Test query must raise the requested SQLSTATE"
        )]
        let error = sqlx::query(sqlx::AssertSqlSafe(format!(
            "DO $$ BEGIN RAISE EXCEPTION USING ERRCODE = '{code}'; END $$"
        )))
        .execute(&mut *tx)
        .await
        .expect_err("PostgreSQL must raise the requested error");

        assert_eq!(
            error.as_database_error().and_then(|error| error.code()),
            Some(code.into())
        );
        let error = Error::from(error);
        assert_eq!(error.kind(), kind);
        assert_eq!(error.is_retryable(), retryable);
        assert_eq!(error.is_unavailable(), unavailable);
        assert_eq!(error.is_requeue_safe(), requeue_safe);
        tx.rollback().await?;
        Ok(())
    }
}
