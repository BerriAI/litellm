#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("database query failed")]
    Query(#[from] sqlx::Error),
    #[error("spend batch has {ids} ids and {costs} costs")]
    InvalidSpendBatch { ids: usize, costs: usize },
    #[error("invalid spend timestamp")]
    InvalidTimestamp(#[from] chrono::ParseError),
}

impl Error {
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Query(sqlx::Error::Database(error)) => {
                matches!(
                    error.code().as_deref(),
                    Some("40P01" | "55P03" | "40001" | "53300")
                )
            }
            Self::Query(sqlx::Error::Io(_) | sqlx::Error::PoolTimedOut) => true,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use rstest::rstest;
    use sqlx::PgPool;

    use super::Error;
    use crate::test_support::database_pool;

    #[rstest]
    #[case::closed_pool(sqlx::Error::PoolClosed, false)]
    #[case::pool_timeout(sqlx::Error::PoolTimedOut, true)]
    #[case::io(sqlx::Error::Io(std::io::ErrorKind::ConnectionReset.into()), true)]
    #[case::missing_row(sqlx::Error::RowNotFound, false)]
    fn classifies_client_errors(#[case] error: sqlx::Error, #[case] retryable: bool) {
        assert_eq!(Error::from(error).is_retryable(), retryable);
    }

    #[rstest]
    #[case::connection_capacity("53300", true)]
    #[case::deadlock("40P01", true)]
    #[case::lock_unavailable("55P03", true)]
    #[case::serialization_failure("40001", true)]
    #[case::unique_violation("23505", false)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn classifies_server_errors(
        #[case] code: &str,
        #[case] retryable: bool,
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
        assert_eq!(Error::from(error).is_retryable(), retryable);
        tx.rollback().await?;
        Ok(())
    }
}
