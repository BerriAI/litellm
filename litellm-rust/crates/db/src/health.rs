use sqlx::PgExecutor;

use crate::Error;

pub async fn health_check(executor: impl PgExecutor<'_>) -> Result<(), Error> {
    sqlx::query_file_scalar!("queries/health/health_check.sql")
        .fetch_one(executor)
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use rstest::rstest;
    use sqlx::PgPool;

    use super::health_check;
    use crate::test_support::database_pool;

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn answers_when_the_database_is_up(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        health_check(&database_pool?).await?;
        Ok(())
    }
}
