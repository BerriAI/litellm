use sqlx::{PgPool, Postgres, Transaction};

use crate::Error;

pub async fn begin(pool: &PgPool) -> Result<Transaction<'static, Postgres>, Error> {
    let mut transaction = pool.begin().await?;
    sqlx::query!("SELECT set_config('lock_timeout', '5000ms', true)")
        .fetch_one(&mut *transaction)
        .await?;
    Ok(transaction)
}

pub async fn lock_team(
    transaction: &mut Transaction<'_, Postgres>,
    team_id: &str,
) -> Result<(), Error> {
    sqlx::query_file!("queries/spend/lock_team.sql", team_id)
        .fetch_one(&mut **transaction)
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use super::{begin, lock_team};
    use crate::test_support::database_pool;
    use rstest::rstest;
    use sqlx::PgPool;

    #[rstest]
    #[case::transaction_scoped("team-lock-scope")]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn team_lock_is_held_until_rollback(
        #[case] team_id: &str,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut first = begin(&pool).await?;
        let mut second = pool.begin().await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Test assertions are outside the compile-checked query set"
        )]
        let timeout: String = sqlx::query_scalar("SELECT current_setting('lock_timeout')")
            .fetch_one(&mut *first)
            .await?;
        assert_eq!(timeout, "5s");
        lock_team(&mut first, team_id).await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Test assertions are outside the compile-checked query set"
        )]
        let held: bool = sqlx::query_scalar("SELECT pg_try_advisory_xact_lock(hashtext($1))")
            .bind(team_id)
            .fetch_one(&mut *second)
            .await?;
        assert!(!held);
        first.rollback().await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Test assertions are outside the compile-checked query set"
        )]
        let released: bool = sqlx::query_scalar("SELECT pg_try_advisory_xact_lock(hashtext($1))")
            .bind(team_id)
            .fetch_one(&mut *second)
            .await?;
        assert!(released);
        second.rollback().await?;
        Ok(())
    }
}
