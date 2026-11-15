use std::num::NonZeroU32;

use sqlx::{PgPool, Postgres, Transaction};

use crate::{Error, Statements};

pub async fn begin(
    pool: &PgPool,
    lock_timeout_ms: NonZeroU32,
    named: bool,
) -> Result<Transaction<'static, Postgres>, Error> {
    let mut transaction = pool.begin().await?;
    sqlx::query!(
        "SELECT set_config('lock_timeout', $1::text, true)",
        format!("{lock_timeout_ms}ms")
    )
    .fetch_one(Statements::new(&mut *transaction, named))
    .await?;
    Ok(transaction)
}

pub async fn lock_team(
    transaction: &mut Transaction<'_, Postgres>,
    team_id: &str,
    named: bool,
) -> Result<(), Error> {
    sqlx::query_file!("queries/spend/lock_team.sql", team_id)
        .fetch_one(Statements::new(&mut **transaction, named))
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{error::Error as StdError, num::NonZeroU32};

    use super::{begin, lock_team};
    use crate::test_support::database_pool;
    use rstest::rstest;
    use sqlx::{PgPool, postgres::PgPoolOptions};

    #[rstest]
    #[case::transaction_scoped("team-lock-scope")]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn team_lock_is_held_until_rollback(
        #[case] team_id: &str,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let database_pool = database_pool?;
        let pool = PgPoolOptions::new()
            .max_connections(2)
            .connect_with(database_pool.connect_options().as_ref().clone())
            .await?;
        let mut first = begin(
            &pool,
            NonZeroU32::new(1500).ok_or("1500 is not zero")?,
            true,
        )
        .await?;
        let mut second = pool.begin().await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Test assertions are outside the compile-checked query set"
        )]
        let original_timeout: String = sqlx::query_scalar("SELECT current_setting('lock_timeout')")
            .fetch_one(&mut *second)
            .await?;
        lock_team(&mut first, team_id, true).await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Test assertions are outside the compile-checked query set"
        )]
        let (backend, timeout): (i32, String) =
            sqlx::query_as("SELECT pg_backend_pid(), current_setting('lock_timeout')")
                .fetch_one(&mut *first)
                .await?;
        assert_eq!(timeout, "1500ms");
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
        let mut next = pool.begin().await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Test assertions are outside the compile-checked query set"
        )]
        let (next_backend, next_timeout): (i32, String) =
            sqlx::query_as("SELECT pg_backend_pid(), current_setting('lock_timeout')")
                .fetch_one(&mut *next)
                .await?;
        assert_eq!(next_backend, backend);
        assert_eq!(next_timeout, original_timeout);
        next.rollback().await?;
        second.rollback().await?;
        Ok(())
    }
}
