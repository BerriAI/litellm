use std::error::Error as StdError;

use litellm_db::{begin, lock_team};
use rstest::rstest;
use sqlx::PgPool;

mod common;
use common::database_pool;

#[rstest]
#[case::transaction_scoped("team-lock-scope")]
#[tokio::test]
async fn team_lock_is_held_until_rollback(
    #[case] team_id: &str,
    #[future(awt)] database_pool: Result<Option<PgPool>, sqlx::Error>,
) -> Result<(), Box<dyn StdError>> {
    let Some(pool) = database_pool? else {
        return Ok(());
    };
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
