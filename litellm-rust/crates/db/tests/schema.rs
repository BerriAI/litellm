#![cfg(feature = "postgres-tests")]

mod support;

use litellm_db::{Error, REQUIRED_MIGRATION, SchemaCompatibility, schema_compatibility};
use rstest::rstest;
use sqlx::{PgConnection, Postgres, Transaction};
use support::transaction;

const PROBE: &str = "99999999999999_litellm_db_probe";

#[derive(Clone, Copy)]
enum Attempt {
    Running,
    Finished,
    RolledBack,
}

async fn record(connection: &mut PgConnection, id: usize, attempt: Attempt) {
    let (finished, rolled_back) = match attempt {
        Attempt::Running => (false, false),
        Attempt::Finished => (true, false),
        Attempt::RolledBack => (true, true),
    };
    sqlx::query!(
        r#"INSERT INTO _prisma_migrations (id, checksum, migration_name, finished_at, rolled_back_at)
        VALUES ($1, '', $2, CASE WHEN $3::bool THEN now() END, CASE WHEN $4::bool THEN now() END)"#,
        format!("litellm-db-probe-{id}"),
        PROBE,
        finished,
        rolled_back,
    )
    .execute(connection)
    .await
    .unwrap();
}

fn behind() -> SchemaCompatibility {
    SchemaCompatibility::Behind {
        required: PROBE.into(),
    }
}

#[rstest]
#[case::never_applied(&[], behind())]
#[case::still_running(&[Attempt::Running], behind())]
#[case::rolled_back(&[Attempt::RolledBack], behind())]
#[case::finished(&[Attempt::Finished], SchemaCompatibility::Compatible)]
#[case::retried_after_rollback(&[Attempt::RolledBack, Attempt::Finished], SchemaCompatibility::Compatible)]
#[tokio::test]
async fn only_a_finished_attempt_that_was_not_rolled_back_counts(
    #[future(awt)]
    #[from(transaction)]
    mut transaction: Transaction<'static, Postgres>,
    #[case] attempts: &[Attempt],
    #[case] expected: SchemaCompatibility,
) {
    for (id, attempt) in attempts.iter().enumerate() {
        record(&mut transaction, id, *attempt).await;
    }

    let compatibility = schema_compatibility(&mut *transaction, PROBE)
        .await
        .unwrap();

    assert_eq!(compatibility, expected);
}

#[rstest]
#[tokio::test]
async fn a_database_migrated_from_this_tree_is_compatible(
    #[future(awt)]
    #[from(transaction)]
    mut transaction: Transaction<'static, Postgres>,
) {
    let compatibility = schema_compatibility(&mut *transaction, REQUIRED_MIGRATION)
        .await
        .unwrap();

    assert_eq!(compatibility, SchemaCompatibility::Compatible);
}

#[rstest]
#[tokio::test]
async fn a_database_without_prisma_migrations_is_an_error(
    #[future(awt)]
    #[from(transaction)]
    mut transaction: Transaction<'static, Postgres>,
) {
    sqlx::query!("SELECT set_config('search_path', 'litellm_db_probe_empty', true)")
        .fetch_one(&mut *transaction)
        .await
        .unwrap();

    let result = schema_compatibility(&mut *transaction, REQUIRED_MIGRATION).await;

    assert!(matches!(result, Err(Error::Query(_))), "{result:?}");
}
