use std::path::Path;

use litellm_db_testing::{MigratedPostgres, PRISMA_MIGRATIONS_DIR, prisma_migrations};
use rstest::rstest;

#[rstest]
#[tokio::test]
async fn real_prisma_migrations_apply_to_a_fresh_postgres() {
    let postgres = MigratedPostgres::start().await.unwrap();

    #[expect(
        clippy::disallowed_methods,
        reason = "the test schema has no offline query data"
    )]
    let applied: i64 = sqlx::query_scalar("SELECT count(*) FROM _sqlx_migrations WHERE success")
        .fetch_one(postgres.pool())
        .await
        .unwrap();

    let expected = prisma_migrations(Path::new(PRISMA_MIGRATIONS_DIR))
        .unwrap()
        .len();
    assert_eq!(applied as usize, expected);
}
