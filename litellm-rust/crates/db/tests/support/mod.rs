use rstest::fixture;
use sqlx::{PgPool, Postgres, Transaction};

#[fixture]
pub async fn transaction() -> Transaction<'static, Postgres> {
    let url = std::env::var("DATABASE_URL")
        .expect("DATABASE_URL must point at a Postgres migrated by prisma migrate deploy");
    PgPool::connect(&url).await.unwrap().begin().await.unwrap()
}
