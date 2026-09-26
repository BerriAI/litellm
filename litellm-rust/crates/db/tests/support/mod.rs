use litellm_db_testing::MigratedPostgres;
use rstest::fixture;
use sqlx::PgPool;

pub struct Database {
    pub pool: PgPool,
    _postgres: MigratedPostgres,
}

#[fixture]
pub async fn database() -> Database {
    let postgres = MigratedPostgres::start().await.unwrap();
    let pool = PgPool::connect(postgres.url()).await.unwrap();
    Database {
        pool,
        _postgres: postgres,
    }
}
