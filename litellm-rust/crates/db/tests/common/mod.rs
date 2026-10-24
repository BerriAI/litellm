use rstest::fixture;
use sqlx::PgPool;

#[fixture]
pub async fn database_pool() -> Result<Option<PgPool>, sqlx::Error> {
    let Ok(database_url) = std::env::var("DATABASE_URL") else {
        return Ok(None);
    };
    PgPool::connect(&database_url).await.map(Some)
}
