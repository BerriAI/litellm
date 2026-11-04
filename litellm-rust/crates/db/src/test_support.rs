use std::error::Error as StdError;

use rstest::fixture;
use sqlx::PgPool;

#[fixture]
pub async fn database_pool() -> Result<PgPool, Box<dyn StdError>> {
    let database_url = std::env::var("DATABASE_URL")?;
    Ok(PgPool::connect(&database_url).await?)
}
