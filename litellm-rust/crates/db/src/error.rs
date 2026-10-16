#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid database URL")]
    Url(#[source] sqlx::Error),
    #[error("database query failed")]
    Query(#[from] sqlx::Error),
}
