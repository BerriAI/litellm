#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("database query failed")]
    Query(#[from] sqlx::Error),
}
