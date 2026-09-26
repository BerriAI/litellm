#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("running a query")]
    Query(#[from] sqlx::Error),
}
