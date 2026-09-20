#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("cache is unavailable")]
    Unavailable,
    #[error("invalid cache entry")]
    InvalidEntry,
}
