#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("key expiration must be in the future")]
    InvalidExpiration,
    #[error("key already exists")]
    AlreadyExists,
    #[error("key storage unavailable")]
    Storage(#[source] litellm_error::Source),
    #[error("key generation unavailable")]
    Entropy(#[source] rand::Error),
}
