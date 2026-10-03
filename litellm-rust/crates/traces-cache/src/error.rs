#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("trace snapshot serialization failed")]
    Serialization(#[from] serde_json::Error),
    #[error("trace snapshot exceeds the size limit")]
    ReadTooLarge,
}
