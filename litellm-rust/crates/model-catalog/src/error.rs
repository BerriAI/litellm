use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid catalog JSON")]
    Json(#[from] serde_json::Error),
    #[error("catalog is empty")]
    Empty,
    #[error("model {model:?} must be an object")]
    EntryNotObject { model: String },
    #[error("catalog has {actual} models, below minimum {minimum}")]
    BelowMinimum { actual: usize, minimum: usize },
    #[error("catalog has {actual} models, below {ratio} of reference count {reference}")]
    Shrunk {
        actual: usize,
        reference: usize,
        ratio: f64,
    },
    #[error("minimum reference ratio must be finite and between zero and one")]
    InvalidRatio,
}
