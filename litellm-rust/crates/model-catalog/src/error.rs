use thiserror::Error;

/// Failures from parsing or validating a catalog snapshot.
#[derive(Debug, Error)]
pub enum Error {
    /// The body is not valid JSON, or a model entry fails typed deserialization.
    #[error("invalid JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// The catalog has no entries at all.
    #[error("catalog is empty")]
    Empty,
    /// A non-reserved top level value is not a JSON object.
    #[error("model {model:?} must be an object")]
    EntryNotObject { model: String },
    /// Canonical entry count is under the configured minimum.
    #[error("catalog has {actual} models, below minimum {minimum}")]
    BelowMinimum { actual: usize, minimum: usize },
    /// Canonical entry count is under the configured backup shrink ratio.
    #[error("catalog has {actual} models, below {ratio} of backup count {backup}")]
    Shrunk {
        actual: usize,
        backup: usize,
        ratio: f64,
    },
    /// The configured minimum backup ratio is not finite or outside `[0, 1]`.
    #[error("minimum backup ratio must be finite and between zero and one")]
    InvalidRatio,
}
