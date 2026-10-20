use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    /// The body is not valid JSON, or `fallback_generalizations` has the wrong shape.
    #[error("invalid JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// The catalog has no entries at all.
    #[error("catalog is empty")]
    Empty,
    #[error(transparent)]
    Rejected(#[from] Rejected),
    /// Canonical entry count is under the configured minimum.
    #[error("catalog has {actual} models, below minimum {minimum}")]
    BelowMinimum { actual: usize, minimum: usize },
    /// Canonical entry count is under the configured share of the catalog it replaces.
    #[error("catalog has {actual} models, below {ratio} of reference count {reference}")]
    Shrunk {
        actual: usize,
        reference: usize,
        ratio: f64,
    },
    /// The configured minimum reference ratio is not finite or outside `[0, 1]`.
    #[error("minimum reference ratio must be finite and between zero and one")]
    InvalidRatio,
}

/// Every problem found in a registry or a registration, so one error report lists them all.
#[derive(Debug, Error)]
#[error("{}", .0.iter().map(ToString::to_string).collect::<Vec<_>>().join("; "))]
pub struct Rejected(pub Vec<Problem>);

#[derive(Debug, Error)]
pub enum Problem {
    #[error("{model} must be a JSON object")]
    NotAnObject { model: String },
    #[error("{model}.{key}: {error}")]
    InvalidField {
        model: String,
        key: String,
        #[source]
        error: serde_json::Error,
    },
    #[error("{model}.litellm_provider is required")]
    MissingProvider { model: String },
    #[error("{model}: {first} and {second} price the same meter under the same conditions")]
    DuplicatePrice {
        model: String,
        first: String,
        second: String,
    },
    #[error("{model}.aliases must be an array of strings")]
    InvalidAliases { model: String },
    #[error("{model}: alias {alias:?} is already a model name")]
    AliasShadowsModel { model: String, alias: String },
    #[error("{model}: alias {alias:?} already belongs to {owner}")]
    AliasTaken {
        model: String,
        alias: String,
        owner: String,
    },
}

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq, Hash)]
#[error("key does not name a price rate")]
pub struct InvalidRateKey;

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq, Hash)]
#[error("expected a UTC window HH:MM-HH:MM")]
pub struct InvalidUtcWindow;

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq, Hash)]
#[error("expected an ISO weekday number 1-7 or an English day name")]
pub struct InvalidWeekday;

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq, Hash)]
#[error("expected a calendar date YYYY-MM-DD")]
pub struct InvalidDate;
