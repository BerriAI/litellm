#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SourceError {
    #[error("unsupported catalog source scheme: {scheme}")]
    UnsupportedScheme { scheme: String },
    #[error("invalid catalog source: {reason}")]
    Invalid { reason: String },
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("catalog refresh superseded by a newer install")]
pub struct Stale;
