use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum SourceError {
    #[error("unsupported model cost map URL scheme {scheme}")]
    UnsupportedScheme { scheme: String },
    #[error("invalid model cost map URL: {reason}")]
    Invalid { reason: String },
}
