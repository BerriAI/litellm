use std::path::PathBuf;

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum Error {
    #[error("could not read {}: {message}", path.display())]
    Read { path: PathBuf, message: String },
    #[error("{} is not a PEM file: {message}", path.display())]
    InvalidPem { path: PathBuf, message: String },
    #[error("could not build the HTTP client: {0}")]
    Client(String),
    #[error("request body could not be serialized: {0}")]
    RequestBody(String),
    #[error("request forwards a header the signer computes: {0}")]
    ComputedHeader(String),
    #[error("request signing failed: {0}")]
    Signature(String),
}

impl From<reqwest::Error> for Error {
    fn from(error: reqwest::Error) -> Self {
        Self::Client(error.without_url().to_string())
    }
}
