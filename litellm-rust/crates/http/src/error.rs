use std::path::PathBuf;

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum Error {
    #[error("{setting} cannot be expressed with rustls: {reason}")]
    Unsupported {
        setting: &'static str,
        reason: String,
    },
    #[error("could not read {}: {message}", path.display())]
    Read { path: PathBuf, message: String },
    #[error("{} is not a PEM file: {message}", path.display())]
    InvalidPem { path: PathBuf, message: String },
    #[error("could not build the HTTP client: {0}")]
    Client(String),
}

impl From<reqwest::Error> for Error {
    fn from(error: reqwest::Error) -> Self {
        Self::Client(error.without_url().to_string())
    }
}
