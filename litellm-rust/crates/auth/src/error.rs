use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AuthServiceError {
    Lookup { source: String },
    CallerToken,
    Headers,
}

impl fmt::Display for AuthServiceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lookup { source } => write!(formatter, "{source} lookup failed"),
            Self::CallerToken => formatter.write_str("caller token failed"),
            Self::Headers => formatter.write_str("header access failed"),
        }
    }
}

impl std::error::Error for AuthServiceError {}
