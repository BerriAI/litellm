#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ApiUrlError {
    #[error("invalid URL: {0}")]
    Parse(#[from] url::ParseError),
    #[error("URL cannot be used as a base")]
    CannotBeBase,
    #[error("unsupported URL scheme: {0}")]
    Scheme(String),
    #[error("URL path segment cannot be . or ..")]
    DotSegment,
}
