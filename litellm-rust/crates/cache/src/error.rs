use litellm_error::Source;

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("cache is unavailable")]
    Unavailable(#[source] Option<Source>),
    #[error("invalid cache entry")]
    InvalidEntry(#[source] Option<Source>),
    #[error("flushing Redis requires an explicit namespace")]
    UnscopedFlush,
    #[error("operation is not supported by this cache")]
    UnsupportedOperation,
    #[error("semantic cache requires request messages")]
    MissingPrompt,
}

#[derive(Debug, thiserror::Error)]
#[error("cache backend returned {returned} results for {requested} keys")]
pub struct ShortBatch {
    pub requested: usize,
    pub returned: usize,
}

#[derive(Debug, thiserror::Error)]
#[error("cache lock is poisoned")]
struct Poisoned;

impl<T> From<std::sync::PoisonError<T>> for Error {
    fn from(_: std::sync::PoisonError<T>) -> Self {
        Self::unavailable(Poisoned)
    }
}

impl Error {
    pub fn unavailable(cause: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::Unavailable(Some(Source::new(cause)))
    }

    pub fn invalid_entry(cause: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::InvalidEntry(Some(Source::new(cause)))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::Error;

    #[rstest]
    #[case::unavailable(
        Error::unavailable(std::io::Error::other("connection refused")),
        "cache is unavailable"
    )]
    #[case::invalid_entry(
        Error::invalid_entry(std::io::Error::other("connection refused")),
        "invalid cache entry"
    )]
    fn the_backend_cause_is_the_source_and_rendered_once(
        #[case] error: Error,
        #[case] message: &str,
    ) {
        let report = litellm_error::Report::new(&error).to_string();
        assert_eq!(report, format!("{message}: connection refused"));
        assert!(!error.to_string().contains("connection refused"));
        let cause = std::error::Error::source(&error)
            .and_then(|cause| cause.downcast_ref::<litellm_error::Source>())
            .expect("cause");
        assert!(cause.downcast_ref::<std::io::Error>().is_some());
    }
}
