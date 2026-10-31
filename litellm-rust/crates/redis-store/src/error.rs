type Source = Box<dyn std::error::Error + Send + Sync>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Redis(#[from] litellm_redis_client::Error),
    #[error("the stored value is not in this keyspace's format")]
    Format(#[source] Source),
}

impl Error {
    pub(crate) fn format(source: impl Into<Source>) -> Self {
        Self::Format(source.into())
    }

    pub(crate) fn unexpected(reply: impl std::fmt::Debug) -> Self {
        Self::format(format!("unexpected reply {reply:?}"))
    }
}
