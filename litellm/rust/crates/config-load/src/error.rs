use litellm_coroutine::Abandoned;

use crate::Location;

#[derive(Debug, thiserror::Error)]
pub enum Error<E> {
    #[error("'include' must be a list of file paths")]
    IncludeNotPaths,
    #[error("config {0} is not a YAML mapping")]
    NotMapping(Location),
    #[error("the host stopped answering the config load")]
    Abandoned,
    #[error("the host could not read a config")]
    Host(#[source] E),
}

impl<E> From<Abandoned> for Error<E> {
    fn from(_: Abandoned) -> Self {
        Self::Abandoned
    }
}
