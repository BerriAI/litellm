use crate::DecodeError;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Redis(#[from] litellm_redis_client::Error),
    #[error(transparent)]
    Memory(#[from] litellm_cache::Error),
    #[error(transparent)]
    Decode(#[from] DecodeError),
    #[error("the value is not JSON serializable: {0}")]
    Encode(#[from] litellm_python_compat::Error),
}

impl Error {
    pub fn kind(&self) -> ErrorKind {
        match self {
            Self::Redis(error) => error.into(),
            Self::Memory(_) => ErrorKind::Memory,
            Self::Decode(_) => ErrorKind::Decode,
            Self::Encode(_) => ErrorKind::Encode,
        }
    }
}

/// The failure class a host reports with a store error, `timeout` and `breaker_open` being the ones a
/// caller may drop.
#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
pub enum ErrorKind {
    #[strum(serialize = "breaker_open")]
    BreakerOpen,
    #[strum(serialize = "timeout")]
    Timeout,
    #[strum(serialize = "transport")]
    Transport,
    #[strum(serialize = "rejected")]
    Rejected,
    #[strum(serialize = "decode")]
    Decode,
    #[strum(serialize = "encode")]
    Encode,
    #[strum(serialize = "memory")]
    Memory,
}

impl From<&Error> for ErrorKind {
    fn from(error: &Error) -> Self {
        error.kind()
    }
}

impl From<&litellm_redis_client::Error> for ErrorKind {
    fn from(error: &litellm_redis_client::Error) -> Self {
        use litellm_redis_client::Error;
        match error {
            Error::BreakerOpen => Self::BreakerOpen,
            Error::Transport(error) if error.is_timeout() => Self::Timeout,
            Error::Transport(_) => Self::Transport,
            Error::Server(_) => Self::Rejected,
            Error::Decode(_) => Self::Decode,
        }
    }
}
