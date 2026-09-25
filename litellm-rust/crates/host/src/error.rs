#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum MachineFault {
    #[error("host driver abandoned an operation")]
    Abandoned,
    #[error("host driver protocol error: {0}")]
    Protocol(#[from] litellm_coroutine::ResumeError),
    #[error("host does not support {0}")]
    Unsupported(&'static str),
}
