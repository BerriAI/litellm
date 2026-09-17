use crate::MAX_FRAME_BYTES;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("body stream failed: {0}")]
    Body(#[source] Box<dyn std::error::Error + Send + Sync>),
    #[error("frame exceeds {} bytes", MAX_FRAME_BYTES)]
    FrameTooLarge,
    #[cfg(feature = "aws")]
    #[error("AWS EventStream framing failed: {0}")]
    Aws(#[from] aws_smithy_eventstream::error::Error),
    #[cfg(feature = "aws")]
    #[error("invalid AWS EventStream frame length: {0}")]
    InvalidLength(usize),
    #[cfg(feature = "aws")]
    #[error("truncated AWS EventStream frame")]
    Truncated,
    #[cfg(feature = "sse")]
    #[error("SSE event is not valid UTF-8: {0}")]
    InvalidUtf8(#[source] std::str::Utf8Error),
}
