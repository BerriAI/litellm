#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[cfg(feature = "sse")]
    #[error("SSE framing failed: {0}")]
    Sse(#[from] sse_stream::Error),
    #[cfg(feature = "aws")]
    #[error("AWS EventStream framing failed: {0}")]
    Aws(#[from] aws_smithy_eventstream::error::Error),
    #[error("body stream failed: {0}")]
    Body(#[source] Box<dyn std::error::Error + Send + Sync>),
    #[cfg(feature = "aws")]
    #[error("invalid AWS EventStream frame length: {0}")]
    InvalidLength(usize),
    #[cfg(feature = "aws")]
    #[error("truncated AWS EventStream frame")]
    Truncated,
}
