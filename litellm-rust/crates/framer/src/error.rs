#[cfg(feature = "sse")]
#[derive(Debug, thiserror::Error)]
pub enum SseError {
    #[error("body stream failed")]
    Body(#[from] std::io::Error),
    #[error("SSE field is not UTF-8")]
    InvalidUtf8(#[from] std::str::Utf8Error),
}

#[cfg(feature = "aws")]
#[derive(Debug, thiserror::Error)]
pub enum EventStreamError {
    #[error("body stream failed")]
    Body(#[from] std::io::Error),
    #[error("invalid AWS EventStream frame length: {0}")]
    InvalidLength(usize),
    #[error("truncated AWS EventStream frame")]
    Truncated,
    #[error("malformed AWS EventStream frame")]
    Malformed(#[from] aws_smithy_eventstream::error::Error),
}
