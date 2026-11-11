/// Full-request timeout ceiling for Anthropic Messages provider calls, in
/// seconds. Mirrors the Python Anthropic Messages default. The per-request
/// timeout from the caller still overrides this on the request builder.
pub(crate) const MESSAGES_TIMEOUT_SECS: u64 = 600;

pub(crate) const MESSAGES_HTTP_ERROR_ATTEMPTS: usize = 2;
pub(crate) const MESSAGES_STREAM_ERROR_BODY_READ_SECS: u64 = 5;
