pub const OPENAI_DEFAULT_API_BASE: &str = "https://api.openai.com";
pub const OPENAI_RESPONSES_DEFAULT_API_BASE: &str = "https://api.openai.com/v1";
pub const OPENAI_RESPONSES_PATH: &str = "/responses";

/// Full-request timeout ceiling for Anthropic Messages provider calls, in
/// seconds. Mirrors the Python Anthropic Messages default. The per-request
/// timeout from the caller still overrides this on the request builder.
pub(crate) const MESSAGES_TIMEOUT_SECS: u64 = 600;

/// Connect timeout for Anthropic Messages provider calls, in seconds.
pub(crate) const MESSAGES_CONNECT_TIMEOUT_SECS: u64 = 10;

/// Max characters of an upstream error body echoed across the call boundary
/// before truncation, so provider bodies are bounded and data-minimized.
pub(crate) const MESSAGES_ERROR_BODY_MAX_CHARS: usize = 256;

/// Provider name used for Anthropic Messages when a deployment's provider model
/// does not carry an explicit provider prefix.
pub const ANTHROPIC_MESSAGES_PROVIDER: &str = "anthropic";
