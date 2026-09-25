pub const OPENAI_DEFAULT_API_BASE: &str = "https://api.openai.com";

/// Full-request timeout ceiling for Anthropic Messages provider calls, in
/// seconds. Mirrors the Python Anthropic Messages default. The per-request
/// timeout from the caller still overrides this on the request builder.
pub(crate) const MESSAGES_TIMEOUT_SECS: u64 = 600;

/// Provider name used for Anthropic Messages when a deployment's provider model
/// does not carry an explicit provider prefix.
pub const ANTHROPIC_MESSAGES_PROVIDER: &str = "anthropic";

/// Full-request timeout ceiling for chat completions provider calls, in
/// seconds. Mirrors the Python chat completions default.
pub(crate) const CHAT_COMPLETIONS_TIMEOUT_SECS: u64 = 600;

/// Connect timeout for chat completions provider calls, in seconds.
pub(crate) const CHAT_COMPLETIONS_CONNECT_TIMEOUT_SECS: u64 = 10;

pub(crate) const AUDIO_TRANSCRIPTION_TIMEOUT_SECS: u64 = 600;

/// `object` field every non-streaming chat completion response carries.
pub const CHAT_COMPLETION_OBJECT: &str = "chat.completion";
