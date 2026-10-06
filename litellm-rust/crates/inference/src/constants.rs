pub const OPENAI_DEFAULT_API_BASE: &str = "https://api.openai.com";

/// Full-request timeout ceiling for Anthropic Messages provider calls, in
/// seconds. Mirrors the Python Anthropic Messages default. The per-request
/// timeout from the caller still overrides this on the request builder.
pub(crate) const MESSAGES_TIMEOUT_SECS: u64 = 600;

/// Full-request timeout ceiling for chat completions provider calls, in
/// seconds. Mirrors the Python chat completions default.
pub(crate) const CHAT_COMPLETIONS_TIMEOUT_SECS: u64 = 600;

pub(crate) const AUDIO_TRANSCRIPTION_TIMEOUT_SECS: u64 = 600;

/// `object` field every non-streaming chat completion response carries.
pub const CHAT_COMPLETION_OBJECT: &str = "chat.completion";
