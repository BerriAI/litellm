/// Full-request timeout ceiling for chat completions provider calls, in
/// seconds. Mirrors the Python chat completions default.
pub(crate) const CHAT_COMPLETIONS_TIMEOUT_SECS: u64 = 600;

/// `object` field every non-streaming chat completion response carries.
pub const CHAT_COMPLETION_OBJECT: &str = "chat.completion";
