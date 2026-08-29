pub const OPENAI_DEFAULT_API_BASE: &str = "https://api.openai.com";
pub const OPENAI_RESPONSES_DEFAULT_API_BASE: &str = "https://api.openai.com/v1";
pub const OPENAI_RESPONSES_PATH: &str = "/responses";

pub(crate) const HTTP_CLIENT_TIMEOUT_SECS: u64 = 600;
pub(crate) const HTTP_CLIENT_CONNECT_TIMEOUT_SECS: u64 = 10;

/// Max characters of an upstream error body echoed across the call boundary
/// before truncation, so provider bodies are bounded and data-minimized.
pub(crate) const UPSTREAM_ERROR_BODY_MAX_CHARS: usize = 256;

/// Provider name used for Anthropic Messages when a deployment's provider model
/// does not carry an explicit provider prefix.
pub const ANTHROPIC_MESSAGES_PROVIDER: &str = "anthropic";

/// Prefix identifying an Anthropic OAuth token. Mirrors Python's
/// `ANTHROPIC_OAUTH_TOKEN_PREFIX`, which is what makes `validate_environment`
/// authenticate with `authorization` and drop `x-api-key` entirely.
pub(crate) const ANTHROPIC_OAUTH_TOKEN_PREFIX: &str = "sk-ant-oat";

/// `object` field every non-streaming chat completion response carries.
pub const CHAT_COMPLETION_OBJECT: &str = "chat.completion";

/// Placeholder Python substitutes for empty or whitespace-only message text,
/// which Anthropic and Bedrock both reject. Must match
/// `_EMPTY_TEXT_PLACEHOLDER` in
/// `litellm/litellm_core_utils/prompt_templates/factory.py`.
pub const EMPTY_TEXT_PLACEHOLDER: &str =
    "[System: Empty message content sanitised to satisfy protocol]";
