//! Crate-level constants for the ai-gateway.
//!
//! Per `litellm-rust/CLAUDE.md`, magic numbers and fixed strings live here
//! (the Rust mirror of Python's `litellm/constants.py`), not inline in feature
//! modules. Env-overridable tunables keep their `DEFAULT_*` value here; the env
//! read + fallback happens at the host/config layer.

/// Provider attributed to realtime sessions in the logging payload.
#[cfg(feature = "server")]
pub(crate) const DEFAULT_PROVIDER: &str = "openai";

/// HTTP path for the non-streaming Anthropic Messages route.
#[cfg(feature = "server")]
pub(crate) const MESSAGES_ROUTE_PATH: &str = "/v1/messages";

/// Request headers owned by the gateway and never forwarded upstream.
#[cfg(feature = "server")]
pub(crate) const MESSAGES_HEADERS_NOT_FORWARDED: &[&str] =
    &["authorization", "connection", "content-length", "host"];
