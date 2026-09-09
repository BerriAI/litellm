//! Crate-level constants for the ai-gateway.
//!
//! Per `litellm-rust/CLAUDE.md`, magic numbers and fixed strings live here
//! (the Rust mirror of Python's `litellm/constants.py`), not inline in feature
//! modules. Env-overridable tunables keep their `DEFAULT_*` value here; the env
//! read + fallback happens at the host/config layer.

/// Default LiteLLM control-plane base URL for request-log egress when
/// `LITELLM_PROXY_BASE_URL` is unset.
pub(crate) const DEFAULT_PROXY_BASE_URL: &str = "http://localhost:4000";

/// The logs ingest path appended to the proxy base. Not a tunable; it is the
/// proxy's API contract (the rust-control-plane router on the Python proxy).
pub(crate) const RUST_CONTROL_PLANE_LOGS_PATH: &str = "/v1/rust_control_plane/logs";

/// Default bounded channel depth for the log-egress worker.
/// Override: `LITELLM_LOG_CHANNEL_CAPACITY`.
pub(crate) const DEFAULT_CHANNEL_CAPACITY: usize = 4096;

/// Default max records POSTed per request to the control plane.
/// Override: `LITELLM_LOG_BATCH_SIZE`.
pub(crate) const DEFAULT_MAX_BATCH_SIZE: usize = 256;

/// Default partial-batch flush cadence, in ms.
/// Override: `LITELLM_LOG_FLUSH_INTERVAL_MS`.
pub(crate) const DEFAULT_FLUSH_INTERVAL_MS: u64 = 500;

/// Provider attributed to realtime sessions in the logging payload.
#[cfg(feature = "server")]
pub(crate) const DEFAULT_PROVIDER: &str = "openai";

pub(crate) const DEFAULT_RESPONSES_WS_CONNECT_TIMEOUT_SECS: u64 = 10;
pub(crate) const DEFAULT_RESPONSES_WS_IDLE_TIMEOUT_SECS: u64 = 300;

/// HTTP path for the non-streaming Anthropic Messages route.
#[cfg(feature = "server")]
pub(crate) const MESSAGES_ROUTE_PATH: &str = "/v1/messages";

/// Request headers owned by the gateway and never forwarded upstream.
#[cfg(feature = "server")]
pub(crate) const MESSAGES_HEADERS_NOT_FORWARDED: &[&str] =
    &["authorization", "connection", "content-length", "host"];

/// Response header carrying the wall time the gateway spent admitting a request.
#[cfg(feature = "server")]
pub(crate) const ADMISSION_DURATION_HEADER: &str = "x-litellm-admission-duration-ms";

/// Largest `/v1/messages` body accepted before parsing. Override: `LITELLM_MAX_REQUEST_BYTES`.
#[cfg(feature = "server")]
pub(crate) const DEFAULT_MAX_REQUEST_BYTES: usize = 32 * 1024 * 1024;

/// Largest admitted input token count. Override: `LITELLM_MAX_INPUT_TOKENS`.
#[cfg(feature = "server")]
pub(crate) const DEFAULT_MAX_INPUT_TOKENS: usize = 1_000_000;

/// Concurrent tokenizer runs on the blocking pool. Override: `LITELLM_TOKENIZER_CONCURRENCY`.
#[cfg(feature = "server")]
pub(crate) const DEFAULT_TOKENIZER_CONCURRENCY: usize = 2;

/// Inputs at or under this size are tokenized inline; larger ones go to the blocking pool.
#[cfg(feature = "server")]
pub(crate) const TOKENIZE_INLINE_MAX_BYTES: usize = 16 * 1024;

/// Bytes per token used when no tokenizer file is configured.
#[cfg(feature = "server")]
pub(crate) const APPROX_BYTES_PER_TOKEN: usize = 4;

/// Input price used to reserve budget before the provider reports usage (USD per token).
#[cfg(feature = "server")]
pub(crate) const DEFAULT_INPUT_COST_PER_TOKEN: f64 = 3e-6;

/// The Python proxy endpoint that resolves a virtual key to its limits.
#[cfg(feature = "server")]
pub(crate) const PROXY_KEY_INFO_PATH: &str = "/key/info";

/// Timeout for a virtual-key lookup against the Python proxy.
#[cfg(feature = "server")]
pub(crate) const KEY_INFO_TIMEOUT_SECS: u64 = 5;
