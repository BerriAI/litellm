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
pub(crate) const CLOUD_PLATFORM_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";

pub(crate) const VERTEXAI_CREDENTIALS_ENV: &str = "VERTEXAI_CREDENTIALS";

pub(crate) const VERTEX_CREDENTIALS_CACHE_CAPACITY: usize = 64;

pub(crate) const ENV_REFERENCE_PREFIX: &str = "os.environ/";

pub(crate) const DEFAULT_OCR_REQUEST_TIMEOUT_SECS: u64 = 600;

pub(crate) const OCR_ERROR_BODY_MAX_CHARS: usize = 256;

pub(crate) const AZURE_DOCUMENT_INTELLIGENCE_POLL_TIMEOUT_SECS: u64 = 120;

pub(crate) const DEFAULT_MAX_IMAGE_URL_DOWNLOAD_SIZE_MB: f64 = 50.0;

pub(crate) const MAX_SAFE_FETCH_REDIRECTS: usize = 10;

/// Full-request timeout ceiling for Anthropic Messages provider calls, in
/// seconds. Mirrors the Python Anthropic Messages default. The per-request
/// timeout from `litellm_params` still overrides this on the request builder.
pub(crate) const MESSAGES_TIMEOUT_SECS: u64 = 600;

/// Connect timeout for Anthropic Messages provider calls, in seconds.
pub(crate) const MESSAGES_CONNECT_TIMEOUT_SECS: u64 = 10;

/// Max characters of an upstream error body echoed across the host boundary
/// before truncation, so provider bodies are bounded and data-minimized.
pub(crate) const MESSAGES_ERROR_BODY_MAX_CHARS: usize = 256;

pub(crate) const DEFAULT_RESPONSES_WS_CONNECT_TIMEOUT_SECS: u64 = 10;
pub(crate) const DEFAULT_RESPONSES_WS_IDLE_TIMEOUT_SECS: u64 = 300;

/// HTTP path for the non-streaming Anthropic Messages route.
#[cfg(feature = "server")]
pub(crate) const MESSAGES_ROUTE_PATH: &str = "/v1/messages";

/// Provider name used by the Anthropic Messages route when a deployment's
/// provider model does not carry an explicit provider prefix.
pub(crate) const ANTHROPIC_MESSAGES_PROVIDER: &str = "anthropic";

/// Request headers owned by the gateway and never forwarded upstream.
#[cfg(feature = "server")]
pub(crate) const MESSAGES_HEADERS_NOT_FORWARDED: &[&str] =
    &["authorization", "connection", "content-length", "host"];
