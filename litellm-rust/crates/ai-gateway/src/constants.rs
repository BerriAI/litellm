//! Crate-level constants for the ai-gateway.
//!
//! Per `litellm-rust/CLAUDE.md`, magic numbers and fixed strings live here
//! (the Rust mirror of Python's `litellm/constants.py`), not inline in feature
//! modules. Env-overridable tunables keep their `DEFAULT_*` value here; the env
//! read + fallback happens at the host/config layer.

use litellm_core::constants::{
    MIME_APPLICATION_OCTET_STREAM, MIME_APPLICATION_PDF, MIME_BINARY_OCTET_STREAM, MIME_IMAGE_BMP,
    MIME_IMAGE_GIF, MIME_IMAGE_JPEG, MIME_IMAGE_PNG, MIME_IMAGE_TIFF, MIME_IMAGE_WEBP,
};

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
pub(crate) const DEFAULT_PROVIDER: &str = "openai";

pub(crate) const DEFAULT_UPLOAD_MIME_TYPE: &str = MIME_APPLICATION_OCTET_STREAM;

pub(crate) const GENERIC_UPLOAD_MIME_TYPES: &[&str] =
    &[MIME_APPLICATION_OCTET_STREAM, MIME_BINARY_OCTET_STREAM];

pub(crate) const OCR_RESERVED_PARAM_KEYS: &[&str] = &[
    "api_key",
    "api_base",
    "custom_llm_provider",
    "extra_headers",
    "vertex_credentials",
    "vertex_ai_credentials",
    "vertex_project",
    "vertex_ai_project",
    "vertex_location",
    "vertex_ai_location",
];

pub(crate) const OCR_MULTIPART_UNIQUE_FIELDS: &[&str] = &["model", "timeout", "document"];

pub(crate) const MAX_OCR_REQUEST_BYTES: usize = 100 * 1024 * 1024;

pub(crate) const OCR_UPLOAD_MIME_BY_EXTENSION: &[(&str, &str)] = &[
    ("pdf", MIME_APPLICATION_PDF),
    ("png", MIME_IMAGE_PNG),
    ("jpg", MIME_IMAGE_JPEG),
    ("jpeg", MIME_IMAGE_JPEG),
    ("gif", MIME_IMAGE_GIF),
    ("webp", MIME_IMAGE_WEBP),
    ("tiff", MIME_IMAGE_TIFF),
    ("tif", MIME_IMAGE_TIFF),
    ("bmp", MIME_IMAGE_BMP),
];
