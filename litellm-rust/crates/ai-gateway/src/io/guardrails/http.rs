//! Shared blocking HTTP client for guardrail providers.
//!
//! One process-wide client (connection pool + TLS reused across calls), mirroring
//! the OCR path. Providers call it with the GIL released from the bridge.
//!
//! This is a Rust-native client and cannot consult LiteLLM's Python HTTP layer,
//! so per-deployment options set there (custom client sessions, injected CA
//! bundles, pool tuning) do not apply to guardrail egress. It does, however,
//! respect the standard `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` environment
//! variables (reqwest reads them by default), so a corporate egress proxy
//! configured via the environment still applies. Honoring LiteLLM's Python-level
//! HTTP configuration would require plumbing it across the bridge and is a
//! follow-up; until then operators relying on non-env proxy/TLS config should
//! keep those guardrails on the Python path.

use std::sync::OnceLock;
use std::time::Duration;

/// Outer ceiling on any single provider call. Per-request timeouts tighten this.
const DEFAULT_TIMEOUT_SECS: u64 = 30;

/// Maximum upstream body characters retained in error messages. Guardrail
/// responses can echo request content; keep enough for debugging without
/// forwarding sensitive payloads across the host boundary.
pub const ERROR_BODY_MAX_CHARS: usize = 256;

pub fn client() -> &'static reqwest::blocking::Client {
    static CLIENT: OnceLock<reqwest::blocking::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
            .pool_max_idle_per_host(32)
            .build()
            .expect("failed to build guardrail reqwest client")
    })
}

pub fn truncate_body(body: &str) -> String {
    if body.chars().count() <= ERROR_BODY_MAX_CHARS {
        return body.to_string();
    }
    let truncated: String = body.chars().take(ERROR_BODY_MAX_CHARS).collect();
    format!("{truncated}... (truncated)")
}
