use serde::{Deserialize, Serialize};

/// Failure modes a network-backed guardrail provider can hit. Messages are kept
/// stable and non-sensitive; the host truncates any upstream body before it is
/// placed in [`ProviderError::Upstream`].
#[derive(Debug, Clone, Serialize, Deserialize, thiserror::Error)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProviderError {
    #[error("upstream returned status {status}")]
    Upstream { status: u16, body: String },

    #[error("network error: {message}")]
    Network { message: String },

    #[error("timeout after {ms}ms")]
    Timeout { ms: u64 },

    #[error("invalid config: {message}")]
    InvalidConfig { message: String },

    #[error("invalid upstream response: {message}")]
    InvalidResponse { message: String },
}

impl ProviderError {
    pub fn is_unreachable(&self) -> bool {
        match self {
            ProviderError::Network { .. } | ProviderError::Timeout { .. } => true,
            ProviderError::Upstream { status, .. } => matches!(status, 502..=504),
            _ => false,
        }
    }
}
