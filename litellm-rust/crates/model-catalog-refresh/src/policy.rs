use std::time::Duration;

use litellm_model_catalog::IntegrityLimits;

/// Main's defaults, 3 attempts, 30 s max wait, 5 s timeout, backoff base 2.0, jitter 0.0 to 1.0 s
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub max_wait: Duration,
    pub timeout: Duration,
    pub backoff_base_seconds: f64,
    pub jitter_min_seconds: f64,
    pub jitter_max_seconds: f64,
    /// Main's 50 model floor and 0.5 shrink ratio
    pub limits: IntegrityLimits,
}

/// Main's default `LITELLM_MODEL_COST_MAP_URL`
pub const DEFAULT_REGISTRY_URL: &str =
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json";
