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

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            max_wait: Duration::from_secs(30),
            timeout: Duration::from_secs(5),
            backoff_base_seconds: 2.0,
            jitter_min_seconds: 0.0,
            jitter_max_seconds: 1.0,
            limits: IntegrityLimits::new(50, 0.5).expect("0.5 is a ratio"),
        }
    }
}

/// Main's rule for serving the bundled map without fetching, `LITELLM_LOCAL_MODEL_COST_MAP=true` or
/// `LITELLM_OFFLINE_MODE=true`, compared as Python does
pub fn local_map_forced(env: impl Fn(&str) -> Option<String>) -> bool {
    let local =
        env("LITELLM_LOCAL_MODEL_COST_MAP").is_some_and(|value| value.to_lowercase() == "true");
    let offline =
        env("LITELLM_OFFLINE_MODE").is_some_and(|value| value.trim().to_lowercase() == "true");
    local || offline
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::neither(&[], false)]
    #[case::local(&[("LITELLM_LOCAL_MODEL_COST_MAP", "True")], true)]
    #[case::local_false(&[("LITELLM_LOCAL_MODEL_COST_MAP", "false")], false)]
    #[case::local_is_not_trimmed(&[("LITELLM_LOCAL_MODEL_COST_MAP", " true")], false)]
    #[case::local_one_is_not_true(&[("LITELLM_LOCAL_MODEL_COST_MAP", "1")], false)]
    #[case::offline(&[("LITELLM_OFFLINE_MODE", "TRUE")], true)]
    #[case::offline_is_trimmed(&[("LITELLM_OFFLINE_MODE", " true\n")], true)]
    fn the_bundled_map_is_forced_only_by_a_true_flag(
        #[case] vars: &[(&str, &str)],
        #[case] forced: bool,
    ) {
        let env = |name: &str| {
            vars.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| (*value).to_owned())
        };
        assert_eq!(local_map_forced(env), forced);
    }
}
