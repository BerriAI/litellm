use std::sync::LazyLock;

use jiff::Timestamp;
use regex::Regex;
use serde_json::Value;

use crate::generic_cost::calculate_generic_cost_from_model_info;
use crate::provider_cache::with_default_cache_read_rate;
use crate::responses_usage::ChatUsage;

static MOE_SIZE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(\d+)x(\d+)b").unwrap());
static MODEL_SIZE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(\d+)b").unwrap());

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FireworksThresholds {
    pub small: u64,
    pub medium: u64,
    pub moe_small: u64,
    pub moe_medium: u64,
}

impl Default for FireworksThresholds {
    fn default() -> Self {
        Self {
            small: 4,
            medium: 16,
            moe_small: 56,
            moe_medium: 176,
        }
    }
}

pub fn get_base_model_for_pricing(
    model_name: &str,
    thresholds: FireworksThresholds,
) -> &'static str {
    let name = model_name.to_ascii_lowercase();
    if let Some(captures) = MOE_SIZE.captures(&name) {
        let total = captures
            .get(1)
            .and_then(|value| value.as_str().parse::<u64>().ok())
            .zip(
                captures
                    .get(2)
                    .and_then(|value| value.as_str().parse::<u64>().ok()),
            )
            .and_then(|(experts, size)| experts.checked_mul(size));
        if let Some(total) = total {
            if total <= thresholds.moe_small {
                return "fireworks-ai-moe-up-to-56b";
            }
            if total <= thresholds.moe_medium {
                return "fireworks-ai-56b-to-176b";
            }
        }
    }
    if let Some(size) = MODEL_SIZE
        .captures(&name)
        .and_then(|captures| captures.get(1)?.as_str().parse::<f64>().ok())
    {
        if size <= thresholds.small as f64 {
            return "fireworks-ai-up-to-4b";
        }
        if size <= thresholds.medium as f64 {
            return "fireworks-ai-4.1b-to-16b";
        }
        return "fireworks-ai-above-16b";
    }
    "fireworks-ai-default"
}

pub fn cost_per_token(usage: &ChatUsage, model_info: &Value, at: Timestamp) -> (f64, f64) {
    calculate_generic_cost_from_model_info(
        usage,
        &with_default_cache_read_rate(model_info),
        None,
        false,
        1.0,
        at,
    )
}
