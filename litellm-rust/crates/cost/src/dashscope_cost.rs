use jiff::Timestamp;
use serde_json::Value;

use crate::generic_usage::{parse_completion_tokens_details, parse_prompt_tokens_details};
use crate::off_peak::{open_off_peak_block, parse_off_peak_rate};
use crate::responses_usage::ChatUsage;
use crate::tiered_pricing::{select_tier_for_input, tier_rate};
use crate::wire::py_float;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TokenBreakdown {
    pub text_tokens: u64,
    pub cached_tokens: u64,
    pub cache_creation_tokens: u64,
    pub completion_tokens: u64,
    pub reasoning_tokens: u64,
}

impl TokenBreakdown {
    pub fn total_input_tokens(self) -> u64 {
        self.text_tokens
            .saturating_add(self.cached_tokens)
            .saturating_add(self.cache_creation_tokens)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TokenRates {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_creation: f64,
    pub reasoning: Option<f64>,
}

pub fn extract_token_breakdown(usage: &ChatUsage) -> TokenBreakdown {
    let prompt = parse_prompt_tokens_details(usage);
    let reasoning = parse_completion_tokens_details(usage).reasoning_tokens;
    TokenBreakdown {
        text_tokens: usage
            .prompt_tokens
            .saturating_sub(prompt.cache_hit_tokens)
            .saturating_sub(prompt.cache_creation_tokens),
        cached_tokens: prompt.cache_hit_tokens,
        cache_creation_tokens: prompt.cache_creation_tokens,
        completion_tokens: usage.completion_tokens.saturating_sub(reasoning),
        reasoning_tokens: reasoning,
    }
}

fn rate(value: Option<&Value>) -> f64 {
    value.and_then(py_float).unwrap_or(0.0)
}

fn flat_rate(model_info: &Value, key: &str, fallback: &str) -> f64 {
    rate(
        model_info
            .get(key)
            .filter(|value| !value.is_null())
            .or_else(|| model_info.get(fallback)),
    )
}

pub fn flat_rates(model_info: &Value) -> TokenRates {
    TokenRates {
        input: rate(model_info.get("input_cost_per_token")),
        output: rate(model_info.get("output_cost_per_token")),
        cache_read: flat_rate(
            model_info,
            "cache_read_input_token_cost",
            "input_cost_per_token",
        ),
        cache_creation: flat_rate(
            model_info,
            "cache_creation_input_token_cost",
            "input_cost_per_token",
        ),
        reasoning: model_info
            .get("output_cost_per_reasoning_token")
            .filter(|value| !value.is_null())
            .map(|value| rate(Some(value))),
    }
}

pub fn tier_rates(model_info: &Value, tier: &Value) -> TokenRates {
    let flat = flat_rates(model_info);
    let tier_has_output = tier.get("output_cost_per_token").is_some();
    TokenRates {
        input: tier_rate(tier, "input_cost_per_token", None),
        output: if tier_has_output {
            tier_rate(tier, "output_cost_per_token", None)
        } else {
            flat.output
        },
        cache_read: tier_rate(
            tier,
            "cache_read_input_token_cost",
            Some("input_cost_per_token"),
        ),
        cache_creation: tier_rate(
            tier,
            "cache_creation_input_token_cost",
            Some("input_cost_per_token"),
        ),
        reasoning: if tier.get("output_cost_per_reasoning_token").is_some() {
            Some(tier_rate(tier, "output_cost_per_reasoning_token", None))
        } else if tier_has_output {
            None
        } else {
            flat.reasoning
        },
    }
}

pub fn bill(breakdown: TokenBreakdown, rates: TokenRates) -> (f64, f64) {
    (
        breakdown.text_tokens as f64 * rates.input
            + breakdown.cached_tokens as f64 * rates.cache_read
            + breakdown.cache_creation_tokens as f64 * rates.cache_creation,
        breakdown.completion_tokens as f64 * rates.output
            + breakdown.reasoning_tokens as f64 * rates.reasoning.unwrap_or(rates.output),
    )
}

pub fn cost_per_token(usage: &ChatUsage, model_info: &Value, at: Timestamp) -> (f64, f64) {
    let breakdown = extract_token_breakdown(usage);
    let selected = model_info
        .get("tiered_pricing")
        .and_then(Value::as_array)
        .and_then(|tiers| {
            i64::try_from(breakdown.total_input_tokens())
                .ok()
                .and_then(|tokens| select_tier_for_input(tiers, tokens))
        });
    let standard = selected.map_or_else(
        || flat_rates(model_info),
        |tier| tier_rates(model_info, tier),
    );
    let off_peak = open_off_peak_block(model_info, at);
    let off_peak_rate = |key, standard| {
        off_peak
            .and_then(|block| parse_off_peak_rate(block.get(key)))
            .unwrap_or(standard)
    };
    let rates = TokenRates {
        input: off_peak_rate("input_cost_per_token", standard.input),
        output: off_peak_rate("output_cost_per_token", standard.output),
        cache_read: off_peak_rate("cache_read_input_token_cost", standard.cache_read),
        cache_creation: off_peak_rate("cache_creation_input_token_cost", standard.cache_creation),
        reasoning: off_peak
            .and_then(|block| parse_off_peak_rate(block.get("output_cost_per_reasoning_token")))
            .or(standard.reasoning),
    };
    bill(breakdown, rates)
}
