use jiff::Timestamp;
use serde_json::Value;

use crate::generic_input::get_cost_per_unit;
use crate::off_peak::{
    TokenRates, apply_off_peak_pricing, open_off_peak_block, parse_off_peak_rate,
};
use crate::tiered_pricing::{select_tier_for_input, tier_rate};

pub const NON_STANDARD_THRESHOLD_SUFFIXES: [&str; 6] = [
    "_ultrafast",
    "_priority",
    "_auto",
    "_flex",
    "_fast",
    "_batches",
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TokenBaseRates {
    pub input: f64,
    pub output: f64,
    pub cache_creation: f64,
    pub cache_creation_above_1hr: f64,
    pub cache_read: f64,
}

#[derive(Clone, Copy)]
struct MissingCacheRates {
    read: bool,
    creation: bool,
    one_hour: bool,
}

pub(crate) fn tier_key(base: &str, service_tier: Option<&str>) -> String {
    match service_tier.map(str::to_ascii_lowercase).as_deref() {
        Some("flex") => format!("{base}_flex"),
        Some("priority" | "fast") => format!("{base}_priority"),
        Some("ultrafast") => format!("{base}_ultrafast"),
        _ => base.to_owned(),
    }
}

fn selected_priced_tier(model_info: &Value, prompt_tokens: u64) -> Option<&Value> {
    let tiers = model_info.get("tiered_pricing")?.as_array()?;
    let tokens = i64::try_from(prompt_tokens).ok()?;
    let tier = select_tier_for_input(tiers, tokens)?;
    tier.get("input_cost_per_token").map(|_| tier)
}

pub fn get_tiered_reasoning_rate(model_info: &Value, prompt_tokens: u64) -> Option<f64> {
    let tier = selected_priced_tier(model_info, prompt_tokens)?;
    if tier.get("output_cost_per_reasoning_token").is_none()
        && tier.get("output_cost_per_token").is_none()
    {
        return None;
    }
    Some(tier_rate(
        tier,
        "output_cost_per_reasoning_token",
        Some("output_cost_per_token"),
    ))
}

fn tiered_base_rates(model_info: &Value, prompt_tokens: u64) -> Option<TokenBaseRates> {
    let tier = selected_priced_tier(model_info, prompt_tokens)?;
    let cache_creation = tier_rate(
        tier,
        "cache_creation_input_token_cost",
        Some("input_cost_per_token"),
    );
    let one_hour = tier_rate(
        tier,
        "cache_creation_input_token_cost_above_1hr",
        Some("cache_creation_input_token_cost"),
    );
    Some(TokenBaseRates {
        input: tier_rate(tier, "input_cost_per_token", None),
        output: if tier.get("output_cost_per_token").is_some() {
            tier_rate(tier, "output_cost_per_token", None)
        } else {
            get_cost_per_unit(model_info, "output_cost_per_token", Some(0.0)).unwrap_or(0.0)
        },
        cache_creation,
        cache_creation_above_1hr: if one_hour == 0.0 {
            cache_creation
        } else {
            one_hour
        },
        cache_read: tier_rate(
            tier,
            "cache_read_input_token_cost",
            Some("input_cost_per_token"),
        ),
    })
}

fn threshold_text(key: &str) -> Option<&str> {
    let above = key.split_once("_above_")?.1;
    Some(
        above
            .split_once("_tokens")
            .map_or(above, |(threshold, _)| threshold),
    )
}

fn threshold_number(key: &str) -> Option<f64> {
    let threshold = threshold_text(key)?;
    let number = threshold.replace('k', "").parse::<f64>().ok()?;
    Some(number * if threshold.contains('k') { 1000.0 } else { 1.0 })
}

fn crossed_threshold(
    model_info: &Value,
    prompt_tokens: u64,
    inclusive: bool,
) -> Option<(&str, f64)> {
    model_info
        .as_object()?
        .iter()
        .filter(|(key, value)| {
            key.starts_with("input_cost_per_token_above_")
                && !NON_STANDARD_THRESHOLD_SUFFIXES
                    .iter()
                    .any(|suffix| key.ends_with(suffix))
                && !value.is_null()
        })
        .filter_map(|(key, _)| threshold_number(key).map(|threshold| (key.as_str(), threshold)))
        .filter(|(_, threshold)| {
            prompt_tokens as f64 > *threshold || (inclusive && prompt_tokens as f64 == *threshold)
        })
        .max_by(|left, right| left.1.total_cmp(&right.1))
}

fn select_base_rates(
    model_info: &Value,
    prompt_tokens: u64,
    service_tier: Option<&str>,
    threshold_inclusive: bool,
) -> (TokenBaseRates, MissingCacheRates) {
    if let Some(rates) = tiered_base_rates(model_info, prompt_tokens) {
        return (
            rates,
            MissingCacheRates {
                read: false,
                creation: false,
                one_hour: false,
            },
        );
    }
    let input = get_cost_per_unit(
        model_info,
        &tier_key("input_cost_per_token", service_tier),
        Some(0.0),
    )
    .unwrap_or(0.0);
    let output = get_cost_per_unit(
        model_info,
        &tier_key("output_cost_per_token", service_tier),
        Some(0.0),
    )
    .unwrap_or(0.0);
    let output = if output == 0.0 {
        get_cost_per_unit(model_info, "output_cost_per_image_token", None).unwrap_or(output)
    } else {
        output
    };
    let cache_creation = get_cost_per_unit(
        model_info,
        &tier_key("cache_creation_input_token_cost", service_tier),
        None,
    );
    let one_hour = get_cost_per_unit(
        model_info,
        "cache_creation_input_token_cost_above_1hr",
        None,
    );
    let cache_read = get_cost_per_unit(
        model_info,
        &tier_key("cache_read_input_token_cost", service_tier),
        None,
    );
    let selected =
        crossed_threshold(model_info, prompt_tokens, threshold_inclusive).map(|(key, _)| {
            let threshold = threshold_text(key).unwrap_or_default();
            let rate_key = |prefix: &str| {
                tier_key(&format!("{prefix}_above_{threshold}_tokens"), service_tier)
            };
            let input_key = if service_tier.is_some_and(|tier| !tier.is_empty()) {
                rate_key("input_cost_per_token")
            } else {
                key.to_owned()
            };
            (
                get_cost_per_unit(model_info, &input_key, Some(input)).unwrap_or(input),
                get_cost_per_unit(model_info, &rate_key("output_cost_per_token"), Some(output))
                    .unwrap_or(output),
                get_cost_per_unit(
                    model_info,
                    &rate_key("cache_creation_input_token_cost"),
                    cache_creation,
                ),
                get_cost_per_unit(
                    model_info,
                    &rate_key("cache_creation_input_token_cost_above_1hr"),
                    one_hour,
                ),
                get_cost_per_unit(
                    model_info,
                    &rate_key("cache_read_input_token_cost"),
                    cache_read,
                ),
            )
        });
    let (input, output, cache_creation, one_hour, cache_read) =
        selected.unwrap_or((input, output, cache_creation, one_hour, cache_read));
    (
        TokenBaseRates {
            input,
            output,
            cache_creation: cache_creation.unwrap_or(input),
            cache_creation_above_1hr: one_hour.unwrap_or(cache_creation.unwrap_or(input)),
            cache_read: cache_read.unwrap_or(input),
        },
        MissingCacheRates {
            read: cache_read.is_none(),
            creation: cache_creation.is_none(),
            one_hour: one_hour.is_none(),
        },
    )
}

pub fn get_token_base_cost_without_off_peak(
    model_info: &Value,
    prompt_tokens: u64,
    service_tier: Option<&str>,
    threshold_inclusive: bool,
) -> TokenBaseRates {
    select_base_rates(model_info, prompt_tokens, service_tier, threshold_inclusive).0
}

pub fn get_token_base_cost(
    model_info: &Value,
    prompt_tokens: u64,
    service_tier: Option<&str>,
    threshold_inclusive: bool,
    at: Timestamp,
) -> TokenBaseRates {
    let (standard, missing) =
        select_base_rates(model_info, prompt_tokens, service_tier, threshold_inclusive);
    let Some(off_peak) = open_off_peak_block(model_info, at) else {
        return standard;
    };
    let rates = apply_off_peak_pricing(
        model_info,
        at,
        TokenRates {
            input_rate: standard.input,
            output_rate: standard.output,
            cache_read_rate: standard.cache_read,
            cache_creation_rate: standard.cache_creation,
            reasoning_rate: None,
        },
    );
    let has_rate = |key| parse_off_peak_rate(off_peak.get(key)).is_some();
    let cache_read = if missing.read && !has_rate("cache_read_input_token_cost") {
        rates.input_rate
    } else {
        rates.cache_read_rate
    };
    let cache_creation = if missing.creation && !has_rate("cache_creation_input_token_cost") {
        rates.input_rate
    } else {
        rates.cache_creation_rate
    };
    TokenBaseRates {
        input: rates.input_rate,
        output: rates.output_rate,
        cache_creation,
        cache_creation_above_1hr: if missing.one_hour {
            cache_creation
        } else {
            standard.cache_creation_above_1hr
        },
        cache_read,
    }
}

pub fn uses_inclusive_token_thresholds(provider: Option<&str>) -> bool {
    provider.is_some_and(|name| {
        name.parse::<crate::provider::LlmProviders>()
            .ok()
            .is_some_and(|provider| provider == crate::provider::LlmProviders::XAI)
    })
}
