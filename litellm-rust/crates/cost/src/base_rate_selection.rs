use serde_json::Value;

use crate::generic_input::get_cost_per_unit;
use crate::tiered_pricing::{select_tier_for_input, tier_rate};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TokenBaseRates {
    pub input: f64,
    pub output: f64,
    pub cache_creation: f64,
    pub cache_creation_above_1hr: f64,
    pub cache_read: f64,
}

fn tier_key(base: &str, service_tier: Option<&str>) -> String {
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

fn threshold_number(key: &str) -> Option<f64> {
    let threshold = key.split_once("_above_")?.1.split_once("_tokens")?.0;
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
                && ![
                    "_ultrafast",
                    "_priority",
                    "_standard",
                    "_flex",
                    "_fast",
                    "_batches",
                ]
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

pub fn get_token_base_cost_without_off_peak(
    model_info: &Value,
    prompt_tokens: u64,
    service_tier: Option<&str>,
    threshold_inclusive: bool,
) -> TokenBaseRates {
    if let Some(rates) = tiered_base_rates(model_info, prompt_tokens) {
        return rates;
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
            let threshold = key
                .split_once("_above_")
                .unwrap()
                .1
                .split_once("_tokens")
                .unwrap()
                .0;
            let rate_key = |prefix: &str| {
                tier_key(&format!("{prefix}_above_{threshold}_tokens"), service_tier)
            };
            (
                get_cost_per_unit(model_info, &rate_key("input_cost_per_token"), Some(input))
                    .unwrap_or(input),
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
    TokenBaseRates {
        input,
        output,
        cache_creation: cache_creation.unwrap_or(input),
        cache_creation_above_1hr: one_hour.unwrap_or(cache_creation.unwrap_or(input)),
        cache_read: cache_read.unwrap_or(input),
    }
}
