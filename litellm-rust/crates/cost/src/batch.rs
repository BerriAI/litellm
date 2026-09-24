use crate::error::CostError;
use serde_json::Value;

use crate::base_rate_selection::uses_inclusive_token_thresholds;
use crate::generic_usage::{get_billable_input_tokens, parse_prompt_tokens_details};
use crate::pricing::Rate;
use crate::regional_uplift::get_regional_uplift_multiplier;
use crate::responses_usage::ChatUsage;
use crate::wire::py_float;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThresholdPolicy {
    Exclusive,
    Inclusive,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BatchCostRates {
    pub input: Rate,
    pub output: Rate,
    pub cache_read: Rate,
    pub cache_creation: Rate,
}

impl BatchCostRates {
    pub const EMPTY: Self = Self {
        input: Rate::Missing,
        output: Rate::Missing,
        cache_read: Rate::Missing,
        cache_creation: Rate::Missing,
    };
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModalityRates {
    pub audio: Rate,
    pub image: Rate,
    pub video: Rate,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BatchTier {
    pub above_prompt_tokens: u64,
    pub rates: BatchCostRates,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BatchPricing<'a> {
    pub batch: BatchCostRates,
    pub regular: BatchCostRates,
    pub modalities: ModalityRates,
    pub tiers: &'a [BatchTier],
    pub threshold_policy: ThresholdPolicy,
    pub regional_uplift: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BatchUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_creation_tokens: u64,
    pub audio_tokens: u64,
    pub image_tokens: u64,
    pub video_tokens: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BatchCost {
    pub prompt: f64,
    pub completion: f64,
}

fn value(rate: Rate) -> Option<f64> {
    rate.value()
}

fn selected_rate(
    flat: Rate,
    tiers: &[BatchTier],
    prompt_tokens: u64,
    policy: ThresholdPolicy,
    field: fn(BatchCostRates) -> Rate,
) -> Rate {
    tiers
        .iter()
        .filter(|tier| {
            value(field(tier.rates)).is_some()
                && (prompt_tokens > tier.above_prompt_tokens
                    || (policy == ThresholdPolicy::Inclusive
                        && prompt_tokens == tier.above_prompt_tokens))
        })
        .max_by_key(|tier| tier.above_prompt_tokens)
        .map_or(flat, |tier| field(tier.rates))
}

pub fn get_batch_cost_rates(pricing: &BatchPricing<'_>, prompt_tokens: u64) -> BatchCostRates {
    let select = |flat, field| {
        selected_rate(
            flat,
            pricing.tiers,
            prompt_tokens,
            pricing.threshold_policy,
            field,
        )
    };
    BatchCostRates {
        input: select(pricing.batch.input, |rates| rates.input),
        output: select(pricing.batch.output, |rates| rates.output),
        cache_read: select(pricing.batch.cache_read, |rates| rates.cache_read),
        cache_creation: select(pricing.batch.cache_creation, |rates| rates.cache_creation),
    }
}

fn valid_rate(rate: Rate) -> bool {
    value(rate).is_none_or(|rate| rate.is_finite() && rate >= 0.0)
}

fn validate(pricing: &BatchPricing<'_>) -> Result<(), CostError> {
    let rates = [
        pricing.batch.input,
        pricing.batch.output,
        pricing.batch.cache_read,
        pricing.batch.cache_creation,
        pricing.regular.input,
        pricing.regular.output,
        pricing.regular.cache_read,
        pricing.regular.cache_creation,
        pricing.modalities.audio,
        pricing.modalities.image,
        pricing.modalities.video,
    ];
    if rates.into_iter().any(|rate| !valid_rate(rate))
        || pricing.tiers.iter().any(|tier| {
            [
                tier.rates.input,
                tier.rates.output,
                tier.rates.cache_read,
                tier.rates.cache_creation,
            ]
            .into_iter()
            .any(|rate| !valid_rate(rate))
        })
    {
        return Err(CostError::InvalidRate);
    }
    if !pricing.regional_uplift.is_finite() || pricing.regional_uplift < 0.0 {
        return Err(CostError::InvalidMultiplier);
    }
    Ok(())
}

pub fn batch_cost_calculator(
    pricing: &BatchPricing<'_>,
    usage: BatchUsage,
) -> Result<BatchCost, CostError> {
    validate(pricing)?;
    let batch = get_batch_cost_rates(pricing, usage.prompt_tokens);
    let prompt = if let Some(input) = value(batch.input) {
        let cached = if value(batch.cache_read).is_some() {
            usage.cache_read_tokens
        } else {
            0
        };
        let written = if value(batch.cache_creation).is_some() {
            usage.cache_creation_tokens
        } else {
            0
        };
        let text = usage.prompt_tokens.saturating_sub(
            usage
                .audio_tokens
                .saturating_add(usage.image_tokens)
                .saturating_add(usage.video_tokens)
                .saturating_add(cached)
                .saturating_add(written),
        );
        text as f64 * input
            + usage.audio_tokens as f64 * value(pricing.modalities.audio).unwrap_or(input)
            + usage.image_tokens as f64 * value(pricing.modalities.image).unwrap_or(input)
            + usage.video_tokens as f64 * value(pricing.modalities.video).unwrap_or(input)
            + cached as f64 * value(batch.cache_read).unwrap_or(0.0)
            + written as f64 * value(batch.cache_creation).unwrap_or(0.0)
    } else if let Some(input) = value(pricing.regular.input).filter(|rate| *rate != 0.0) {
        let base = get_billable_input_tokens(usage.prompt_tokens, usage.cache_read_tokens)
            - usage.cache_creation_tokens as i128;
        let creation_rate = value(pricing.regular.cache_creation)
            .filter(|rate| *rate != 0.0)
            .unwrap_or(input);
        (base as f64 * input
            + usage.cache_read_tokens as f64 * value(pricing.regular.cache_read).unwrap_or(0.0)
            + usage.cache_creation_tokens as f64 * creation_rate)
            / 2.0
    } else {
        0.0
    };
    let completion = if let Some(output) = value(batch.output) {
        usage.completion_tokens as f64 * output
    } else {
        usage.completion_tokens as f64 * value(pricing.regular.output).unwrap_or(0.0) / 2.0
    };
    let result = BatchCost {
        prompt: prompt * pricing.regional_uplift,
        completion: completion * pricing.regional_uplift,
    };
    if !result.prompt.is_finite() || !result.completion.is_finite() {
        return Err(CostError::NonFiniteCost);
    }
    Ok(result)
}

fn model_rate(model_info: &Value, key: &str) -> Rate {
    model_info
        .get(key)
        .and_then(py_float)
        .map_or(Rate::Missing, Rate::Value)
}

fn threshold(value: &str) -> Option<u64> {
    let (digits, multiplier) = value
        .strip_suffix('k')
        .map_or((value, 1), |digits| (digits, 1000));
    if digits.is_empty() || !digits.bytes().all(|digit| digit.is_ascii_digit()) {
        return None;
    }
    digits.parse::<u64>().ok()?.checked_mul(multiplier)
}

fn selected_model_rate(
    model_info: &Value,
    prefix: &str,
    prompt_tokens: u64,
    policy: ThresholdPolicy,
) -> Rate {
    let flat = model_rate(model_info, &format!("{prefix}_batches"));
    model_info
        .as_object()
        .into_iter()
        .flat_map(|entry| entry.iter())
        .filter_map(|(key, value)| {
            let suffix = key
                .strip_prefix(&format!("{prefix}_above_"))?
                .strip_suffix("_tokens_batches")?;
            let threshold = threshold(suffix)?;
            let crossed = prompt_tokens > threshold
                || (policy == ThresholdPolicy::Inclusive && prompt_tokens == threshold);
            (crossed && !value.is_null()).then_some((threshold, key.as_str()))
        })
        .max_by_key(|(threshold, _)| *threshold)
        .map_or(flat, |(_, key)| match model_rate(model_info, key) {
            Rate::Value(rate) => Rate::Value(rate),
            Rate::Missing | Rate::Null | Rate::Invalid => flat,
        })
}

pub fn batch_cost_rates_from_model_info(
    model_info: &Value,
    prompt_tokens: u64,
    inclusive: bool,
) -> BatchCostRates {
    let policy = if inclusive {
        ThresholdPolicy::Inclusive
    } else {
        ThresholdPolicy::Exclusive
    };
    BatchCostRates {
        input: selected_model_rate(model_info, "input_cost_per_token", prompt_tokens, policy),
        output: selected_model_rate(model_info, "output_cost_per_token", prompt_tokens, policy),
        cache_read: selected_model_rate(
            model_info,
            "cache_read_input_token_cost",
            prompt_tokens,
            policy,
        ),
        cache_creation: selected_model_rate(
            model_info,
            "cache_creation_input_token_cost",
            prompt_tokens,
            policy,
        ),
    }
}

pub fn batch_cost_from_model_info(
    model_info: &Value,
    usage: &ChatUsage,
    provider: Option<&str>,
    data_residency: Option<&str>,
) -> Result<BatchCost, CostError> {
    let policy = if uses_inclusive_token_thresholds(provider) {
        ThresholdPolicy::Inclusive
    } else {
        ThresholdPolicy::Exclusive
    };
    let details = parse_prompt_tokens_details(usage);
    let pricing = BatchPricing {
        batch: BatchCostRates {
            input: selected_model_rate(
                model_info,
                "input_cost_per_token",
                usage.prompt_tokens,
                policy,
            ),
            output: selected_model_rate(
                model_info,
                "output_cost_per_token",
                usage.prompt_tokens,
                policy,
            ),
            cache_read: selected_model_rate(
                model_info,
                "cache_read_input_token_cost",
                usage.prompt_tokens,
                policy,
            ),
            cache_creation: selected_model_rate(
                model_info,
                "cache_creation_input_token_cost",
                usage.prompt_tokens,
                policy,
            ),
        },
        regular: BatchCostRates {
            input: model_rate(model_info, "input_cost_per_token"),
            output: model_rate(model_info, "output_cost_per_token"),
            cache_read: model_rate(model_info, "cache_read_input_token_cost"),
            cache_creation: model_rate(model_info, "cache_creation_input_token_cost"),
        },
        modalities: ModalityRates {
            audio: model_rate(model_info, "input_cost_per_audio_token_batches"),
            image: model_rate(model_info, "input_cost_per_image_token_batches"),
            video: model_rate(model_info, "input_cost_per_video_token_batches"),
        },
        tiers: &[],
        threshold_policy: policy,
        regional_uplift: get_regional_uplift_multiplier(model_info, data_residency),
    };
    batch_cost_calculator(
        &pricing,
        BatchUsage {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            cache_read_tokens: details.cache_hit_tokens,
            cache_creation_tokens: details.cache_creation_tokens,
            audio_tokens: details.audio_tokens,
            image_tokens: details.image_tokens,
            video_tokens: details.video_tokens,
        },
    )
}
