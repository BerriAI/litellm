use crate::{Rate, ThresholdPolicy};

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchError {
    InvalidRate,
    InvalidMultiplier,
    NonFiniteCost,
}

fn value(rate: Rate) -> Option<f64> {
    match rate {
        Rate::Value(value) => Some(value),
        Rate::Missing | Rate::Null => None,
    }
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

fn validate(pricing: &BatchPricing<'_>) -> Result<(), BatchError> {
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
        return Err(BatchError::InvalidRate);
    }
    if !pricing.regional_uplift.is_finite() || pricing.regional_uplift < 0.0 {
        return Err(BatchError::InvalidMultiplier);
    }
    Ok(())
}

pub fn batch_cost_calculator(
    pricing: &BatchPricing<'_>,
    usage: BatchUsage,
) -> Result<BatchCost, BatchError> {
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
        let base = usage.prompt_tokens as i128
            - usage.cache_read_tokens as i128
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
        return Err(BatchError::NonFiniteCost);
    }
    Ok(result)
}
