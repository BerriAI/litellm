#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RawUsage {
    pub prompt_tokens: f64,
    pub completion_tokens: f64,
    pub details_cached_tokens: Option<f64>,
    pub details_cache_write_tokens: Option<f64>,
    pub details_cache_creation_tokens: Option<f64>,
    pub top_level_cache_read_tokens: Option<f64>,
    pub top_level_cache_creation_tokens: Option<f64>,
    pub fallback_cache_read_tokens: Option<f64>,
    pub fallback_cache_creation_tokens: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalizedUsage {
    pub prompt_tokens: f64,
    pub completion_tokens: f64,
    pub cached_tokens: f64,
    pub cache_creation_tokens: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CustomTokenRates {
    pub input: f64,
    pub output: f64,
    pub cache_read: Option<f64>,
    pub cache_creation: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CustomPricing {
    pub token: Option<CustomTokenRates>,
    pub per_second: Option<f64>,
}

impl CustomPricing {
    pub const NONE: Self = Self {
        token: None,
        per_second: None,
    };
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CustomCost {
    pub input: f64,
    pub output: f64,
}

fn nonzero(value: Option<f64>) -> Option<f64> {
    value.filter(|number| *number != 0.0)
}

pub fn normalize_cache_usage(raw: RawUsage) -> Result<NormalizedUsage, CostError> {
    let values = [
        Some(raw.prompt_tokens),
        Some(raw.completion_tokens),
        raw.details_cached_tokens,
        raw.details_cache_write_tokens,
        raw.details_cache_creation_tokens,
        raw.top_level_cache_read_tokens,
        raw.top_level_cache_creation_tokens,
        raw.fallback_cache_read_tokens,
        raw.fallback_cache_creation_tokens,
    ];
    if values
        .into_iter()
        .flatten()
        .any(|value| !value.is_finite() || value < 0.0)
    {
        return Err(CostError::InvalidUsage);
    }
    let detail_read = raw.details_cached_tokens.unwrap_or(0.0);
    let detail_creation = nonzero(raw.details_cache_write_tokens)
        .or(raw.details_cache_creation_tokens)
        .unwrap_or(0.0);
    let top_level_present =
        raw.top_level_cache_read_tokens.is_some() || raw.top_level_cache_creation_tokens.is_some();
    let read_before_fallback = raw.top_level_cache_read_tokens.unwrap_or(detail_read);
    let creation_before_fallback = raw
        .top_level_cache_creation_tokens
        .unwrap_or(detail_creation);
    let fallback_read_used =
        read_before_fallback == 0.0 && nonzero(raw.fallback_cache_read_tokens).is_some();
    let fallback_creation_used =
        creation_before_fallback == 0.0 && nonzero(raw.fallback_cache_creation_tokens).is_some();
    let cached_tokens = if fallback_read_used {
        raw.fallback_cache_read_tokens.unwrap_or(0.0)
    } else {
        read_before_fallback
    };
    let cache_creation_tokens = if fallback_creation_used {
        raw.fallback_cache_creation_tokens.unwrap_or(0.0)
    } else {
        creation_before_fallback
    };
    let prompt_tokens = if top_level_present || fallback_read_used || fallback_creation_used {
        raw.prompt_tokens + cached_tokens + cache_creation_tokens
    } else {
        raw.prompt_tokens
    };
    if !prompt_tokens.is_finite() {
        return Err(CostError::InvalidUsage);
    }
    Ok(NormalizedUsage {
        prompt_tokens,
        completion_tokens: raw.completion_tokens,
        cached_tokens,
        cache_creation_tokens,
    })
}

pub fn cost_per_token_custom_pricing_helper(
    usage: NormalizedUsage,
    pricing: CustomPricing,
    response_time_ms: Option<f64>,
) -> Result<Option<CustomCost>, CostError> {
    if pricing.token.is_none() && pricing.per_second.is_none() {
        return Ok(None);
    }
    let values = [
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.cached_tokens,
        usage.cache_creation_tokens,
    ];
    if values
        .into_iter()
        .any(|value| !value.is_finite() || value < 0.0)
    {
        return Err(CostError::InvalidUsage);
    }
    let cost = match (pricing.token, pricing.per_second) {
        (Some(rates), _) => {
            if [
                Some(rates.input),
                Some(rates.output),
                rates.cache_read,
                rates.cache_creation,
            ]
            .into_iter()
            .flatten()
            .any(|rate| !rate.is_finite() || rate < 0.0)
            {
                return Err(CostError::InvalidRate);
            }
            let regular =
                (usage.prompt_tokens - usage.cached_tokens - usage.cache_creation_tokens).max(0.0);
            CustomCost {
                input: regular * rates.input
                    + usage.cached_tokens * rates.cache_read.unwrap_or(rates.input)
                    + usage.cache_creation_tokens * rates.cache_creation.unwrap_or(rates.input),
                output: usage.completion_tokens * rates.output,
            }
        }
        (None, Some(rate)) => {
            if !rate.is_finite() || rate < 0.0 {
                return Err(CostError::InvalidRate);
            }
            let duration = response_time_ms.unwrap_or(0.0);
            if !duration.is_finite() || duration < 0.0 {
                return Err(CostError::InvalidDuration);
            }
            CustomCost {
                input: 0.0,
                output: rate * duration / 1000.0,
            }
        }
        (None, None) => return Ok(None),
    };
    if !cost.input.is_finite() || !cost.output.is_finite() {
        return Err(CostError::NonFiniteCost);
    }
    Ok(Some(cost))
}

pub fn cost_from_chat_usage(
    usage: &ChatUsage,
    pricing: CustomPricing,
    response_time_ms: Option<f64>,
) -> Result<Option<CustomCost>, CostError> {
    if pricing == CustomPricing::NONE {
        return Ok(None);
    }
    let details = usage.prompt_tokens_details.as_ref();
    let top_level = |field: &str| usage.extra.get(field).and_then(Value::as_f64);
    let normalized = normalize_cache_usage(RawUsage {
        prompt_tokens: usage.prompt_tokens as f64,
        completion_tokens: usage.completion_tokens as f64,
        details_cached_tokens: details.map(|details| details.cached_tokens as f64),
        details_cache_write_tokens: details
            .and_then(|details| details.cache_write_tokens)
            .map(|tokens| tokens as f64),
        details_cache_creation_tokens: details
            .and_then(|details| details.cache_creation_tokens)
            .map(|tokens| tokens as f64),
        top_level_cache_read_tokens: top_level("cache_read_input_tokens"),
        top_level_cache_creation_tokens: top_level("cache_creation_input_tokens"),
        fallback_cache_read_tokens: None,
        fallback_cache_creation_tokens: None,
    })?;
    cost_per_token_custom_pricing_helper(normalized, pricing, response_time_ms)
}
use crate::error::CostError;
use serde_json::Value;

use crate::responses_usage::ChatUsage;
