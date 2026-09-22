#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rate {
    Missing,
    Null,
    Value(f64),
}

impl Rate {
    fn value(self) -> Option<f64> {
        match self {
            Self::Value(value) => Some(value),
            Self::Missing | Self::Null => None,
        }
    }

    fn or(self, fallback: Self) -> Self {
        if self.value().is_some() {
            self
        } else {
            fallback
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rates {
    pub input: Rate,
    pub output: Rate,
    pub cache_read: Rate,
    pub cache_write: Rate,
    pub cache_write_1h: Rate,
}

impl Rates {
    pub const EMPTY: Self = Self {
        input: Rate::Missing,
        output: Rate::Missing,
        cache_read: Rate::Missing,
        cache_write: Rate::Missing,
        cache_write_1h: Rate::Missing,
    };

    fn overlay(self, base: Self) -> Self {
        Self {
            input: self.input.or(base.input),
            output: self.output.or(base.output),
            cache_read: self.cache_read.or(base.cache_read),
            cache_write: self.cache_write.or(base.cache_write),
            cache_write_1h: self.cache_write_1h.or(base.cache_write_1h),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ServiceTier {
    Standard,
    Flex,
    Priority,
    Fast,
    Ultrafast,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThresholdPolicy {
    Exclusive,
    Inclusive,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PromptConvention {
    IncludesCache,
    ExcludesCache,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
    pub cache_write_5m_tokens: Option<u64>,
    pub cache_write_1h_tokens: Option<u64>,
    pub prompt_convention: PromptConvention,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TierRates {
    pub tier: ServiceTier,
    pub rates: Rates,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThresholdRates {
    pub above_prompt_tokens: u64,
    pub standard: Rates,
    pub tier: Option<TierRates>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OffPeakRates {
    pub start_utc_minute: u16,
    pub end_utc_minute: u16,
    pub rates: Rates,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pricing<'a> {
    pub standard: Rates,
    pub tiers: &'a [TierRates],
    pub thresholds: &'a [ThresholdRates],
    pub off_peak: Option<OffPeakRates>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Request {
    pub usage: Usage,
    pub service_tier: ServiceTier,
    pub threshold_policy: ThresholdPolicy,
    pub region_multiplier: Option<f64>,
    pub billed_at_utc_minute: Option<u16>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cost {
    pub input: f64,
    pub output: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PricingError {
    MissingInputRate,
    MissingOutputRate,
    InvalidRate,
    InvalidRegionMultiplier,
    InvalidBillingTime,
    InvalidOffPeakWindow,
    CacheExceedsPrompt,
    InvalidCacheWriteDetails,
    TokenCountOverflow,
}

fn selected_tier(tier: ServiceTier) -> ServiceTier {
    if tier == ServiceTier::Fast {
        ServiceTier::Priority
    } else {
        tier
    }
}

fn resolve_rates(pricing: &Pricing<'_>, request: &Request) -> Result<Rates, PricingError> {
    let tier = selected_tier(request.service_tier);
    let base = pricing
        .tiers
        .iter()
        .find(|entry| tier != ServiceTier::Standard && entry.tier == tier)
        .map_or(pricing.standard, |entry| {
            entry.rates.overlay(pricing.standard)
        });
    let threshold = pricing
        .thresholds
        .iter()
        .filter(|entry| {
            request.usage.prompt_tokens > entry.above_prompt_tokens
                || (request.threshold_policy == ThresholdPolicy::Inclusive
                    && request.usage.prompt_tokens == entry.above_prompt_tokens)
        })
        .max_by_key(|entry| entry.above_prompt_tokens);
    let selected = threshold.map_or(base, |entry| {
        let standard = entry.standard.overlay(base);
        entry
            .tier
            .filter(|specific| tier != ServiceTier::Standard && specific.tier == tier)
            .map_or(standard, |specific| specific.rates.overlay(standard))
    });
    match pricing.off_peak {
        None => Ok(selected),
        Some(window) => {
            if window.start_utc_minute >= 1440
                || window.end_utc_minute > 1440
                || window.start_utc_minute >= window.end_utc_minute
            {
                return Err(PricingError::InvalidOffPeakWindow);
            }
            let minute = request
                .billed_at_utc_minute
                .ok_or(PricingError::InvalidBillingTime)?;
            if minute >= 1440 {
                return Err(PricingError::InvalidBillingTime);
            }
            if (window.start_utc_minute..window.end_utc_minute).contains(&minute) {
                Ok(window.rates.overlay(selected))
            } else {
                Ok(selected)
            }
        }
    }
}

fn checked_rate(rate: Rate, missing: PricingError) -> Result<f64, PricingError> {
    let value = rate.value().ok_or(missing)?;
    if !value.is_finite() || value < 0.0 {
        return Err(PricingError::InvalidRate);
    }
    Ok(value)
}

pub fn calculate(pricing: &Pricing<'_>, request: &Request) -> Result<Cost, PricingError> {
    let usage = request.usage;
    let cached = usage
        .cache_read_tokens
        .checked_add(usage.cache_write_tokens)
        .ok_or(PricingError::TokenCountOverflow)?;
    let regular = match usage.prompt_convention {
        PromptConvention::IncludesCache => usage
            .prompt_tokens
            .checked_sub(cached)
            .ok_or(PricingError::CacheExceedsPrompt)?,
        PromptConvention::ExcludesCache => usage.prompt_tokens,
    };
    let writes = match (usage.cache_write_5m_tokens, usage.cache_write_1h_tokens) {
        (None, None) => (usage.cache_write_tokens, 0),
        (Some(five), Some(one)) if five.checked_add(one) == Some(usage.cache_write_tokens) => {
            (five, one)
        }
        _ => return Err(PricingError::InvalidCacheWriteDetails),
    };
    let rates = resolve_rates(pricing, request)?;
    let input = checked_rate(rates.input, PricingError::MissingInputRate)?;
    let output = checked_rate(rates.output, PricingError::MissingOutputRate)?;
    let read = checked_rate(
        rates.cache_read.or(rates.input),
        PricingError::MissingInputRate,
    )?;
    let write = checked_rate(
        rates.cache_write.or(rates.input),
        PricingError::MissingInputRate,
    )?;
    let write_1h = checked_rate(
        rates.cache_write_1h.or(rates.cache_write).or(rates.input),
        PricingError::MissingInputRate,
    )?;
    let multiplier = request.region_multiplier.unwrap_or(1.0);
    if !multiplier.is_finite() || multiplier <= 0.0 {
        return Err(PricingError::InvalidRegionMultiplier);
    }
    let input_cost = (regular as f64 * input
        + usage.cache_read_tokens as f64 * read
        + writes.0 as f64 * write
        + writes.1 as f64 * write_1h)
        * multiplier;
    let output_cost = usage.completion_tokens as f64 * output * multiplier;
    if !input_cost.is_finite() || !output_cost.is_finite() {
        return Err(PricingError::InvalidRate);
    }
    Ok(Cost {
        input: input_cost,
        output: output_cost,
    })
}
