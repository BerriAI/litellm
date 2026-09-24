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
    Unknown,
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
pub struct ThresholdRates<'a> {
    pub above_prompt_tokens: u64,
    pub standard: Rates,
    pub tiers: &'a [TierRates],
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
    pub thresholds: &'a [ThresholdRates<'a>],
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
    pub uncached_input: f64,
    pub cache_read: f64,
    pub cache_write_5m: f64,
    pub cache_write_1h: f64,
    pub output: f64,
    pub multiplier: f64,
    pub rates: EffectiveRates,
}

impl Cost {
    pub fn input(self) -> f64 {
        (self.uncached_input + self.cache_read + self.cache_write_5m + self.cache_write_1h)
            * self.multiplier
    }

    pub fn output(self) -> f64 {
        self.output * self.multiplier
    }

    pub fn total(self) -> f64 {
        self.input() + self.output()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EffectiveRates {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write_5m: f64,
    pub cache_write_1h: f64,
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
    DuplicateTier,
    DuplicateThreshold,
    DuplicateThresholdTier,
}

fn selected_tier(tier: ServiceTier) -> ServiceTier {
    if tier == ServiceTier::Fast {
        ServiceTier::Priority
    } else {
        tier
    }
}

#[derive(Clone, Debug)]
struct CompiledThreshold {
    above_prompt_tokens: u64,
    standard: Rates,
    tiers: Vec<TierRates>,
}

#[derive(Clone, Debug)]
pub struct PricingPlan {
    standard: Rates,
    tiers: Vec<TierRates>,
    thresholds: Vec<CompiledThreshold>,
    off_peak: Option<OffPeakRates>,
}

fn valid_rates(rates: Rates) -> bool {
    [
        rates.input,
        rates.output,
        rates.cache_read,
        rates.cache_write,
        rates.cache_write_1h,
    ]
    .into_iter()
    .all(|rate| {
        rate.value()
            .is_none_or(|value| value.is_finite() && value >= 0.0)
    })
}

fn validate_tiers(tiers: &[TierRates], duplicate: PricingError) -> Result<(), PricingError> {
    if tiers.iter().any(|entry| !valid_rates(entry.rates)) {
        return Err(PricingError::InvalidRate);
    }
    if tiers.iter().enumerate().any(|(index, entry)| {
        matches!(
            entry.tier,
            ServiceTier::Standard | ServiceTier::Unknown | ServiceTier::Fast
        ) || tiers[..index]
            .iter()
            .any(|previous| previous.tier == entry.tier)
    }) {
        return Err(duplicate);
    }
    Ok(())
}

pub fn compile(pricing: &Pricing<'_>) -> Result<PricingPlan, PricingError> {
    if !valid_rates(pricing.standard) {
        return Err(PricingError::InvalidRate);
    }
    validate_tiers(pricing.tiers, PricingError::DuplicateTier)?;
    if let Some(window) = pricing.off_peak {
        if window.start_utc_minute >= 1440
            || window.end_utc_minute > 1440
            || window.start_utc_minute >= window.end_utc_minute
        {
            return Err(PricingError::InvalidOffPeakWindow);
        }
        if !valid_rates(window.rates) {
            return Err(PricingError::InvalidRate);
        }
    }
    let mut thresholds: Vec<_> = pricing
        .thresholds
        .iter()
        .map(|entry| {
            if !valid_rates(entry.standard) {
                return Err(PricingError::InvalidRate);
            }
            validate_tiers(entry.tiers, PricingError::DuplicateThresholdTier)?;
            Ok(CompiledThreshold {
                above_prompt_tokens: entry.above_prompt_tokens,
                standard: entry.standard,
                tiers: entry.tiers.to_vec(),
            })
        })
        .collect::<Result<_, _>>()?;
    thresholds.sort_unstable_by_key(|entry| entry.above_prompt_tokens);
    if thresholds
        .windows(2)
        .any(|pair| pair[0].above_prompt_tokens == pair[1].above_prompt_tokens)
    {
        return Err(PricingError::DuplicateThreshold);
    }
    Ok(PricingPlan {
        standard: pricing.standard,
        tiers: pricing.tiers.to_vec(),
        thresholds,
        off_peak: pricing.off_peak,
    })
}

impl PricingPlan {
    fn resolve_rates(
        &self,
        request: &Request,
        threshold_tokens: u64,
    ) -> Result<Rates, PricingError> {
        let tier = selected_tier(request.service_tier);
        let base = self
            .tiers
            .iter()
            .find(|entry| tier != ServiceTier::Standard && entry.tier == tier)
            .map_or(self.standard, |entry| entry.rates.overlay(self.standard));
        let threshold = self.thresholds.iter().rev().find(|entry| {
            threshold_tokens > entry.above_prompt_tokens
                || (request.threshold_policy == ThresholdPolicy::Inclusive
                    && threshold_tokens == entry.above_prompt_tokens)
        });
        let selected = threshold.map_or(base, |entry| {
            let standard = entry.standard.overlay(base);
            entry
                .tiers
                .iter()
                .find(|specific| tier != ServiceTier::Standard && specific.tier == tier)
                .map_or(standard, |specific| specific.rates.overlay(standard))
        });
        match self.off_peak {
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

    pub fn calculate(&self, request: &Request) -> Result<Cost, PricingError> {
        let usage = request.usage;
        let cached = usage
            .cache_read_tokens
            .checked_add(usage.cache_write_tokens)
            .ok_or(PricingError::TokenCountOverflow)?;
        let (regular, threshold_tokens) = match usage.prompt_convention {
            PromptConvention::IncludesCache => (
                usage
                    .prompt_tokens
                    .checked_sub(cached)
                    .ok_or(PricingError::CacheExceedsPrompt)?,
                usage.prompt_tokens,
            ),
            PromptConvention::ExcludesCache => (
                usage.prompt_tokens,
                usage
                    .prompt_tokens
                    .checked_add(cached)
                    .ok_or(PricingError::TokenCountOverflow)?,
            ),
        };
        let writes = match (usage.cache_write_5m_tokens, usage.cache_write_1h_tokens) {
            (None, None) => (usage.cache_write_tokens, 0),
            (Some(five), Some(one)) if five.checked_add(one) == Some(usage.cache_write_tokens) => {
                (five, one)
            }
            _ => return Err(PricingError::InvalidCacheWriteDetails),
        };
        let rates = self.resolve_rates(request, threshold_tokens)?;
        let input = Self::checked_rate(rates.input, PricingError::MissingInputRate)?;
        let output = Self::checked_rate(rates.output, PricingError::MissingOutputRate)?;
        let read = Self::checked_rate(
            rates.cache_read.or(rates.input),
            PricingError::MissingInputRate,
        )?;
        let write = Self::checked_rate(
            rates.cache_write.or(rates.input),
            PricingError::MissingInputRate,
        )?;
        let write_1h = Self::checked_rate(
            rates.cache_write_1h.or(rates.cache_write).or(rates.input),
            PricingError::MissingInputRate,
        )?;
        let multiplier = request.region_multiplier.unwrap_or(1.0);
        if !multiplier.is_finite() || multiplier <= 0.0 {
            return Err(PricingError::InvalidRegionMultiplier);
        }
        let cost = Cost {
            uncached_input: regular as f64 * input,
            cache_read: usage.cache_read_tokens as f64 * read,
            cache_write_5m: writes.0 as f64 * write,
            cache_write_1h: writes.1 as f64 * write_1h,
            output: usage.completion_tokens as f64 * output,
            multiplier,
            rates: EffectiveRates {
                input,
                output,
                cache_read: read,
                cache_write_5m: write,
                cache_write_1h: write_1h,
            },
        };
        if !cost.total().is_finite() {
            return Err(PricingError::TokenCountOverflow);
        }
        Ok(cost)
    }
}

pub fn calculate(pricing: &Pricing<'_>, request: &Request) -> Result<Cost, PricingError> {
    compile(pricing)?.calculate(request)
}
