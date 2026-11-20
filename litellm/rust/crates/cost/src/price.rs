mod meter;

use std::collections::BTreeMap;
use std::num::NonZeroU64;

pub use meter::*;

use crate::call::{CallContext, InferenceGeo, Speed};
use crate::money::{Multiplier, UnitPrice};
use crate::schedule::Schedule;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ServiceTier {
    #[default]
    Standard,
    Priority,
    Flex,
    Ultrafast,
    Balanced,
}

/// A prompt-size range. `OverPromptTokens(n)` is spelled `_above_<n>_tokens` or `_above_<n / 1000>k_tokens`
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Band {
    #[default]
    Base,
    OverPromptTokens(NonZeroU64),
}

/// The request condition a price applies under, its service tier and prompt band
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct When {
    pub tier: ServiceTier,
    pub band: Band,
}

/// The route whose price list a call reads, from the `_batches` and `_decisions` suffixes. Not part of `When`,
/// since a missing batch price is half the online rate and a missing decisions price is the full online rate
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Channel {
    #[default]
    Online,
    Batch,
    Decisions,
}

/// The key of one rate, what is counted, through which channel and under which condition
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Price {
    pub meter: Meter,
    pub channel: Channel,
    pub when: When,
}

/// Every rate a row lists, its off-peak overlay and the row attributes that say how the rates apply
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Prices {
    rates: BTreeMap<Price, UnitPrice>,
    off_peak: Option<OffPeak>,
    billing: Billing,
}

/// Row attributes pricing reads, lowered by model-catalog from the row and its provider
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Billing {
    /// Unit the row bills text in. `Character` only when the row prices characters and not tokens
    pub text_unit: TextUnit,
    /// Whether a band's floor is in the band. xAI rows bill the band from the floor itself
    pub band_edge: BandEdge,
    /// Rate a cache read borrows when the row has no standard cache-read price. Fireworks rows borrow half the input
    /// rate, since its serverless cached prompt tokens default to a 50% discount (https://docs.fireworks.ai/guides/prompt-caching)
    pub cache_read_fallback: CacheReadFallback,
    /// Whether a row with no token price is priced by wall clock, from its `mode`
    pub wall_clock: WallClock,
    /// From `web_search_billing_unit`, whether web search and Maps grounding bill per query or once per prompt
    pub web_search: WebSearchUnit,
    /// Whether `tiered_pricing` bills each meter in its own tier or the whole call at the tier the input picks.
    /// Dashscope rows bill the whole call (inferred, 10 checks no other row has that shape)
    pub tier_scope: TierScope,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TextUnit {
    #[default]
    Token,
    Character,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BandEdge {
    #[default]
    Exclusive,
    Inclusive,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CacheReadFallback {
    #[default]
    InputRate,
    HalfInputRate,
}

/// `ResponseTime` for `chat`, `completion`, `embedding`, `responses` or no mode, `AudioFile` for
/// `audio_transcription`, which prices the input audio when it has one, `Never` for every other mode
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WallClock {
    Never,
    #[default]
    ResponseTime,
    AudioFile,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WebSearchUnit {
    PerQuery,
    #[default]
    PerPrompt,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TierScope {
    #[default]
    PerMeter,
    WholeCall,
}

/// Rates that replace the row's token rates while a UTC window in the schedule is open
#[derive(Clone, Debug, PartialEq)]
pub struct OffPeak {
    schedule: Schedule,
    rates: OffPeakRates,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OffPeakRates {
    pub input: Option<UnitPrice>,
    pub output: Option<UnitPrice>,
    pub cache_read: Option<UnitPrice>,
    pub explicit_cache_read: Option<UnitPrice>,
    pub cache_write: Option<UnitPrice>,
    pub reasoning: Option<UnitPrice>,
}

/// Multipliers applied on top of a priced call, never prices themselves
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Adjustments(BTreeMap<Uplift, Multiplier>);

/// Which call fact picks which multiplier in `Adjustments`, lowered by model-catalog from the row
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Uplift {
    /// OpenAI data residency
    Residency(Residency),
    /// Vertex non-global endpoint, from `regional_endpoint_uplift_multiplier`
    NonGlobalEndpoint,
    /// OpenAI FedRAMP endpoint (gov.api.openai.com), from `fedramp_uplift_multiplier`
    FedrampEndpoint,
    /// Anthropic `speed: fast`, from `provider_specific_entry.fast`
    Speed(Speed),
    /// Anthropic and Bedrock geo, from `provider_specific_entry.<geo>`, lowercased as on main
    InferenceGeo(InferenceGeo),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Residency {
    Eu,
    Us,
}

impl Band {
    pub fn over_prompt_tokens(tokens: u64) -> Self {
        NonZeroU64::new(tokens).map_or(Self::Base, Self::OverPromptTokens)
    }
}

impl Price {
    pub fn online(meter: Meter, when: When) -> Self {
        Self {
            meter,
            channel: Channel::Online,
            when,
        }
    }
}

impl Prices {
    pub fn new(
        rates: BTreeMap<Price, UnitPrice>,
        off_peak: Option<OffPeak>,
        billing: Billing,
    ) -> Self {
        Self {
            rates,
            off_peak,
            billing,
        }
    }

    pub fn get(&self, price: &Price) -> Option<UnitPrice> {
        self.rates.get(price).copied()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Price, UnitPrice)> {
        self.rates.iter().map(|(price, rate)| (price, *rate))
    }

    pub fn off_peak(&self) -> Option<&OffPeak> {
        self.off_peak.as_ref()
    }

    pub fn billing(&self) -> Billing {
        self.billing
    }
}

impl OffPeak {
    pub fn new(schedule: Schedule, rates: OffPeakRates) -> Option<Self> {
        (rates != OffPeakRates::default()).then_some(Self { schedule, rates })
    }

    pub fn schedule(&self) -> &Schedule {
        &self.schedule
    }

    pub fn rates(&self) -> &OffPeakRates {
        &self.rates
    }
}

impl Adjustments {
    pub fn new(uplifts: impl IntoIterator<Item = (Uplift, Multiplier)>) -> Self {
        Self(uplifts.into_iter().collect())
    }

    pub fn get(&self, uplift: &Uplift) -> Option<Multiplier> {
        self.0.get(uplift).copied()
    }

    /// The multipliers `call` picks out of the ones the row carries, in the order main applies them
    pub fn for_call(&self, call: &CallContext) -> impl Iterator<Item = Multiplier> + '_ {
        [
            call.residency.map(Uplift::Residency),
            call.fedramp_endpoint.then_some(Uplift::FedrampEndpoint),
            call.non_global_endpoint
                .then_some(Uplift::NonGlobalEndpoint),
            call.speed.map(Uplift::Speed),
            call.inference_geo.clone().map(Uplift::InferenceGeo),
        ]
        .into_iter()
        .flatten()
        .filter_map(|uplift| self.get(&uplift))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    fn off_peak_needs_a_price() {
        let schedule = Schedule {
            windows: Vec::new(),
            weekday_timezone: None,
        };
        assert_eq!(
            OffPeak::new(schedule.clone(), OffPeakRates::default()),
            None
        );
        assert!(
            OffPeak::new(
                schedule,
                OffPeakRates {
                    output: UnitPrice::new(1.0),
                    ..OffPeakRates::default()
                }
            )
            .is_some()
        );
    }

    #[rstest]
    #[case::zero(0, Band::Base)]
    #[case::positive(272_000, Band::OverPromptTokens(NonZeroU64::new(272_000).unwrap()))]
    fn a_zero_floor_is_the_base_band(#[case] tokens: u64, #[case] expected: Band) {
        assert_eq!(Band::over_prompt_tokens(tokens), expected);
    }
}
