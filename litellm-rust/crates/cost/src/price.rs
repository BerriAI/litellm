mod meter;

use std::collections::BTreeMap;
use std::num::NonZeroU64;

pub use meter::*;

use crate::PriceEvaluationError;
use crate::money::{Multiplier, ScaleFactor, UnitPrice};
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Band {
    #[default]
    Base,
    OverPromptTokens(NonZeroU64),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct When {
    pub tier: ServiceTier,
    pub band: Band,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Channel {
    #[default]
    Online,
    Batch,
    Decisions,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Price {
    pub meter: Meter,
    pub channel: Channel,
    pub when: When,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Prices {
    rates: BTreeMap<(Price, PriceSource), UnitPrice>,
    off_peak: Option<OffPeak>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum PriceSource {
    #[default]
    Flat,
    TokenTier(u64),
}

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

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Adjustments(BTreeMap<Uplift, Multiplier>);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Uplift {
    NonGlobalEndpoint,
    Residency(Residency),
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
    pub fn new(rates: BTreeMap<Price, UnitPrice>, off_peak: Option<OffPeak>) -> Self {
        Self::from_sources(
            rates
                .into_iter()
                .map(|(price, rate)| ((price, PriceSource::Flat), rate))
                .collect(),
            off_peak,
        )
    }

    pub fn from_sources(
        rates: BTreeMap<(Price, PriceSource), UnitPrice>,
        off_peak: Option<OffPeak>,
    ) -> Self {
        Self { rates, off_peak }
    }

    pub fn get(&self, price: &Price) -> Option<UnitPrice> {
        self.get_from(price, PriceSource::Flat)
    }

    pub fn get_from(&self, price: &Price, source: PriceSource) -> Option<UnitPrice> {
        self.rates.get(&(price.clone(), source)).copied()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Price, UnitPrice)> {
        self.rates.iter().filter_map(|((price, source), rate)| {
            (*source == PriceSource::Flat).then_some((price, *rate))
        })
    }

    pub fn off_peak(&self) -> Option<&OffPeak> {
        self.off_peak.as_ref()
    }

    pub fn with_cache_read_default(
        mut self,
        ratio: ScaleFactor,
        off_peak_read_declared: bool,
    ) -> Result<Self, PriceEvaluationError> {
        let read = Price::online(
            Meter::CacheRead {
                kind: TokenKind::Text,
                mode: CacheReadMode::Implicit,
            },
            When::default(),
        );
        if self.get(&read).is_some() {
            return Ok(self);
        }
        let Some(input) = self.get(&Price::online(Meter::TEXT_INPUT, When::default())) else {
            return Ok(self);
        };
        let rate = ratio
            .price(input)
            .ok_or(PriceEvaluationError::NonFiniteValue)?;
        if !off_peak_read_declared
            && let Some(off_peak) = self.off_peak.as_mut()
            && off_peak.rates.cache_read.is_none()
        {
            off_peak.rates.cache_read = Some(
                ratio
                    .price(off_peak.rates.input.unwrap_or(input))
                    .ok_or(PriceEvaluationError::NonFiniteValue)?,
            );
        }
        self.rates.insert((read, PriceSource::Flat), rate);
        Ok(self)
    }

    pub fn has_token_prices(&self) -> bool {
        self.rates.keys().any(|(price, _)| price.meter.is_token()) || self.off_peak.is_some()
    }

    /// The online prompt band a request falls in. Every token meter's `_above_` prices count,
    /// and the highest band the prompt passes wins.
    pub fn band(&self, tier: ServiceTier, prompt_tokens: u64, inclusive: bool) -> Band {
        self.channel_band(Channel::Online, tier, prompt_tokens, inclusive)
    }

    pub(crate) fn channel_band(
        &self,
        channel: Channel,
        tier: ServiceTier,
        prompt_tokens: u64,
        inclusive: bool,
    ) -> Band {
        Self::find_band(
            self.iter().map(|(price, _)| price),
            channel,
            tier,
            prompt_tokens,
            inclusive,
        )
    }

    pub(crate) fn all_source_band(
        &self,
        channel: Channel,
        tier: ServiceTier,
        prompt_tokens: u64,
        inclusive: bool,
    ) -> Band {
        Self::find_band(
            self.rates.keys().map(|(price, _)| price),
            channel,
            tier,
            prompt_tokens,
            inclusive,
        )
    }

    fn find_band<'a>(
        prices: impl Iterator<Item = &'a Price>,
        channel: Channel,
        tier: ServiceTier,
        prompt_tokens: u64,
        inclusive: bool,
    ) -> Band {
        prices
            .filter(|price| {
                price.channel == channel
                    && price.meter.sets_prompt_band()
                    && (price.when.tier == ServiceTier::Standard || price.when.tier == tier)
            })
            .filter_map(|price| match price.when.band {
                Band::OverPromptTokens(floor) => Some(floor.get()),
                Band::Base => None,
            })
            .filter(|floor| prompt_tokens > *floor || (inclusive && prompt_tokens == *floor))
            .max()
            .map_or(Band::Base, Band::over_prompt_tokens)
    }

    /// The online rate for `meter`, tried at `(tier, band)`, `(standard, band)`, `(tier, base)`
    /// and `(standard, base)` in that order. A missing cache price falls back to the input price
    /// resolved the same way.
    pub fn rate(&self, meter: &Meter, tier: ServiceTier, band: Band) -> Option<UnitPrice> {
        self.direct_rate(meter, Channel::Online, tier, band)
            .or_else(|| self.rate(&meter.fallback()?, tier, band))
    }

    pub(crate) fn direct_rate(
        &self,
        meter: &Meter,
        channel: Channel,
        tier: ServiceTier,
        band: Band,
    ) -> Option<UnitPrice> {
        [
            When { tier, band },
            When {
                tier: ServiceTier::Standard,
                band,
            },
            When {
                tier,
                band: Band::Base,
            },
            When::default(),
        ]
        .into_iter()
        .find_map(|when| {
            self.get(&Price {
                meter: meter.clone(),
                channel,
                when,
            })
        })
    }
}

impl OffPeakRates {
    pub(crate) fn get(&self, meter: &Meter) -> Option<UnitPrice> {
        match meter {
            Meter::Input(InputUnit::Token(TokenKind::Text)) => self.input,
            Meter::Output(OutputUnit::Token(TokenKind::Text)) => self.output,
            Meter::Output(OutputUnit::ReasoningToken) => self.reasoning,
            Meter::CacheRead {
                kind: TokenKind::Text,
                mode: CacheReadMode::Implicit,
            } => self.cache_read,
            Meter::CacheRead {
                kind: TokenKind::Text,
                mode: CacheReadMode::Explicit,
            } => self.explicit_cache_read.or(self.cache_read),
            Meter::CacheWrite {
                kind: TokenKind::Text,
                ttl: CacheTtl::Default,
            } => self.cache_write,
            _ => None,
        }
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

    pub fn get(&self, uplift: Uplift) -> Option<Multiplier> {
        self.0.get(&uplift).copied()
    }
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};

    use super::*;

    const INPUT: Meter = Meter::TEXT_INPUT;
    const OUTPUT: Meter = Meter::Output(OutputUnit::Token(TokenKind::Text));
    const CACHE_READ: Meter = Meter::CacheRead {
        kind: TokenKind::Text,
        mode: CacheReadMode::Implicit,
    };
    const CACHE_WRITE_HOUR: Meter = Meter::CacheWrite {
        kind: TokenKind::Text,
        ttl: CacheTtl::OneHour,
    };

    #[rstest]
    fn source_qualified_rates_do_not_enter_generic_lookup_or_bands() {
        let flat = Price::online(INPUT, When::default());
        let first_tier = Price::online(OUTPUT, When::default());
        let last_tier = Price::online(
            OUTPUT,
            When {
                band: Band::over_prompt_tokens(100),
                ..Default::default()
            },
        );
        let prices = Prices::from_sources(
            [
                (
                    (flat.clone(), PriceSource::Flat),
                    UnitPrice::new(1.0).unwrap(),
                ),
                (
                    (flat.clone(), PriceSource::TokenTier(0)),
                    UnitPrice::new(3.0).unwrap(),
                ),
                (
                    (first_tier.clone(), PriceSource::TokenTier(0)),
                    UnitPrice::new(4.0).unwrap(),
                ),
                (
                    (last_tier, PriceSource::TokenTier(100)),
                    UnitPrice::new(5.0).unwrap(),
                ),
            ]
            .into(),
            None,
        );
        assert_eq!(prices.get(&flat), UnitPrice::new(1.0));
        assert_eq!(
            prices.get_from(&flat, PriceSource::TokenTier(0)),
            UnitPrice::new(3.0)
        );
        assert_eq!(
            prices.get_from(&first_tier, PriceSource::TokenTier(0)),
            UnitPrice::new(4.0)
        );
        assert_eq!(prices.get(&first_tier), None);
        assert_eq!(
            prices.rate(&OUTPUT, ServiceTier::Standard, Band::Base),
            None
        );
        assert_eq!(prices.band(ServiceTier::Standard, 101, true), Band::Base);
        assert_eq!(
            prices.iter().collect::<Vec<_>>(),
            vec![(&flat, UnitPrice::new(1.0).unwrap())]
        );
        assert!(prices.has_token_prices());
    }

    #[rstest]
    #[case::input(INPUT, true)]
    #[case::output(OUTPUT, true)]
    #[case::read(CACHE_READ, true)]
    #[case::hour_write(CACHE_WRITE_HOUR, true)]
    #[case::reasoning(Meter::Output(OutputUnit::ReasoningToken), true)]
    #[case::audio(Meter::Input(InputUnit::Token(TokenKind::Audio)), true)]
    #[case::image_token(Meter::Output(OutputUnit::Token(TokenKind::Image)), true)]
    #[case::video_token(Meter::Output(OutputUnit::Token(TokenKind::Video)), true)]
    #[case::image(Meter::Input(InputUnit::Image), false)]
    #[case::characters(Meter::Output(OutputUnit::Character), false)]
    #[case::query(Meter::Input(InputUnit::Query { over_results: None }), false)]
    #[case::wall_clock(Meter::WallClockSecond, false)]
    fn token_presence_is_explicit_and_independent_of_request(
        #[case] meter: Meter,
        #[case] expected: bool,
        #[values(PriceSource::Flat, PriceSource::TokenTier(0))] source: PriceSource,
        #[values(Channel::Online, Channel::Batch, Channel::Decisions)] channel: Channel,
    ) {
        let key = Price {
            meter,
            channel,
            when: When {
                tier: ServiceTier::Flex,
                band: Band::over_prompt_tokens(100),
            },
        };
        let prices =
            Prices::from_sources([((key, source), UnitPrice::new(0.0).unwrap())].into(), None);
        assert_eq!(prices.has_token_prices(), expected);
    }

    #[rstest]
    #[case::base(Some(2.0), None, (None, None, false), (Some(1.0), Some(1.0)))]
    #[case::zero_input(Some(0.0), None, (Some(6.0), None, false), (Some(0.0), Some(3.0)))]
    #[case::absent_input(None, None, (Some(6.0), None, false), (None, None))]
    #[case::explicit_zero(Some(2.0), Some(0.0), (Some(6.0), None, false), (Some(0.0), None))]
    #[case::explicit_read(Some(2.0), Some(0.25), (Some(6.0), None, false), (Some(0.25), None))]
    #[case::offpeak_input(Some(2.0), None, (Some(6.0), None, false), (Some(1.0), Some(3.0)))]
    #[case::offpeak_null_read(Some(2.0), None, (Some(6.0), None, true), (Some(1.0), None))]
    #[case::offpeak_zero_read(Some(2.0), None, (Some(6.0), Some(0.0), true), (Some(1.0), Some(0.0)))]
    #[case::offpeak_read(Some(2.0), None, (Some(6.0), Some(0.25), true), (Some(1.0), Some(0.25)))]
    fn cache_defaults_derive_only_missing_rates_from_base_input(
        #[case] input: Option<f64>,
        #[case] read: Option<f64>,
        #[case] off_peak: (Option<f64>, Option<f64>, bool),
        #[case] expected: (Option<f64>, Option<f64>),
    ) {
        let read_meter = Meter::CacheRead {
            kind: TokenKind::Text,
            mode: CacheReadMode::Implicit,
        };
        let read_key = Price::online(read_meter, When::default());
        let prices = Prices::new(
            [
                input.map(|value| {
                    (
                        Price::online(INPUT, When::default()),
                        UnitPrice::new(value).unwrap(),
                    )
                }),
                read.map(|value| (read_key.clone(), UnitPrice::new(value).unwrap())),
            ]
            .into_iter()
            .flatten()
            .collect(),
            OffPeak::new(
                Schedule {
                    windows: vec![],
                    weekday_timezone: None,
                },
                OffPeakRates {
                    input: off_peak.0.and_then(UnitPrice::new),
                    cache_read: off_peak.1.and_then(UnitPrice::new),
                    output: UnitPrice::new(3.0),
                    ..Default::default()
                },
            ),
        );
        let original = prices.clone();
        let prepared = prices
            .with_cache_read_default(ScaleFactor::new(0.5).unwrap(), off_peak.2)
            .unwrap();
        assert_eq!(prepared.get(&read_key).map(UnitPrice::get), expected.0);
        assert_eq!(
            prepared
                .off_peak()
                .unwrap()
                .rates()
                .cache_read
                .map(UnitPrice::get),
            expected.1
        );
        assert_eq!(original.get(&read_key).map(UnitPrice::get), read);
        assert_eq!(original.has_token_prices(), prepared.has_token_prices());
    }

    #[rstest]
    #[case::base_input(f64::MAX, 2.0)]
    #[case::offpeak_input(2.0, f64::MAX)]
    fn cache_default_overflow_is_an_error(#[case] input: f64, #[case] off_peak: f64) {
        let prices = Prices::new(
            [(
                Price::online(INPUT, When::default()),
                UnitPrice::new(input).unwrap(),
            )]
            .into(),
            OffPeak::new(
                Schedule {
                    windows: vec![],
                    weekday_timezone: None,
                },
                OffPeakRates {
                    input: UnitPrice::new(off_peak),
                    ..Default::default()
                },
            ),
        );
        assert_eq!(
            prices.with_cache_read_default(ScaleFactor::new(2.0).unwrap(), false),
            Err(PriceEvaluationError::NonFiniteValue)
        );
    }

    fn over(tokens: u64) -> Band {
        Band::over_prompt_tokens(tokens)
    }

    fn prices(rates: &[(Meter, Channel, ServiceTier, Band, f64)]) -> Prices {
        Prices::new(
            rates
                .iter()
                .map(|(meter, channel, tier, band, rate)| {
                    (
                        Price {
                            meter: meter.clone(),
                            channel: *channel,
                            when: When {
                                tier: *tier,
                                band: *band,
                            },
                        },
                        UnitPrice::new(*rate).unwrap(),
                    )
                })
                .collect(),
            None,
        )
    }

    #[fixture]
    fn priority_with_a_standard_band() -> Prices {
        use Channel::Online;
        use ServiceTier::{Priority, Standard};
        prices(&[
            (INPUT, Online, Standard, Band::Base, 1.0),
            (INPUT, Online, Priority, Band::Base, 2.0),
            (INPUT, Online, Standard, over(200_000), 10.0),
            (OUTPUT, Online, Standard, Band::Base, 3.0),
            (OUTPUT, Online, Priority, over(200_000), 30.0),
        ])
    }

    #[rstest]
    #[case::below(1_000, false, Band::Base)]
    #[case::at_floor_exclusive(200_000, false, Band::Base)]
    #[case::at_floor_inclusive(200_000, true, over(200_000))]
    #[case::above(200_001, false, over(200_000))]
    fn the_band_starts_strictly_above_its_floor_unless_inclusive(
        priority_with_a_standard_band: Prices,
        #[case] prompt_tokens: u64,
        #[case] inclusive: bool,
        #[case] expected: Band,
    ) {
        assert_eq!(
            priority_with_a_standard_band.band(ServiceTier::Standard, prompt_tokens, inclusive),
            expected
        );
    }

    #[rstest]
    fn a_band_only_another_tier_prices_does_not_apply() {
        let prices = prices(&[(
            OUTPUT,
            Channel::Online,
            ServiceTier::Flex,
            over(100_000),
            1.0,
        )]);
        assert_eq!(
            prices.band(ServiceTier::Standard, 300_000, false),
            Band::Base
        );
        assert_eq!(
            prices.band(ServiceTier::Flex, 300_000, false),
            over(100_000)
        );
    }

    #[rstest]
    fn batch_bands_never_pick_the_online_band() {
        let prices = prices(&[(
            INPUT,
            Channel::Batch,
            ServiceTier::Standard,
            over(100_000),
            1.0,
        )]);
        assert_eq!(
            prices.band(ServiceTier::Standard, 300_000, false),
            Band::Base
        );
    }

    #[rstest]
    fn the_highest_band_passed_wins() {
        let prices = prices(&[
            (
                INPUT,
                Channel::Online,
                ServiceTier::Standard,
                over(100_000),
                1.0,
            ),
            (
                OUTPUT,
                Channel::Online,
                ServiceTier::Standard,
                over(200_000),
                1.0,
            ),
        ]);
        assert_eq!(
            prices.band(ServiceTier::Standard, 300_000, false),
            over(200_000)
        );
    }

    #[rstest]
    #[case::tier_base(INPUT, ServiceTier::Priority, Band::Base, Some(2.0))]
    #[case::standard_band_beats_tier_base(INPUT, ServiceTier::Priority, over(200_000), Some(10.0))]
    #[case::tier_band(OUTPUT, ServiceTier::Priority, over(200_000), Some(30.0))]
    #[case::standard_base_when_nothing_else(OUTPUT, ServiceTier::Flex, over(200_000), Some(3.0))]
    #[case::cache_read_falls_back_to_the_resolved_input(
        CACHE_READ,
        ServiceTier::Priority,
        over(200_000),
        Some(10.0)
    )]
    #[case::one_hour_write_falls_back_through_write_to_input(
        CACHE_WRITE_HOUR,
        ServiceTier::Priority,
        Band::Base,
        Some(2.0)
    )]
    #[case::audio_falls_back_to_input(
        Meter::Input(InputUnit::Token(TokenKind::Audio)),
        ServiceTier::Standard,
        Band::Base,
        Some(1.0)
    )]
    fn rate_walks_tier_then_band(
        priority_with_a_standard_band: Prices,
        #[case] meter: Meter,
        #[case] tier: ServiceTier,
        #[case] band: Band,
        #[case] expected: Option<f64>,
    ) {
        assert_eq!(
            priority_with_a_standard_band
                .rate(&meter, tier, band)
                .map(UnitPrice::get),
            expected
        );
    }

    #[rstest]
    fn an_explicit_cache_price_wins_over_the_input_fallback() {
        let prices = prices(&[
            (
                INPUT,
                Channel::Online,
                ServiceTier::Standard,
                Band::Base,
                1.0,
            ),
            (
                CACHE_READ,
                Channel::Online,
                ServiceTier::Standard,
                Band::Base,
                0.0,
            ),
        ]);
        assert_eq!(
            prices
                .rate(&CACHE_READ, ServiceTier::Standard, Band::Base)
                .map(UnitPrice::get),
            Some(0.0)
        );
    }

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
