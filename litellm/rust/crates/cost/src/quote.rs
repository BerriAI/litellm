use crate::call::CallContext;
use crate::money::UnitPrice;
use crate::price::{
    Band, BandEdge, CacheReadFallback, CacheReadMode, CacheTtl, Channel, InputUnit, Meter,
    OffPeakRates, OutputUnit, Price, Prices, ServiceTier, TokenKind, When,
};

/// A row's rates for one call, with the service tier, prompt size and off-peak window settled
#[derive(Clone, Copy, Debug)]
pub struct Quote<'a> {
    prices: &'a Prices,
    tier: ServiceTier,
    prompt_tokens: u64,
    /// The off-peak rates, only while a window is open
    off_peak: Option<&'a OffPeakRates>,
    /// Whether this crate prices the call at all
    priced_here: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Rate {
    pub unit_price: UnitPrice,
    /// The meter whose rate was borrowed, when this meter had none
    pub fallback_from: Option<Meter>,
}

impl Prices {
    /// A batch or decisions call, or one with custom pricing, gets a quote with no rates. Python still prices those
    pub fn quote(&self, call: &CallContext, prompt_tokens: u64) -> Quote<'_> {
        Quote {
            priced_here: call.channel == Channel::Online && call.overlay.is_none(),
            prices: self,
            tier: call.tier,
            prompt_tokens,
            off_peak: self
                .off_peak()
                .filter(|off_peak| off_peak.schedule().is_open(call.at))
                .map(|off_peak| off_peak.rates()),
        }
    }
}

impl Quote<'_> {
    /// `None` when the row has no price for `meter` and nothing to borrow one from
    pub fn rate(&self, meter: &Meter) -> Option<Rate> {
        if !self.priced_here {
            return None;
        }
        match meter {
            Meter::Input(InputUnit::Token(TokenKind::Text)) => self
                .off_peak(|rates| rates.input)
                .or_else(|| self.listed(meter))
                .map(Rate::own),
            Meter::Output(OutputUnit::Token(TokenKind::Text)) => self.text_output(),
            Meter::Output(OutputUnit::ReasoningToken) => self.reasoning(),
            Meter::CacheRead {
                kind: TokenKind::Text,
                mode: CacheReadMode::Implicit,
            } => self.cache_read(),
            Meter::CacheRead {
                kind: TokenKind::Text,
                mode: CacheReadMode::Explicit,
            } => self.explicit_cache_read(),
            Meter::CacheWrite {
                kind: TokenKind::Text,
                ttl: CacheTtl::Default,
            } => self
                .off_peak(|rates| rates.cache_write)
                .or_else(|| self.listed(meter))
                .map(Rate::own)
                .or_else(|| self.borrowed(&Meter::TEXT_INPUT)),
            Meter::CacheWrite {
                kind: TokenKind::Text,
                ttl: CacheTtl::OneHour | CacheTtl::TwelveHours,
            } => self
                .long_lived_write(meter)
                .map(Rate::own)
                .or_else(|| self.borrowed(&Meter::CACHE_WRITE)),
            Meter::CacheRead { mode, .. } => self.listed(meter).map(Rate::own).or_else(|| {
                self.borrowed(&Meter::CacheRead {
                    kind: TokenKind::Text,
                    mode: *mode,
                })
            }),
            Meter::Input(InputUnit::Token(_)) => self
                .listed(meter)
                .map(Rate::own)
                .or_else(|| self.borrowed(&Meter::TEXT_INPUT)),
            Meter::Output(OutputUnit::Token(_)) => self
                .listed(meter)
                .map(Rate::own)
                .or_else(|| self.borrowed(&Meter::TEXT_OUTPUT)),
            _ => self.base(meter, ServiceTier::Standard).map(Rate::own),
        }
    }

    /// An output price that is missing or 0.0 at the base band borrows the image token price,
    /// since an image generation row prices every output token as an image token
    fn text_output(&self) -> Option<Rate> {
        const IMAGE: Meter = Meter::Output(OutputUnit::Token(TokenKind::Image));
        if let Some(rate) = self.off_peak(|rates| rates.output) {
            return Some(Rate::own(rate));
        }
        match self.highest(&Meter::TEXT_OUTPUT, |_| true) {
            Some((when, rate)) if when.band != Band::Base || rate.get() != 0.0 => {
                Some(Rate::own(rate))
            }
            base => self
                .base(&IMAGE, ServiceTier::Standard)
                .map(|rate| Rate {
                    unit_price: rate,
                    fallback_from: Some(IMAGE),
                })
                .or(base.map(|(_, rate)| Rate::own(rate))),
        }
    }

    /// A tier that prices output but not reasoning bills reasoning as output, ahead of the
    /// standard reasoning price
    fn reasoning(&self) -> Option<Rate> {
        if let Some(rate) = self.off_peak(|rates| rates.reasoning) {
            return Some(Rate::own(rate));
        }
        let tier_prices_output = self.tier != ServiceTier::Standard
            && self.base(&Meter::TEXT_OUTPUT, self.tier).is_some();
        match self.highest(&Meter::REASONING, |_| true) {
            Some((when, rate)) if !(tier_prices_output && when == When::default()) => {
                Some(Rate::own(rate))
            }
            _ => self.borrowed(&Meter::TEXT_OUTPUT),
        }
    }

    /// A Fireworks row with no cache-read price reads at half its input price, off-peak included
    fn cache_read(&self) -> Option<Rate> {
        let from_input = |rate: UnitPrice| Rate {
            unit_price: rate.halved(),
            fallback_from: Some(Meter::TEXT_INPUT),
        };
        let input_to_halve = match self.prices.billing().cache_read_fallback {
            CacheReadFallback::HalfInputRate => {
                self.base(&Meter::TEXT_INPUT, ServiceTier::Standard)
            }
            CacheReadFallback::InputRate => None,
        };
        self.off_peak(|rates| rates.cache_read)
            .map(Rate::own)
            .or_else(|| {
                let (off_peak, input) = self.off_peak.zip(input_to_halve)?;
                Some(from_input(off_peak.input.unwrap_or(input)))
            })
            .or_else(|| self.listed(&Meter::CACHE_READ).map(Rate::own))
            .or_else(|| input_to_halve.map(from_input))
            .or_else(|| self.borrowed(&Meter::TEXT_INPUT))
    }

    /// A row with no explicit price anywhere reads an explicit cache at the implicit rate. One
    /// with an explicit price lets an off-peak implicit price stand in for a missing off-peak
    /// explicit price
    fn explicit_cache_read(&self) -> Option<Rate> {
        let off_peak_explicit = self.off_peak(|rates| rates.explicit_cache_read);
        let listed = self.listed(&Meter::EXPLICIT_CACHE_READ);
        off_peak_explicit
            .or_else(|| listed.and(self.off_peak(|rates| rates.cache_read)))
            .or(listed)
            .map(Rate::own)
            .or_else(|| self.borrowed(&Meter::CACHE_READ))
    }

    /// A long-lived write reads a band only when its own tier prices it there. Main falls back to
    /// the standard tier at the base band but not inside a band
    fn long_lived_write(&self, meter: &Meter) -> Option<UnitPrice> {
        self.highest(meter, |when| {
            when.tier == self.tier || when.band == Band::Base
        })
        .map(|(_, rate)| rate)
    }

    fn off_peak(&self, rate: fn(&OffPeakRates) -> Option<UnitPrice>) -> Option<UnitPrice> {
        self.off_peak.and_then(rate)
    }

    fn base(&self, meter: &Meter, tier: ServiceTier) -> Option<UnitPrice> {
        self.prices.get(&Price::online(
            meter.clone(),
            When {
                tier,
                band: Band::Base,
            },
        ))
    }

    fn borrowed(&self, from: &Meter) -> Option<Rate> {
        let rate = self.rate(from)?;
        Some(Rate {
            unit_price: rate.unit_price,
            fallback_from: rate.fallback_from.or_else(|| Some(from.clone())),
        })
    }

    fn reads(&self, tier: ServiceTier) -> bool {
        tier == self.tier || tier == ServiceTier::Standard
    }

    fn listed(&self, meter: &Meter) -> Option<UnitPrice> {
        self.highest(meter, |_| true).map(|(_, rate)| rate)
    }

    /// The meter's online price at the highest of its own bands the prompt reaches, the call's
    /// tier ahead of standard within a band. `counts` narrows which of those prices may be read
    fn highest(&self, meter: &Meter, counts: impl Fn(When) -> bool) -> Option<(When, UnitPrice)> {
        self.prices
            .iter()
            .filter(|(price, _)| {
                price.meter == *meter
                    && price.channel == Channel::Online
                    && self.reads(price.when.tier)
                    && counts(price.when)
                    && self.reaches(price.when.band)
            })
            .max_by_key(|(price, _)| (price.when.band, price.when.tier == self.tier))
            .map(|(price, rate)| (price.when, rate))
    }

    fn reaches(&self, band: Band) -> bool {
        match band {
            Band::Base => true,
            Band::OverPromptTokens(floor) => {
                self.prompt_tokens > floor.get()
                    || (self.prices.billing().band_edge == BandEdge::Inclusive
                        && self.prompt_tokens == floor.get())
            }
        }
    }
}

impl Rate {
    fn own(unit_price: UnitPrice) -> Self {
        Self {
            unit_price,
            fallback_from: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::price::Billing;
    use crate::schedule::{Schedule, Window};
    use crate::testing::{
        AUDIO_INPUT, BASE, CACHE_WRITE_HOUR, IMAGE_OUTPUT, call, off_peak_all_day, over, prices,
        prices_with, quoted, tier_call,
    };

    use ServiceTier::{Flex, Priority, Standard};

    const INPUT: Meter = Meter::TEXT_INPUT;
    const OUTPUT: Meter = Meter::TEXT_OUTPUT;
    const REASONING: Meter = Meter::REASONING;
    const CACHE_READ: Meter = Meter::CACHE_READ;
    const EXPLICIT: Meter = Meter::EXPLICIT_CACHE_READ;
    const CACHE_WRITE: Meter = Meter::CACHE_WRITE;

    fn inclusive() -> Billing {
        Billing {
            band_edge: BandEdge::Inclusive,
            ..Billing::default()
        }
    }

    #[rstest]
    #[case::below(199_999, Billing::default(), 1.0)]
    #[case::at_the_floor(200_000, Billing::default(), 1.0)]
    #[case::at_the_floor_inclusive(200_000, inclusive(), 10.0)]
    #[case::above(200_001, Billing::default(), 10.0)]
    fn a_band_starts_above_its_floor_unless_the_row_is_inclusive(
        #[case] prompt_tokens: u64,
        #[case] billing: Billing,
        #[case] expected: f64,
    ) {
        let prices = prices_with(
            &[
                (INPUT, Standard, BASE, 1.0),
                (INPUT, Standard, over(200_000), 10.0),
            ],
            None,
            billing,
        );
        assert_eq!(
            quoted(&prices, &call(), prompt_tokens, &INPUT),
            Some((expected, None))
        );
    }

    #[rstest]
    #[case::tier_base(INPUT, Priority, 1_000, 2.0)]
    #[case::standard_band_beats_tier_base(INPUT, Priority, 300_000, 10.0)]
    #[case::tier_band_beats_standard_band(OUTPUT, Priority, 300_000, 30.0)]
    #[case::another_tiers_band_is_not_read(OUTPUT, Flex, 300_000, 3.0)]
    #[case::standard_base_for_a_tier_the_row_lacks(INPUT, Flex, 1_000, 1.0)]
    fn a_rate_reads_its_highest_band_then_the_tier(
        #[case] meter: Meter,
        #[case] tier: ServiceTier,
        #[case] prompt_tokens: u64,
        #[case] expected: f64,
    ) {
        let prices = prices(&[
            (INPUT, Standard, BASE, 1.0),
            (INPUT, Priority, BASE, 2.0),
            (INPUT, Standard, over(200_000), 10.0),
            (OUTPUT, Standard, BASE, 3.0),
            (OUTPUT, Priority, over(200_000), 30.0),
        ]);
        assert_eq!(
            quoted(&prices, &tier_call(tier), prompt_tokens, &meter),
            Some((expected, None))
        );
    }

    #[rstest]
    fn a_meter_without_a_price_in_the_top_band_keeps_its_own_lower_band() {
        let prices = prices(&[
            (INPUT, Standard, BASE, 1.0),
            (INPUT, Standard, over(128_000), 2.0),
            (INPUT, Standard, over(200_000), 3.0),
            (OUTPUT, Standard, BASE, 10.0),
            (OUTPUT, Standard, over(128_000), 20.0),
        ]);
        assert_eq!(quoted(&prices, &call(), 250_000, &INPUT), Some((3.0, None)));
        assert_eq!(
            quoted(&prices, &call(), 250_000, &OUTPUT),
            Some((20.0, None))
        );
    }

    #[rstest]
    fn audio_input_reads_its_own_bands_when_text_has_none() {
        let prices = prices(&[
            (INPUT, Standard, BASE, 1.0),
            (AUDIO_INPUT, Standard, BASE, 5.0),
            (AUDIO_INPUT, Standard, over(200_000), 7.0),
            (AUDIO_INPUT, Priority, over(200_000), 9.0),
        ]);
        assert_eq!(
            quoted(&prices, &call(), 200_001, &AUDIO_INPUT),
            Some((7.0, None))
        );
        assert_eq!(
            quoted(&prices, &tier_call(Priority), 200_001, &AUDIO_INPUT),
            Some((9.0, None))
        );
        assert_eq!(quoted(&prices, &call(), 200_001, &INPUT), Some((1.0, None)));
    }

    #[rstest]
    #[case::audio_input(AUDIO_INPUT, INPUT, 1.0)]
    #[case::video_input(Meter::Input(InputUnit::Token(TokenKind::Video)), INPUT, 1.0)]
    #[case::audio_output(Meter::Output(OutputUnit::Token(TokenKind::Audio)), OUTPUT, 3.0)]
    #[case::image_output(IMAGE_OUTPUT, OUTPUT, 3.0)]
    #[case::cache_read(CACHE_READ, INPUT, 1.0)]
    #[case::cache_write(CACHE_WRITE, INPUT, 1.0)]
    #[case::one_hour_write_through_write_to_input(CACHE_WRITE_HOUR, INPUT, 1.0)]
    #[case::cached_audio_through_cache_read_to_input(
        Meter::CacheRead { kind: TokenKind::Audio, mode: CacheReadMode::Implicit },
        INPUT,
        1.0
    )]
    #[case::explicit_cache_read_through_implicit_to_input(EXPLICIT, INPUT, 1.0)]
    #[case::reasoning(REASONING, OUTPUT, 3.0)]
    fn a_token_meter_without_a_price_borrows_the_text_rate(
        #[case] meter: Meter,
        #[case] from: Meter,
        #[case] expected: f64,
    ) {
        let prices = prices(&[(INPUT, Standard, BASE, 1.0), (OUTPUT, Standard, BASE, 3.0)]);
        assert_eq!(
            quoted(&prices, &call(), 10, &meter),
            Some((expected, Some(from)))
        );
    }

    #[rstest]
    fn a_borrowed_rate_names_the_meter_that_had_the_price() {
        let prices = prices(&[
            (INPUT, Standard, BASE, 1.0),
            (CACHE_WRITE, Standard, BASE, 1.25),
            (CACHE_READ, Standard, BASE, 0.1),
        ]);
        assert_eq!(
            quoted(&prices, &call(), 10, &CACHE_WRITE_HOUR),
            Some((1.25, Some(CACHE_WRITE)))
        );
        assert_eq!(
            quoted(&prices, &call(), 10, &EXPLICIT),
            Some((0.1, Some(CACHE_READ)))
        );
    }

    #[rstest]
    #[case::character(Meter::Input(InputUnit::Character))]
    #[case::image(Meter::Input(InputUnit::Image))]
    #[case::audio_second(Meter::Input(InputUnit::AudioSecond))]
    #[case::query(Meter::Input(InputUnit::Query { over_results: None }))]
    fn a_unit_meter_reads_only_its_standard_base_price(#[case] meter: Meter) {
        let prices = prices(&[
            (INPUT, Standard, BASE, 1.0),
            (meter.clone(), Standard, BASE, 4.0),
            (meter.clone(), Standard, over(128_000), 8.0),
            (meter.clone(), Priority, BASE, 16.0),
        ]);
        assert_eq!(
            quoted(&prices, &tier_call(Priority), 200_000, &meter),
            Some((4.0, None))
        );
    }

    #[rstest]
    fn a_unit_meter_without_a_price_has_no_rate() {
        let prices = prices(&[(INPUT, Standard, BASE, 1.0)]);
        assert_eq!(
            quoted(&prices, &call(), 10, &Meter::Input(InputUnit::Character)),
            None
        );
    }

    #[rstest]
    #[case::missing(&[(IMAGE_OUTPUT, Standard, BASE, 4.0)], 10, Some((4.0, Some(IMAGE_OUTPUT))))]
    #[case::zero(
        &[(OUTPUT, Standard, BASE, 0.0), (IMAGE_OUTPUT, Standard, BASE, 4.0)],
        10,
        Some((4.0, Some(IMAGE_OUTPUT)))
    )]
    #[case::zero_with_no_image_price(&[(OUTPUT, Standard, BASE, 0.0)], 10, Some((0.0, None)))]
    #[case::priced(
        &[(OUTPUT, Standard, BASE, 3.0), (IMAGE_OUTPUT, Standard, BASE, 4.0)],
        10,
        Some((3.0, None))
    )]
    #[case::a_zero_band_price_stays_free(
        &[
            (OUTPUT, Standard, BASE, 0.0),
            (OUTPUT, Standard, over(100), 0.0),
            (IMAGE_OUTPUT, Standard, BASE, 4.0),
        ],
        101,
        Some((0.0, None))
    )]
    fn output_missing_or_free_at_the_base_borrows_the_image_token_rate(
        #[case] rates: &[(Meter, ServiceTier, Band, f64)],
        #[case] prompt_tokens: u64,
        #[case] expected: Option<(f64, Option<Meter>)>,
    ) {
        assert_eq!(
            quoted(&prices(rates), &call(), prompt_tokens, &OUTPUT),
            expected
        );
    }

    #[rstest]
    #[case::own_tier_price(
        &[(REASONING, Standard, BASE, 5.0), (REASONING, Priority, BASE, 6.0), (OUTPUT, Priority, BASE, 9.0)],
        Priority,
        (6.0, None)
    )]
    #[case::tier_output_ahead_of_standard_reasoning(
        &[(REASONING, Standard, BASE, 5.0), (OUTPUT, Standard, BASE, 3.0), (OUTPUT, Priority, BASE, 9.0)],
        Priority,
        (9.0, Some(OUTPUT))
    )]
    #[case::standard_reasoning_when_the_tier_prices_no_output(
        &[(REASONING, Standard, BASE, 5.0), (OUTPUT, Standard, BASE, 3.0)],
        Priority,
        (5.0, None)
    )]
    #[case::standard_call_reads_standard_reasoning(
        &[(REASONING, Standard, BASE, 5.0), (OUTPUT, Standard, BASE, 3.0), (OUTPUT, Priority, BASE, 9.0)],
        Standard,
        (5.0, None)
    )]
    fn reasoning_reads_the_tier_before_the_standard_price(
        #[case] rates: &[(Meter, ServiceTier, Band, f64)],
        #[case] tier: ServiceTier,
        #[case] expected: (f64, Option<Meter>),
    ) {
        assert_eq!(
            quoted(&prices(rates), &tier_call(tier), 10, &REASONING),
            Some(expected)
        );
    }

    #[rstest]
    fn a_band_that_prices_reasoning_wins_over_the_base_reasoning_price() {
        let prices = prices(&[
            (OUTPUT, Standard, BASE, 3.0),
            (REASONING, Standard, BASE, 5.0),
            (OUTPUT, Standard, over(32_000), 6.0),
            (REASONING, Standard, over(32_000), 8.0),
        ]);
        assert_eq!(
            quoted(&prices, &call(), 32_001, &REASONING),
            Some((8.0, None))
        );
        assert_eq!(
            quoted(&prices, &call(), 32_000, &REASONING),
            Some((5.0, None))
        );
    }

    #[rstest]
    #[case::input(INPUT, (0.5, None))]
    #[case::output(OUTPUT, (1.5, None))]
    #[case::cache_read_has_its_own_off_peak_price(CACHE_READ, (0.05, None))]
    #[case::cache_write_borrows_the_off_peak_input(CACHE_WRITE, (0.5, Some(INPUT)))]
    #[case::one_hour_write_follows_the_off_peak_write(CACHE_WRITE_HOUR, (0.5, Some(INPUT)))]
    #[case::reasoning_keeps_its_standard_price(REASONING, (5.0, None))]
    #[case::audio_input_keeps_its_standard_price(AUDIO_INPUT, (7.0, None))]
    fn an_open_window_swaps_in_the_off_peak_rates(
        #[case] meter: Meter,
        #[case] expected: (f64, Option<Meter>),
    ) {
        let prices = prices_with(
            &[
                (INPUT, Standard, BASE, 1.0),
                (INPUT, Standard, over(100), 2.0),
                (OUTPUT, Standard, BASE, 3.0),
                (CACHE_READ, Standard, BASE, 0.1),
                (REASONING, Standard, BASE, 5.0),
                (AUDIO_INPUT, Standard, BASE, 7.0),
            ],
            off_peak_all_day(OffPeakRates {
                input: UnitPrice::new(0.5),
                output: UnitPrice::new(1.5),
                cache_read: UnitPrice::new(0.05),
                ..OffPeakRates::default()
            }),
            Billing::default(),
        );
        assert_eq!(quoted(&prices, &call(), 200, &meter), Some(expected));
    }

    #[rstest]
    fn a_priced_cache_write_is_not_swapped_for_the_off_peak_input() {
        let prices = prices_with(
            &[
                (INPUT, Standard, BASE, 1.0),
                (CACHE_WRITE, Standard, BASE, 1.25),
                (CACHE_WRITE_HOUR, Standard, BASE, 2.0),
            ],
            off_peak_all_day(OffPeakRates {
                input: UnitPrice::new(0.5),
                ..OffPeakRates::default()
            }),
            Billing::default(),
        );
        assert_eq!(
            quoted(&prices, &call(), 10, &CACHE_WRITE),
            Some((1.25, None))
        );
        assert_eq!(
            quoted(&prices, &call(), 10, &CACHE_WRITE_HOUR),
            Some((2.0, None))
        );
    }

    #[rstest]
    fn off_peak_reasoning_without_its_own_price_follows_the_off_peak_output() {
        let prices = prices_with(
            &[(OUTPUT, Standard, BASE, 3.0)],
            off_peak_all_day(OffPeakRates {
                output: UnitPrice::new(1.5),
                ..OffPeakRates::default()
            }),
            Billing::default(),
        );
        assert_eq!(
            quoted(&prices, &call(), 10, &REASONING),
            Some((1.5, Some(OUTPUT)))
        );
    }

    #[rstest]
    fn a_closed_window_leaves_the_standard_rates() {
        let prices = prices_with(
            &[(INPUT, Standard, BASE, 1.0)],
            crate::price::OffPeak::new(
                Schedule {
                    windows: vec![Window {
                        hours: vec!["01:00-02:00".parse().unwrap()],
                        weekdays: None,
                    }],
                    weekday_timezone: None,
                },
                OffPeakRates {
                    input: UnitPrice::new(0.5),
                    ..OffPeakRates::default()
                },
            ),
            Billing::default(),
        );
        assert_eq!(quoted(&prices, &call(), 10, &INPUT), Some((1.0, None)));
    }

    #[rstest]
    #[case::no_explicit_price_reads_at_the_implicit_rate(
        &[(INPUT, Standard, BASE, 1.0), (CACHE_READ, Standard, BASE, 0.2)],
        OffPeakRates::default(),
        (0.2, Some(CACHE_READ))
    )]
    #[case::explicit_price(
        &[(CACHE_READ, Standard, BASE, 0.2), (EXPLICIT, Standard, BASE, 0.1)],
        OffPeakRates::default(),
        (0.1, None)
    )]
    #[case::off_peak_explicit_price(
        &[(CACHE_READ, Standard, BASE, 0.2), (EXPLICIT, Standard, BASE, 0.1)],
        OffPeakRates {
            cache_read: UnitPrice::new(0.07),
            explicit_cache_read: UnitPrice::new(0.03),
            ..OffPeakRates::default()
        },
        (0.03, None)
    )]
    #[case::off_peak_implicit_price_stands_in(
        &[(CACHE_READ, Standard, BASE, 0.2), (EXPLICIT, Standard, BASE, 0.1)],
        OffPeakRates { cache_read: UnitPrice::new(0.07), ..OffPeakRates::default() },
        (0.07, None)
    )]
    #[case::off_peak_input_does_not_replace_an_explicit_price(
        &[(INPUT, Standard, BASE, 1.0), (EXPLICIT, Standard, BASE, 0.1)],
        OffPeakRates { input: UnitPrice::new(0.5), ..OffPeakRates::default() },
        (0.1, None)
    )]
    #[case::off_peak_explicit_price_alone(
        &[(INPUT, Standard, BASE, 1.0)],
        OffPeakRates { explicit_cache_read: UnitPrice::new(0.03), ..OffPeakRates::default() },
        (0.03, None)
    )]
    fn an_explicit_cache_read_needs_an_explicit_price(
        #[case] rates: &[(Meter, ServiceTier, Band, f64)],
        #[case] off_peak: OffPeakRates,
        #[case] expected: (f64, Option<Meter>),
    ) {
        let prices = prices_with(rates, off_peak_all_day(off_peak), Billing::default());
        assert_eq!(quoted(&prices, &call(), 10, &EXPLICIT), Some(expected));
    }

    #[rstest]
    #[case::standard_reads_its_band(Standard, 200_001, 4.0)]
    #[case::a_tier_reads_its_own_band(Priority, 200_001, 8.0)]
    #[case::a_tier_never_reads_the_standard_band(Flex, 200_001, 2.0)]
    #[case::a_tier_reads_the_standard_base(Flex, 10, 2.0)]
    fn a_long_lived_write_reads_a_band_only_from_its_own_tier(
        #[case] tier: ServiceTier,
        #[case] prompt_tokens: u64,
        #[case] expected: f64,
    ) {
        let prices = prices(&[
            (INPUT, Standard, BASE, 1.0),
            (CACHE_WRITE_HOUR, Standard, BASE, 2.0),
            (CACHE_WRITE_HOUR, Standard, over(200_000), 4.0),
            (CACHE_WRITE_HOUR, Priority, over(200_000), 8.0),
        ]);
        assert_eq!(
            quoted(&prices, &tier_call(tier), prompt_tokens, &CACHE_WRITE_HOUR),
            Some((expected, None))
        );
    }

    fn fireworks() -> Billing {
        Billing {
            cache_read_fallback: CacheReadFallback::HalfInputRate,
            ..Billing::default()
        }
    }

    #[rstest]
    #[case::half_the_base_input(&[(INPUT, Standard, BASE, 1.0)], None, 10, (0.5, Some(INPUT)))]
    #[case::half_the_base_input_inside_a_band(
        &[(INPUT, Standard, BASE, 1.0), (INPUT, Standard, over(100), 4.0)],
        None,
        101,
        (0.5, Some(INPUT))
    )]
    #[case::a_tier_cache_read_price_wins(
        &[(INPUT, Standard, BASE, 1.0), (CACHE_READ, Priority, BASE, 0.3)],
        None,
        10,
        (0.3, None)
    )]
    #[case::half_the_off_peak_input(
        &[(INPUT, Standard, BASE, 1.0)],
        Some(OffPeakRates { input: UnitPrice::new(0.6), ..OffPeakRates::default() }),
        10,
        (0.3, Some(INPUT))
    )]
    #[case::half_the_base_input_when_off_peak_prices_no_input(
        &[(INPUT, Standard, BASE, 1.0)],
        Some(OffPeakRates { output: UnitPrice::new(0.6), ..OffPeakRates::default() }),
        10,
        (0.5, Some(INPUT))
    )]
    #[case::off_peak_cache_read_price_wins(
        &[(INPUT, Standard, BASE, 1.0)],
        Some(OffPeakRates { cache_read: UnitPrice::new(0.2), ..OffPeakRates::default() }),
        10,
        (0.2, None)
    )]
    fn a_fireworks_row_without_a_cache_read_price_reads_at_half_input(
        #[case] rates: &[(Meter, ServiceTier, Band, f64)],
        #[case] off_peak: Option<OffPeakRates>,
        #[case] prompt_tokens: u64,
        #[case] expected: (f64, Option<Meter>),
    ) {
        let prices = prices_with(rates, off_peak.and_then(off_peak_all_day), fireworks());
        assert_eq!(
            quoted(&prices, &tier_call(Priority), prompt_tokens, &CACHE_READ),
            Some(expected)
        );
    }

    #[rstest]
    #[case::batch(CallContext { channel: Channel::Batch, ..call() })]
    #[case::decisions(CallContext { channel: Channel::Decisions, ..call() })]
    #[case::custom_pricing(CallContext { overlay: Some(Prices::default()), ..call() })]
    fn a_call_this_crate_cannot_price_yet_has_no_rates(#[case] call: CallContext) {
        let prices = prices(&[(INPUT, Standard, BASE, 1.0), (OUTPUT, Standard, BASE, 3.0)]);
        assert_eq!(quoted(&prices, &call, 10, &INPUT), None);
        assert_eq!(quoted(&prices, &call, 10, &REASONING), None);
    }

    #[rstest]
    fn batch_prices_are_never_quoted() {
        let prices = Prices::new(
            [(
                Price {
                    meter: INPUT,
                    channel: Channel::Batch,
                    when: When::default(),
                },
                UnitPrice::new(1.0).unwrap(),
            )]
            .into(),
            None,
            Billing::default(),
        );
        assert_eq!(quoted(&prices, &call(), 10, &INPUT), None);
    }
}
