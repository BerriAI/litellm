use std::collections::BTreeSet;
use std::num::NonZeroU64;

use crate::calculate::{Usage, When, batch_direct_rate, resolve};
use crate::money::UnitPrice;
use crate::price::{
    Band, CacheReadMode, CacheTtl, Channel, InputUnit, Meter, OutputUnit, Price, PriceSource,
    Prices, ServiceTier, TokenKind,
};

#[derive(Clone, Copy, Debug, Default)]
pub enum SelectionPolicy<'a> {
    #[default]
    Generic,
    SdkTokens(&'a TokenPriceSources),
    SdkOther {
        tokens: &'a TokenPriceSources,
        context: &'a crate::OtherContext,
    },
    DirectionalTokens {
        threshold: NonZeroU64,
    },
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TokenPriceSources {
    pub flat: BTreeSet<Price>,
    pub tiers: Vec<TokenTier>,
    pub off_peak_cache_read_declared: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenTier {
    pub range: [u64; 2],
    pub meters: BTreeSet<Meter>,
}

impl<'a> SelectionPolicy<'a> {
    pub(crate) fn resolve_charge(
        self,
        prices: &Prices,
        meter: &Meter,
        band: Band,
        usage: &Usage,
        when: &When,
    ) -> Option<crate::ResolvedCharge> {
        match self {
            Self::SdkOther { tokens, context } => {
                context.resolve(prices, meter, band, usage, when, tokens)
            }
            _ => self
                .resolve(prices, meter, band, usage, when)
                .map(Into::into),
        }
    }

    pub(crate) fn other(self) -> Option<&'a crate::OtherContext> {
        match self {
            Self::SdkOther { context, .. } => Some(context),
            _ => None,
        }
    }

    pub(crate) fn resolve(
        self,
        prices: &Prices,
        meter: &Meter,
        band: Band,
        usage: &Usage,
        when: &When,
    ) -> Option<(Meter, UnitPrice)> {
        if let Self::SdkOther { .. } = self {
            return self
                .resolve_charge(prices, meter, band, usage, when)
                .map(|charge| (charge.meter, charge.price));
        }
        if let Self::DirectionalTokens { threshold } = self {
            return directional_token_rate(prices, meter, usage, threshold);
        }
        let Self::SdkTokens(sources) = self else {
            return resolve(prices, meter, band, usage, when);
        };
        if when.channel == Channel::Batch && meter.is_token() {
            return sdk_batch_rate(prices, meter, usage);
        }
        if when.channel != Channel::Online {
            return resolve(prices, meter, band, usage, when);
        }
        let output = Meter::Output(OutputUnit::Token(TokenKind::Text));
        match meter {
            Meter::Output(OutputUnit::ReasoningToken) => {
                let off_peak = prices
                    .off_peak()
                    .filter(|off_peak| off_peak.schedule().contains_sdk_tokens(when.at))
                    .and_then(|off_peak| off_peak.rates().reasoning);
                if let Some(rate) = off_peak {
                    return Some((meter.clone(), rate));
                }
                if let Some(tier) = sources.priced_tier(usage.prompt_tokens)
                    && (tier.meters.contains(meter) || tier.meters.contains(&output))
                {
                    return tier
                        .rate(prices, meter)
                        .map(|rate| (meter.clone(), rate))
                        .or_else(|| tier.rate(prices, &output).map(|rate| (output, rate)));
                }
                if let Some(rate) = sources.flat_rate(prices, meter, when.tier) {
                    return Some((meter.clone(), rate));
                }
                if when.tier != ServiceTier::Standard
                    && sources.flat_rate(prices, &output, when.tier).is_some()
                {
                    return self.resolve(prices, &output, band, usage, when);
                }
                sources
                    .flat_rate(prices, meter, ServiceTier::Standard)
                    .map(|rate| (meter.clone(), rate))
                    .or_else(|| self.resolve(prices, &output, band, usage, when))
            }
            Meter::Output(OutputUnit::Token(kind)) if *kind != TokenKind::Text => sources
                .flat_rate(prices, meter, ServiceTier::Standard)
                .map(|rate| (meter.clone(), rate))
                .or_else(|| self.resolve(prices, &output, band, usage, when)),
            Meter::Input(InputUnit::Token(_))
            | Meter::Output(OutputUnit::Token(_))
            | Meter::CacheRead { .. }
            | Meter::CacheWrite { .. } => {
                if let Some(tier) = sources.priced_tier(usage.prompt_tokens)
                    && matches!(
                        meter,
                        Meter::Input(InputUnit::Token(TokenKind::Text))
                            | Meter::Output(OutputUnit::Token(TokenKind::Text))
                            | Meter::CacheRead {
                                kind: TokenKind::Text,
                                ..
                            }
                            | Meter::CacheWrite {
                                kind: TokenKind::Text,
                                ..
                            }
                    )
                {
                    return tier.resolve(prices, meter, when);
                }
                overlay(prices, meter, when)
                    .or_else(|| {
                        if meter.sets_prompt_band() {
                            sdk_flat_rate(prices, meter, usage, when.tier)
                        } else {
                            prices.direct_rate(meter, when.channel, when.tier, band)
                        }
                    })
                    .map(|rate| (meter.clone(), rate))
                    .or_else(|| self.resolve(prices, &meter.fallback()?, band, usage, when))
            }
            _ => resolve(
                prices,
                meter,
                prices.all_source_band(
                    when.channel,
                    when.tier,
                    usage.prompt_tokens,
                    usage.inclusive_threshold,
                ),
                usage,
                when,
            ),
        }
    }
}

fn sdk_flat_rate(
    prices: &Prices,
    meter: &Meter,
    usage: &Usage,
    tier: ServiceTier,
) -> Option<UnitPrice> {
    prices
        .iter()
        .filter(|(price, _)| {
            price.meter == *meter
                && price.channel == Channel::Online
                && (price.when.tier == tier || price.when.tier == ServiceTier::Standard)
                && match price.when.band {
                    Band::Base => true,
                    Band::OverPromptTokens(floor) => {
                        usage.prompt_tokens > floor.get()
                            || (usage.inclusive_threshold && usage.prompt_tokens == floor.get())
                    }
                }
        })
        .max_by_key(|(price, _)| (price.when.band, price.when.tier == tier))
        .map(|(_, rate)| rate)
}

fn directional_token_rate(
    prices: &Prices,
    meter: &Meter,
    usage: &Usage,
    threshold: NonZeroU64,
) -> Option<(Meter, UnitPrice)> {
    if !meter.is_token() {
        return None;
    }
    let source = match meter {
        Meter::Output(_) => Meter::Output(OutputUnit::Token(TokenKind::Text)),
        _ => Meter::TEXT_INPUT,
    };
    directional_rate(
        prices,
        source.clone(),
        usage
            .quantities
            .get(&source)
            .copied()
            .unwrap_or(crate::Quantity::Count(0)),
        threshold,
    )
}

pub(crate) fn directional_rate(
    prices: &Prices,
    source: Meter,
    quantity: crate::Quantity,
    threshold: NonZeroU64,
) -> Option<(Meter, UnitPrice)> {
    let base = Price::online(source.clone(), Default::default());
    let rate = quantity
        .exceeds(threshold.get())
        .then(|| {
            prices.get(&Price::online(
                source.clone(),
                crate::price::When {
                    band: Band::OverPromptTokens(threshold),
                    ..Default::default()
                },
            ))
        })
        .flatten()
        .or_else(|| prices.get(&base));
    rate.map(|rate| (source, rate))
}

fn sdk_batch_rate(prices: &Prices, meter: &Meter, usage: &Usage) -> Option<(Meter, UnitPrice)> {
    let base = |meter: &Meter, channel| {
        prices.get(&Price {
            meter: meter.clone(),
            channel,
            when: Default::default(),
        })
    };
    let half = |meter: &Meter| base(meter, Channel::Online).map(UnitPrice::half);
    let output = Meter::Output(OutputUnit::Token(TokenKind::Text));
    let text_output = || {
        batch_direct_rate(prices, &output, usage)
            .or_else(|| half(&output))
            .map(|rate| (output.clone(), rate))
    };
    let read = Meter::CacheRead {
        kind: TokenKind::Text,
        mode: CacheReadMode::Implicit,
    };
    let write = Meter::CacheWrite {
        kind: TokenKind::Text,
        ttl: CacheTtl::Default,
    };
    let batch_input = batch_direct_rate(prices, &Meter::TEXT_INPUT, usage);
    let input = || {
        batch_input
            .or_else(|| half(&Meter::TEXT_INPUT))
            .map(|rate| (Meter::TEXT_INPUT, rate))
    };
    match meter {
        Meter::Output(OutputUnit::Token(TokenKind::Image)) => base(meter, Channel::Batch)
            .or_else(|| half(meter))
            .map(|rate| (meter.clone(), rate))
            .or_else(text_output),
        Meter::Output(OutputUnit::Token(_) | OutputUnit::ReasoningToken) => text_output(),
        Meter::Input(InputUnit::Token(kind))
            if batch_input.is_some() && *kind != TokenKind::Text =>
        {
            base(meter, Channel::Batch)
                .map(|rate| (meter.clone(), rate))
                .or_else(input)
        }
        Meter::Input(InputUnit::Token(_)) => input(),
        Meter::CacheRead { kind, .. } if batch_input.is_some() => {
            let Some(rate) = batch_direct_rate(prices, &read, usage) else {
                return input();
            };
            let audio = Meter::CacheRead {
                kind: TokenKind::Audio,
                mode: CacheReadMode::Implicit,
            };
            if *kind == TokenKind::Audio
                && let Some(audio_rate) = base(&audio, Channel::Batch)
            {
                return Some((audio, audio_rate));
            }
            Some((read, rate))
        }
        Meter::CacheRead { .. } => half(&read).map(|rate| (read, rate)).or_else(input),
        Meter::CacheWrite { ttl, .. } if batch_input.is_some() => {
            let selected = Meter::CacheWrite {
                kind: TokenKind::Text,
                ttl: *ttl,
            };
            batch_direct_rate(prices, &selected, usage)
                .map(|rate| (selected, rate))
                .or_else(|| batch_direct_rate(prices, &write, usage).map(|rate| (write, rate)))
                .or_else(input)
        }
        Meter::CacheWrite { .. } => half(&write).map(|rate| (write, rate)).or_else(input),
        _ => None,
    }
}

fn overlay(prices: &Prices, meter: &Meter, when: &When) -> Option<UnitPrice> {
    prices
        .off_peak()
        .filter(|off_peak| off_peak.schedule().contains_sdk_tokens(when.at))
        .and_then(|off_peak| off_peak.rates().get(meter))
}

impl TokenPriceSources {
    fn flat_rate(&self, prices: &Prices, meter: &Meter, tier: ServiceTier) -> Option<UnitPrice> {
        let key = Price::online(
            meter.clone(),
            crate::price::When {
                tier,
                band: Band::Base,
            },
        );
        self.flat.contains(&key).then(|| prices.get(&key)).flatten()
    }

    fn priced_tier(&self, prompt_tokens: u64) -> Option<&TokenTier> {
        if prompt_tokens == 0 {
            return None;
        }
        self.tiers
            .iter()
            .filter(|tier| tier.range[0] < prompt_tokens && prompt_tokens <= tier.range[1])
            .min_by_key(|tier| tier.range[0])
            .or_else(|| self.tiers.iter().max_by_key(|tier| tier.range[0]))
            .filter(|tier| tier.meters.contains(&Meter::TEXT_INPUT))
    }
}

impl TokenTier {
    fn resolve(&self, prices: &Prices, meter: &Meter, when: &When) -> Option<(Meter, UnitPrice)> {
        if matches!(
            meter,
            Meter::CacheWrite {
                ttl: CacheTtl::OneHour | CacheTtl::TwelveHours,
                ..
            }
        ) {
            return self
                .rate(prices, meter)
                .map(|rate| (meter.clone(), rate))
                .or_else(|| self.resolve(prices, &meter.fallback()?, when));
        }
        if let Some(rate) = overlay(prices, meter, when).or_else(|| self.rate(prices, meter)) {
            return Some((meter.clone(), rate));
        }
        match meter {
            Meter::Output(OutputUnit::Token(TokenKind::Text)) if !self.meters.contains(meter) => {
                prices
                    .get(&Price::online(meter.clone(), Default::default()))
                    .map(|rate| (meter.clone(), rate))
            }
            Meter::CacheRead {
                mode: CacheReadMode::Explicit,
                ..
            } => self.resolve(prices, &meter.fallback()?, when),
            Meter::CacheRead { .. } | Meter::CacheWrite { .. } => self
                .rate(prices, &Meter::TEXT_INPUT)
                .map(|rate| (Meter::TEXT_INPUT, rate)),
            _ => None,
        }
    }

    fn rate(&self, prices: &Prices, meter: &Meter) -> Option<UnitPrice> {
        self.meters
            .contains(meter)
            .then(|| {
                prices.get_from(
                    &Price::online(
                        meter.clone(),
                        crate::price::When {
                            band: Band::over_prompt_tokens(self.range[0]),
                            ..Default::default()
                        },
                    ),
                    PriceSource::TokenTier(self.range[0]),
                )
            })
            .flatten()
    }
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};
    use time::OffsetDateTime;

    use super::*;
    use crate::price::{InputUnit, OffPeak, OffPeakRates, When as PriceWhen};
    use crate::schedule::{Schedule, Window};
    use crate::{Breakdown, Line, Priced, calculate, calculate_with_policy};

    const OUTPUT: Meter = Meter::Output(OutputUnit::Token(TokenKind::Text));
    const REASONING: Meter = Meter::Output(OutputUnit::ReasoningToken);
    const INPUT: Meter = Meter::TEXT_INPUT;
    const READ: Meter = Meter::CacheRead {
        kind: TokenKind::Text,
        mode: CacheReadMode::Implicit,
    };
    const EXPLICIT: Meter = Meter::CacheRead {
        kind: TokenKind::Text,
        mode: CacheReadMode::Explicit,
    };
    const WRITE: Meter = Meter::CacheWrite {
        kind: TokenKind::Text,
        ttl: CacheTtl::Default,
    };
    const HOUR: Meter = Meter::CacheWrite {
        kind: TokenKind::Text,
        ttl: CacheTtl::OneHour,
    };
    const TWELVE: Meter = Meter::CacheWrite {
        kind: TokenKind::Text,
        ttl: CacheTtl::TwelveHours,
    };

    #[fixture]
    fn tier_prices() -> (Prices, TokenPriceSources) {
        let flat = [INPUT, OUTPUT, READ, EXPLICIT, WRITE, HOUR, TWELVE]
            .into_iter()
            .map(|meter| rate(meter, ServiceTier::Standard, Band::Base, 9.0));
        let tier_rates = [
            (INPUT, 0, 1.0),
            (OUTPUT, 0, 2.0),
            (WRITE, 0, 3.0),
            (INPUT, 100, 4.0),
            (READ, 100, 0.0),
            (WRITE, 100, 6.0),
            (HOUR, 100, 0.0),
        ]
        .into_iter()
        .map(|(meter, start, value)| {
            let (price, rate) = rate(
                meter,
                ServiceTier::Standard,
                Band::over_prompt_tokens(start),
                value,
            );
            ((price, PriceSource::TokenTier(start)), rate)
        });
        let prices = Prices::from_sources(
            flat.map(|(price, rate)| ((price, PriceSource::Flat), rate))
                .chain(tier_rates)
                .collect(),
            OffPeak::new(
                Schedule {
                    windows: vec![Window {
                        hours: vec!["01:00-02:00".parse().unwrap()],
                        weekdays: None,
                    }],
                    weekday_timezone: None,
                },
                OffPeakRates {
                    input: UnitPrice::new(7.0),
                    output: UnitPrice::new(8.0),
                    cache_write: UnitPrice::new(0.0),
                    ..Default::default()
                },
            ),
        );
        let sources = TokenPriceSources {
            off_peak_cache_read_declared: false,
            flat: prices.iter().map(|(price, _)| price.clone()).collect(),
            tiers: vec![
                TokenTier {
                    range: [0, 100],
                    meters: [INPUT, OUTPUT, WRITE].into(),
                },
                TokenTier {
                    range: [100, 200],
                    meters: [INPUT, READ, WRITE, HOUR].into(),
                },
            ],
        };
        (prices, sources)
    }

    #[rstest]
    #[case::input(INPUT, 10, 0, INPUT, 1.0)]
    #[case::boundary(INPUT, 100, 0, INPUT, 1.0)]
    #[case::next(INPUT, 101, 0, INPUT, 4.0)]
    #[case::final_boundary(INPUT, 200, 0, INPUT, 4.0)]
    #[case::above_last(INPUT, 201, 0, INPUT, 4.0)]
    #[case::zero_prompt(INPUT, 0, 0, INPUT, 9.0)]
    #[case::read_ignores_flat(READ, 10, 0, INPUT, 1.0)]
    #[case::explicit_fallback(EXPLICIT, 10, 0, INPUT, 1.0)]
    #[case::write_ignores_flat(WRITE, 10, 0, WRITE, 3.0)]
    #[case::hour_ignores_flat(HOUR, 10, 0, WRITE, 3.0)]
    #[case::twelve_ignores_flat(TWELVE, 10, 0, WRITE, 3.0)]
    #[case::zero_read(READ, 101, 0, READ, 0.0)]
    #[case::explicit_fallback_zero(EXPLICIT, 101, 0, READ, 0.0)]
    #[case::zero_hour(HOUR, 101, 0, HOUR, 0.0)]
    #[case::output_falls_back_flat_not_previous_tier(OUTPUT, 101, 0, OUTPUT, 9.0)]
    #[case::input_offpeak(INPUT, 10, 3600, INPUT, 7.0)]
    #[case::read_fallback_precedes_input_offpeak(READ, 10, 3600, INPUT, 1.0)]
    #[case::default_write_offpeak(WRITE, 10, 3600, WRITE, 0.0)]
    #[case::hour_follows_write_offpeak(HOUR, 10, 3600, WRITE, 0.0)]
    #[case::twelve_follows_write_offpeak(TWELVE, 10, 3600, WRITE, 0.0)]
    #[case::explicit_hour_isolated_offpeak(HOUR, 101, 3600, HOUR, 0.0)]
    #[case::output_offpeak(OUTPUT, 10, 3600, OUTPUT, 8.0)]
    #[case::offpeak_end_excluded(WRITE, 10, 7200, WRITE, 3.0)]
    fn sdk_tier_cache_fallback_and_ttl_isolation(
        tier_prices: (Prices, TokenPriceSources),
        #[case] requested: Meter,
        #[case] prompt: u64,
        #[case] at: i64,
        #[case] billed: Meter,
        #[case] value: f64,
        #[values(ServiceTier::Standard, ServiceTier::Priority)] tier: ServiceTier,
    ) {
        let (prices, sources) = tier_prices;
        let usage = usage(requested.clone(), prompt, false);
        let when = When {
            tier,
            channel: Channel::Online,
            at: OffsetDateTime::from_unix_timestamp(at).unwrap(),
        };
        let result =
            calculate_with_policy(&prices, &usage, &when, SelectionPolicy::SdkTokens(&sources));
        assert_eq!(result, expected(billed, requested, value));
    }

    #[fixture]
    fn when() -> When {
        When {
            tier: ServiceTier::Priority,
            channel: Channel::Online,
            at: OffsetDateTime::UNIX_EPOCH,
        }
    }

    fn rate(meter: Meter, tier: ServiceTier, band: Band, value: f64) -> (Price, UnitPrice) {
        (
            Price::online(meter, PriceWhen { tier, band }),
            UnitPrice::new(value).unwrap(),
        )
    }

    fn sources(prices: &Prices) -> TokenPriceSources {
        TokenPriceSources {
            off_peak_cache_read_declared: false,
            flat: prices.iter().map(|(key, _)| key.clone()).collect(),
            tiers: vec![],
        }
    }

    #[rstest]
    #[case::below(199_999, false, (1.0, 2.0, 3.0, 4.0))]
    #[case::lower_boundary(200_000, false, (1.0, 2.0, 3.0, 4.0))]
    #[case::lower_inclusive(200_000, true, (5.0, 6.0, 3.0, 7.0))]
    #[case::lower_crossed(200_001, false, (5.0, 6.0, 3.0, 7.0))]
    #[case::upper_boundary(512_000, false, (5.0, 6.0, 3.0, 7.0))]
    #[case::upper_inclusive(512_000, true, (5.0, 0.0, 9.0, 7.0))]
    #[case::upper_crossed(512_001, false, (5.0, 0.0, 9.0, 7.0))]
    fn sdk_flat_bands_carry_each_meter_independently(
        when: When,
        #[case] prompt: u64,
        #[case] inclusive: bool,
        #[case] expected: (f64, f64, f64, f64),
    ) {
        let prices = Prices::new(
            [
                rate(INPUT, ServiceTier::Standard, Band::Base, 1.0),
                rate(OUTPUT, ServiceTier::Standard, Band::Base, 2.0),
                rate(READ, ServiceTier::Standard, Band::Base, 3.0),
                rate(WRITE, ServiceTier::Standard, Band::Base, 4.0),
                rate(
                    INPUT,
                    ServiceTier::Standard,
                    Band::over_prompt_tokens(200_000),
                    5.0,
                ),
                rate(
                    OUTPUT,
                    ServiceTier::Standard,
                    Band::over_prompt_tokens(200_000),
                    6.0,
                ),
                rate(
                    WRITE,
                    ServiceTier::Standard,
                    Band::over_prompt_tokens(200_000),
                    7.0,
                ),
                rate(
                    OUTPUT,
                    ServiceTier::Standard,
                    Band::over_prompt_tokens(512_000),
                    0.0,
                ),
                rate(
                    READ,
                    ServiceTier::Standard,
                    Band::over_prompt_tokens(512_000),
                    9.0,
                ),
            ]
            .into(),
            None,
        );
        let sources = TokenPriceSources::default();
        let usage = Usage {
            quantities: [
                (INPUT, prompt - 200),
                (OUTPUT, 3),
                (READ, 100),
                (WRITE, 100),
            ]
            .map(|(meter, count)| (meter, crate::Quantity::Count(count)))
            .into(),
            prompt_tokens: prompt,
            inclusive_threshold: inclusive,
            ..Default::default()
        };
        let evaluated = crate::calculate_with_rates(
            &prices,
            &usage,
            &when,
            SelectionPolicy::SdkTokens(&sources),
            &[INPUT, OUTPUT, READ, WRITE, HOUR],
            &[],
        )
        .unwrap();
        let Priced::Known(priced) = evaluated.pricing else {
            panic!("expected known pricing")
        };
        assert_eq!(
            priced.input().get(),
            (prompt - 200) as f64 * expected.0 + 100.0 * (expected.2 + expected.3)
        );
        assert_eq!(priced.output().get(), 3.0 * expected.1);
        assert_eq!(
            evaluated
                .rates
                .iter()
                .map(|rate| rate.resolved.as_ref().unwrap().1.get())
                .collect::<Vec<_>>(),
            [expected.0, expected.1, expected.2, expected.3, expected.3]
        );
        if prompt > 512_000 || (inclusive && prompt == 512_000) {
            let Priced::Known(generic) = calculate(&prices, &usage, &when) else {
                panic!("expected generic pricing")
            };
            assert_eq!(generic.input().get(), (prompt - 200) as f64 + 100.0 * 13.0);
        }
    }

    #[rstest]
    #[case::below(127_999, 127_999, (1.0, 2.0))]
    #[case::boundary(128_000, 128_000, (1.0, 2.0))]
    #[case::input_only(128_001, 10, (3.0, 2.0))]
    #[case::output_only(10, 128_001, (1.0, 4.0))]
    #[case::both(128_001, 128_001, (3.0, 4.0))]
    #[case::zero_input(0, 128_001, (1.0, 4.0))]
    #[case::zero_output(128_001, 0, (3.0, 2.0))]
    fn directional_threshold_uses_each_aggregate_and_exact_flat_band(
        when: When,
        #[case] input: u64,
        #[case] output: u64,
        #[case] expected: (f64, f64),
        #[values(false, true)] inclusive: bool,
    ) {
        let prices = Prices::new(
            [
                rate(INPUT, ServiceTier::Standard, Band::Base, 1.0),
                rate(OUTPUT, ServiceTier::Standard, Band::Base, 2.0),
                rate(
                    INPUT,
                    ServiceTier::Standard,
                    Band::over_prompt_tokens(128_000),
                    3.0,
                ),
                rate(
                    OUTPUT,
                    ServiceTier::Standard,
                    Band::over_prompt_tokens(128_000),
                    4.0,
                ),
                rate(
                    INPUT,
                    ServiceTier::Standard,
                    Band::over_prompt_tokens(50),
                    9.0,
                ),
                rate(
                    OUTPUT,
                    ServiceTier::Standard,
                    Band::over_prompt_tokens(50),
                    9.0,
                ),
                rate(INPUT, ServiceTier::Priority, Band::Base, 7.0),
                rate(OUTPUT, ServiceTier::Priority, Band::Base, 8.0),
            ]
            .into(),
            overlay(None, Some(9.0)),
        );
        let usage = Usage {
            quantities: [(INPUT, input), (OUTPUT, output)]
                .map(|(meter, count)| (meter, crate::Quantity::Count(count)))
                .into(),
            prompt_tokens: 200_000,
            inclusive_threshold: inclusive,
            ..Default::default()
        };
        let evaluation = crate::calculate_with_rates(
            &prices,
            &usage,
            &when,
            SelectionPolicy::DirectionalTokens {
                threshold: NonZeroU64::new(128_000).unwrap(),
            },
            &[INPUT, OUTPUT, READ, REASONING],
            &[],
        )
        .unwrap();
        let Priced::Known(priced) = evaluation.pricing else {
            panic!("expected aggregate pricing")
        };
        assert_eq!(priced.input().get(), input as f64 * expected.0);
        assert_eq!(priced.output().get(), output as f64 * expected.1);
        assert_eq!(
            evaluation.rates,
            vec![
                crate::SelectedRate {
                    source_id: None,
                    caller_default: false,
                    requested_meter: INPUT,
                    resolved: Some((INPUT, UnitPrice::new(expected.0).unwrap()))
                },
                crate::SelectedRate {
                    source_id: None,
                    caller_default: false,
                    requested_meter: OUTPUT,
                    resolved: Some((OUTPUT, UnitPrice::new(expected.1).unwrap()))
                },
                crate::SelectedRate {
                    source_id: None,
                    caller_default: false,
                    requested_meter: READ,
                    resolved: Some((INPUT, UnitPrice::new(expected.0).unwrap()))
                },
                crate::SelectedRate {
                    source_id: None,
                    caller_default: false,
                    requested_meter: REASONING,
                    resolved: Some((OUTPUT, UnitPrice::new(expected.1).unwrap()))
                },
            ]
        );
    }

    fn overlay(reasoning: Option<f64>, output: Option<f64>) -> Option<OffPeak> {
        OffPeak::new(
            Schedule {
                windows: vec![Window {
                    hours: vec!["00:00-01:00".parse().unwrap()],
                    weekdays: None,
                }],
                weekday_timezone: None,
            },
            OffPeakRates {
                reasoning: reasoning.map(|value| UnitPrice::new(value).unwrap()),
                output: output.map(|value| UnitPrice::new(value).unwrap()),
                ..Default::default()
            },
        )
    }

    fn usage(meter: Meter, prompt_tokens: u64, inclusive_threshold: bool) -> Usage {
        Usage {
            quantities: [(meter, 5)]
                .map(|(meter, count)| (meter, crate::Quantity::Count(count)))
                .into(),
            prompt_tokens,
            inclusive_threshold,
            ..Default::default()
        }
    }

    fn expected(meter: Meter, original: Meter, value: f64) -> Priced {
        let unit_price = UnitPrice::new(value).unwrap();
        Priced::Known(Breakdown {
            total: unit_price * 5,
            lines: vec![Line {
                fallback_from: (meter != original).then_some(original),
                source_id: None,
                caller_default: false,
                report_to: None,
                meter,
                quantity: 5.into(),
                unit_price,
                amount: unit_price * 5,
            }],
        })
    }

    #[rstest]
    #[case::explicit_tier_output(None, Some(4.0), Some(3.0), OUTPUT, 4.0)]
    #[case::explicit_tier_reasoning(Some(7.0), Some(4.0), Some(3.0), REASONING, 7.0)]
    #[case::free_tier_output(None, Some(0.0), Some(3.0), OUTPUT, 0.0)]
    #[case::free_tier_reasoning(Some(0.0), Some(4.0), Some(3.0), REASONING, 0.0)]
    #[case::standard_reasoning(None, None, Some(3.0), REASONING, 3.0)]
    #[case::free_standard_reasoning(None, None, Some(0.0), REASONING, 0.0)]
    #[case::text_fallback(None, None, None, OUTPUT, 2.0)]
    fn reasoning_uses_declared_service_rates(
        when: When,
        #[values(
            ServiceTier::Priority,
            ServiceTier::Flex,
            ServiceTier::Balanced,
            ServiceTier::Ultrafast
        )]
        tier: ServiceTier,
        #[case] tier_reasoning: Option<f64>,
        #[case] tier_output: Option<f64>,
        #[case] standard_reasoning: Option<f64>,
        #[case] meter: Meter,
        #[case] value: f64,
    ) {
        let prices = Prices::new(
            [
                (OUTPUT, ServiceTier::Standard, Some(2.0)),
                (OUTPUT, tier, tier_output),
                (REASONING, tier, tier_reasoning),
                (REASONING, ServiceTier::Standard, standard_reasoning),
            ]
            .into_iter()
            .filter_map(|(meter, tier, value)| Some(rate(meter, tier, Band::Base, value?)))
            .collect(),
            None,
        );
        let result = calculate_with_policy(
            &prices,
            &usage(REASONING, 10, false),
            &When { tier, ..when },
            SelectionPolicy::SdkTokens(&sources(&prices)),
        );
        assert_eq!(result, expected(meter, REASONING, value));
    }

    #[rstest]
    #[case::below(99, false, None, None, 4.0, OUTPUT)]
    #[case::at_exclusive(100, false, None, None, 4.0, OUTPUT)]
    #[case::at_inclusive(100, true, None, None, 8.0, OUTPUT)]
    #[case::above(101, false, None, None, 8.0, OUTPUT)]
    #[case::offpeak_output(101, false, None, Some(6.0), 6.0, OUTPUT)]
    #[case::offpeak_reasoning(101, false, Some(0.0), Some(6.0), 0.0, REASONING)]
    fn reasoning_output_fallback_uses_selected_threshold_and_offpeak(
        when: When,
        #[case] prompt: u64,
        #[case] inclusive: bool,
        #[case] offpeak_reasoning: Option<f64>,
        #[case] offpeak_output: Option<f64>,
        #[case] value: f64,
        #[case] meter: Meter,
    ) {
        let prices = Prices::new(
            [
                rate(OUTPUT, ServiceTier::Standard, Band::Base, 2.0),
                rate(REASONING, ServiceTier::Standard, Band::Base, 3.0),
                rate(OUTPUT, ServiceTier::Priority, Band::Base, 4.0),
                rate(
                    OUTPUT,
                    ServiceTier::Priority,
                    Band::over_prompt_tokens(100),
                    8.0,
                ),
            ]
            .into(),
            overlay(offpeak_reasoning, offpeak_output),
        );
        let result = calculate_with_policy(
            &prices,
            &usage(REASONING, prompt, inclusive),
            &when,
            SelectionPolicy::SdkTokens(&sources(&prices)),
        );
        assert_eq!(result, expected(meter, REASONING, value));
        assert_eq!(
            prices.rate(
                &REASONING,
                ServiceTier::Priority,
                Band::over_prompt_tokens(100)
            ),
            UnitPrice::new(3.0)
        );
    }

    #[rstest]
    #[case::zero_prompt(0, (true, true), (None, None), REASONING, 5.0)]
    #[case::first(10, (true, true), (None, None), OUTPUT, 3.0)]
    #[case::boundary(100, (true, true), (None, None), OUTPUT, 3.0)]
    #[case::second(101, (true, true), (None, None), OUTPUT, 6.0)]
    #[case::above_last(1001, (true, true), (None, None), OUTPUT, 6.0)]
    #[case::no_input(101, (false, true), (None, None), REASONING, 5.0)]
    #[case::no_tier_output(101, (true, false), (None, None), REASONING, 5.0)]
    #[case::tier_reasoning(101, (true, true), (Some(9.0), None), REASONING, 9.0)]
    #[case::free_tier_reasoning(101, (true, true), (Some(0.0), None), REASONING, 0.0)]
    #[case::offpeak_wins(101, (true, true), (Some(9.0), Some(0.0)), REASONING, 0.0)]
    fn reasoning_uses_eligible_tier_before_global_rates(
        when: When,
        #[case] prompt: u64,
        #[case] declared: (bool, bool),
        #[case] reasoning: (Option<f64>, Option<f64>),
        #[case] meter: Meter,
        #[case] value: f64,
        #[values(false, true)] inclusive: bool,
    ) {
        let (input_declared, output_declared) = declared;
        let (tier_reasoning, offpeak_reasoning) = reasoning;
        let prices = Prices::from_sources(
            [
                rate(REASONING, ServiceTier::Standard, Band::Base, 5.0),
                rate(OUTPUT, ServiceTier::Standard, Band::Base, 3.0),
                rate(
                    OUTPUT,
                    ServiceTier::Standard,
                    Band::over_prompt_tokens(100),
                    6.0,
                ),
            ]
            .into_iter()
            .chain(tier_reasoning.map(|value| {
                rate(
                    REASONING,
                    ServiceTier::Standard,
                    Band::over_prompt_tokens(100),
                    value,
                )
            }))
            .map(|(price, value)| {
                let source = match (price.meter.clone(), price.when.band) {
                    (REASONING, Band::Base) => PriceSource::Flat,
                    (_, Band::Base) => PriceSource::TokenTier(0),
                    (_, Band::OverPromptTokens(start)) => PriceSource::TokenTier(start.get()),
                };
                ((price, source), value)
            })
            .collect(),
            overlay(offpeak_reasoning, Some(1.0)),
        );
        let declared = [
            input_declared.then_some(Meter::TEXT_INPUT),
            output_declared.then_some(OUTPUT),
            tier_reasoning.map(|_| REASONING),
        ]
        .into_iter()
        .flatten()
        .collect();
        let sources = TokenPriceSources {
            flat: [Price::online(REASONING, PriceWhen::default())].into(),
            off_peak_cache_read_declared: false,
            tiers: vec![
                TokenTier {
                    range: [100, 1000],
                    meters: declared,
                },
                TokenTier {
                    range: [0, 100],
                    meters: [Meter::TEXT_INPUT, OUTPUT].into(),
                },
            ],
        };
        let result = calculate_with_policy(
            &prices,
            &usage(REASONING, prompt, inclusive),
            &when,
            SelectionPolicy::SdkTokens(&sources),
        );
        assert_eq!(result, expected(meter, REASONING, value));
    }

    #[rstest]
    #[case::standard(Some(3.0), true, false, 3.0)]
    #[case::free(Some(0.0), true, false, 0.0)]
    #[case::priority_only(None, true, true, 8.0)]
    #[case::no_modality(None, false, true, 8.0)]
    fn output_modalities_use_only_flat_standard_rates(
        when: When,
        #[values(TokenKind::Audio, TokenKind::Image, TokenKind::Video)] kind: TokenKind,
        #[case] standard: Option<f64>,
        #[case] priority: bool,
        #[case] fallback: bool,
        #[case] value: f64,
    ) {
        let meter = Meter::Output(OutputUnit::Token(kind));
        let prices = Prices::new(
            [
                rate(OUTPUT, ServiceTier::Priority, Band::Base, 4.0),
                rate(
                    OUTPUT,
                    ServiceTier::Priority,
                    Band::over_prompt_tokens(100),
                    8.0,
                ),
            ]
            .into_iter()
            .chain(
                standard.map(|value| rate(meter.clone(), ServiceTier::Standard, Band::Base, value)),
            )
            .chain(priority.then(|| rate(meter.clone(), ServiceTier::Priority, Band::Base, 7.0)))
            .collect(),
            None,
        );
        let result = calculate_with_policy(
            &prices,
            &usage(meter.clone(), 101, false),
            &when,
            SelectionPolicy::SdkTokens(&sources(&prices)),
        );
        assert_eq!(
            result,
            expected(if fallback { OUTPUT } else { meter.clone() }, meter, value)
        );
    }

    #[rstest]
    #[case::input(Meter::TEXT_INPUT, Channel::Online)]
    #[case::input_audio(Meter::Input(InputUnit::Token(TokenKind::Audio)), Channel::Online)]
    #[case::output_text(OUTPUT, Channel::Online)]
    #[case::image(Meter::Output(OutputUnit::Image(None)), Channel::Online)]
    #[case::batch_image(Meter::Output(OutputUnit::Image(None)), Channel::Batch)]
    #[case::decisions(REASONING, Channel::Decisions)]
    fn unrelated_meters_and_channels_use_the_generic_engine(
        when: When,
        #[case] meter: Meter,
        #[case] channel: Channel,
    ) {
        let prices = Prices::new(
            [
                rate(meter.clone(), ServiceTier::Standard, Band::Base, 3.0),
                rate(meter.clone(), ServiceTier::Priority, Band::Base, 7.0),
            ]
            .into(),
            None,
        );
        let usage = usage(meter, 101, false);
        let when = When { channel, ..when };
        assert_eq!(
            calculate_with_policy(
                &prices,
                &usage,
                &when,
                SelectionPolicy::SdkTokens(&sources(&prices))
            ),
            calculate(&prices, &usage, &when)
        );
    }

    #[rstest]
    #[case::text_present(Some(2.0))]
    #[case::text_zero(Some(0.0))]
    #[case::text_missing(None)]
    fn sdk_batch_nonimage_outputs_resolve_only_through_text(
        when: When,
        #[case] text: Option<f64>,
        #[values(
            REASONING,
            Meter::Output(OutputUnit::Token(TokenKind::Audio)),
            Meter::Output(OutputUnit::Token(TokenKind::Video))
        )]
        meter: Meter,
    ) {
        let prices = Prices::new(
            [
                Some(rate(meter.clone(), ServiceTier::Standard, Band::Base, 8.0)),
                text.map(|value| rate(OUTPUT, ServiceTier::Standard, Band::Base, value)),
            ]
            .into_iter()
            .flatten()
            .collect(),
            None,
        );
        let sources = sources(&prices);
        let when = When {
            channel: Channel::Batch,
            ..when
        };
        let usage = usage(meter.clone(), 100, false);
        let evaluation = crate::calculate_with_rates(
            &prices,
            &usage,
            &when,
            SelectionPolicy::SdkTokens(&sources),
            std::slice::from_ref(&meter),
            &[],
        )
        .unwrap();
        assert_eq!(
            evaluation.rates[0].resolved,
            text.map(|value| (OUTPUT, UnitPrice::new(value / 2.0).unwrap()))
        );
        match text {
            Some(value) => assert_eq!(
                evaluation.pricing,
                expected(OUTPUT, meter.clone(), value / 2.0)
            ),
            None => assert_eq!(
                evaluation.pricing,
                Priced::Unpriced {
                    missing: vec![meter.clone()],
                    partial: Breakdown {
                        lines: vec![],
                        total: crate::money::Usd::ZERO
                    },
                }
            ),
        }
        assert_eq!(
            calculate(&prices, &usage, &when),
            expected(meter.clone(), meter, 4.0)
        );
    }

    #[rstest]
    fn unpriced_preserves_missing_meters_and_partial_amounts(when: When) {
        let prices = Prices::new(
            [rate(
                Meter::TEXT_INPUT,
                ServiceTier::Standard,
                Band::Base,
                2.0,
            )]
            .into(),
            None,
        );
        let usage = Usage {
            quantities: [(Meter::TEXT_INPUT, 5), (REASONING, 5), (OUTPUT, 0)]
                .map(|(meter, count)| (meter, crate::Quantity::Count(count)))
                .into(),
            ..Default::default()
        };
        assert_eq!(
            calculate_with_policy(
                &prices,
                &usage,
                &when,
                SelectionPolicy::SdkTokens(&sources(&prices))
            ),
            Priced::Unpriced {
                missing: vec![REASONING],
                partial: match expected(Meter::TEXT_INPUT, Meter::TEXT_INPUT, 2.0) {
                    Priced::Known(breakdown) => breakdown,
                    _ => unreachable!(),
                },
            }
        );
    }
}
