use std::collections::BTreeMap;

use time::OffsetDateTime;

use crate::money::{UnitPrice, Usd};
use crate::price::{Band, Channel, InputUnit, Meter, OutputUnit, Prices, ServiceTier};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    pub quantities: BTreeMap<Meter, u64>,
    pub prompt_tokens: u64,
    pub inclusive_threshold: bool,
    pub search_results: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct When {
    pub tier: ServiceTier,
    pub channel: Channel,
    pub at: OffsetDateTime,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Priced {
    Known(Breakdown),
    Unpriced {
        missing: Vec<Meter>,
        partial: Breakdown,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Breakdown {
    pub lines: Vec<Line>,
    pub total: Usd,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Line {
    pub meter: Meter,
    pub quantity: u64,
    pub unit_price: UnitPrice,
    pub amount: Usd,
    pub fallback_from: Option<Meter>,
}

impl Breakdown {
    pub fn input(&self) -> Usd {
        self.lines
            .iter()
            .filter(|line| !matches!(line.meter, Meter::Output(_)))
            .map(|line| line.amount)
            .sum()
    }

    pub fn output(&self) -> Usd {
        self.lines
            .iter()
            .filter(|line| matches!(line.meter, Meter::Output(_)))
            .map(|line| line.amount)
            .sum()
    }
}

pub fn calculate(prices: &Prices, usage: &Usage, when: &When) -> Priced {
    let band = prices.channel_band(
        when.channel,
        when.tier,
        usage.prompt_tokens,
        usage.inclusive_threshold,
    );
    let mut lines = Vec::new();
    let mut missing = Vec::new();
    for (meter, &quantity) in &usage.quantities {
        if quantity == 0 {
            continue;
        }
        let selected = select_meter(prices, meter, quantity, usage, when);
        match resolve(prices, &selected, band, usage, when) {
            Some((priced_meter, unit_price)) => lines.push(Line {
                fallback_from: (priced_meter != selected).then(|| meter.clone()),
                meter: priced_meter,
                quantity,
                unit_price,
                amount: unit_price * quantity,
            }),
            None => missing.push(meter.clone()),
        }
    }
    let breakdown = Breakdown {
        total: lines.iter().map(|line| line.amount).sum(),
        lines,
    };
    if missing.is_empty() {
        Priced::Known(breakdown)
    } else {
        Priced::Unpriced {
            missing,
            partial: breakdown,
        }
    }
}

fn select_meter(
    prices: &Prices,
    meter: &Meter,
    quantity: u64,
    usage: &Usage,
    when: &When,
) -> Meter {
    if !matches!(
        meter,
        Meter::Input(
            InputUnit::Query { over_results: None } | InputUnit::VideoSecond { over_seconds: None }
        )
    ) {
        return meter.clone();
    }
    prices
        .iter()
        .filter(|(price, _)| {
            price.channel == when.channel
                && (price.when.tier == when.tier || price.when.tier == ServiceTier::Standard)
        })
        .filter_map(|(price, _)| match (meter, &price.meter) {
            (
                Meter::Input(InputUnit::Query { over_results: None }),
                Meter::Input(InputUnit::Query {
                    over_results: Some(floor),
                }),
            ) if usage.search_results > floor.get() => Some((floor, &price.meter)),
            (
                Meter::Input(InputUnit::VideoSecond { over_seconds: None }),
                Meter::Input(InputUnit::VideoSecond {
                    over_seconds: Some(floor),
                }),
            ) if quantity > floor.get() => Some((floor, &price.meter)),
            _ => None,
        })
        .max_by_key(|(floor, _)| *floor)
        .map_or_else(|| meter.clone(), |(_, selected)| selected.clone())
}

fn resolve(
    prices: &Prices,
    meter: &Meter,
    band: Band,
    usage: &Usage,
    when: &When,
) -> Option<(Meter, UnitPrice)> {
    if when.channel == Channel::Batch {
        return batch_rate(prices, meter, usage);
    }
    let overlay = (when.channel == Channel::Online)
        .then(|| prices.off_peak())
        .flatten()
        .filter(|off_peak| off_peak.schedule().contains(when.at))
        .and_then(|off_peak| off_peak.rates().get(meter));
    overlay
        .or_else(|| prices.direct_rate(meter, when.channel, when.tier, band))
        .map(|rate| (meter.clone(), rate))
        .or_else(|| resolve(prices, &meter.fallback()?, band, usage, when))
}

fn batch_direct_rate(prices: &Prices, meter: &Meter, usage: &Usage) -> Option<UnitPrice> {
    let band = prices
        .iter()
        .filter(|(price, _)| price.channel == Channel::Batch && price.meter == *meter)
        .filter_map(|(price, _)| match price.when.band {
            Band::OverPromptTokens(floor)
                if usage.prompt_tokens > floor.get()
                    || (usage.inclusive_threshold && usage.prompt_tokens == floor.get()) =>
            {
                Some(price.when.band)
            }
            _ => None,
        })
        .max()
        .unwrap_or_default();
    prices.direct_rate(meter, Channel::Batch, ServiceTier::Standard, band)
}

fn batch_rate(prices: &Prices, meter: &Meter, usage: &Usage) -> Option<(Meter, UnitPrice)> {
    batch_direct_rate(prices, meter, usage)
        .map(|rate| (meter.clone(), rate))
        .or_else(|| {
            let online =
                prices.direct_rate(meter, Channel::Online, ServiceTier::Standard, Band::Base);
            let batch_input = batch_direct_rate(prices, &Meter::TEXT_INPUT, usage).is_some();
            if (matches!(meter, Meter::Output(_)) || !batch_input)
                && matches!(
                    meter,
                    Meter::Input(InputUnit::Token(_))
                        | Meter::Output(OutputUnit::Token(_) | OutputUnit::ReasoningToken)
                        | Meter::CacheRead { .. }
                        | Meter::CacheWrite { .. }
                )
                && let Some(rate) = online
            {
                return Some((meter.clone(), rate.half()));
            }
            batch_rate(prices, &meter.fallback()?, usage)
        })
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

    use rstest::{fixture, rstest};
    use time::format_description::well_known::Rfc3339;

    use super::*;
    use crate::price::{
        CacheReadMode, CacheTtl, Feature, ImageSize, OffPeak, OffPeakRates, Price,
        SearchContextSize, TokenKind, When as PriceWhen,
    };
    use crate::schedule::{Schedule, Window};

    const INPUT: Meter = Meter::TEXT_INPUT;
    const OUTPUT: Meter = Meter::Output(OutputUnit::Token(TokenKind::Text));
    const READ: Meter = Meter::CacheRead {
        kind: TokenKind::Text,
        mode: CacheReadMode::Implicit,
    };
    const WRITE: Meter = Meter::CacheWrite {
        kind: TokenKind::Text,
        ttl: CacheTtl::Default,
    };
    const HOUR: Meter = Meter::CacheWrite {
        kind: TokenKind::Text,
        ttl: CacheTtl::OneHour,
    };

    fn prices(rates: impl IntoIterator<Item = (Meter, f64)>) -> Prices {
        Prices::new(
            rates
                .into_iter()
                .map(|(meter, rate)| {
                    (
                        Price::online(meter, PriceWhen::default()),
                        UnitPrice::new(rate).unwrap(),
                    )
                })
                .collect(),
            None,
        )
    }

    fn known(result: Priced) -> Breakdown {
        match result {
            Priced::Known(breakdown) => breakdown,
            Priced::Unpriced { missing, .. } => panic!("unexpected missing prices: {missing:?}"),
        }
    }

    #[fixture]
    fn when() -> When {
        When {
            tier: ServiceTier::Standard,
            channel: Channel::Online,
            at: OffsetDateTime::UNIX_EPOCH,
        }
    }

    #[rstest]
    #[case::audio_input(Meter::Input(InputUnit::Token(TokenKind::Audio)), INPUT, 500)]
    #[case::image_input(Meter::Input(InputUnit::Token(TokenKind::Image)), INPUT, 500)]
    #[case::video_input(Meter::Input(InputUnit::Token(TokenKind::Video)), INPUT, 500)]
    #[case::cache_read(READ, INPUT, 800)]
    #[case::cache_write(WRITE, INPUT, 800)]
    #[case::cache_write_hour(HOUR, INPUT, 800)]
    #[case::audio_output(Meter::Output(OutputUnit::Token(TokenKind::Audio)), OUTPUT, 500)]
    #[case::image_output(Meter::Output(OutputUnit::Token(TokenKind::Image)), OUTPUT, 500)]
    #[case::reasoning(Meter::Output(OutputUnit::ReasoningToken), OUTPUT, 500)]
    fn missing_submeters_bill_as_text_and_mark_the_fallback(
        when: When,
        #[case] meter: Meter,
        #[case] base: Meter,
        #[case] quantity: u64,
    ) {
        let usage = Usage {
            quantities: [
                (INPUT, if base == INPUT { 1000 - quantity } else { 1000 }),
                (
                    OUTPUT,
                    if base == OUTPUT {
                        1000 - quantity
                    } else {
                        1000
                    },
                ),
                (meter.clone(), quantity),
            ]
            .into(),
            prompt_tokens: 1000,
            ..Usage::default()
        };
        let result = known(calculate(
            &prices([(INPUT, 0.000001), (OUTPUT, 0.000002)]),
            &usage,
            &when,
        ));
        assert!((result.input().get() - 0.001).abs() < 1e-12);
        assert!((result.output().get() - 0.002).abs() < 1e-12);
        assert!((result.total.get() - 0.003).abs() < 1e-12);
        let fallback = result
            .lines
            .iter()
            .find(|line| line.fallback_from == Some(meter.clone()))
            .unwrap();
        assert_eq!(fallback.meter, base);
        assert_eq!(fallback.quantity, quantity);
    }

    #[rstest]
    #[case::input(INPUT)]
    #[case::read(READ)]
    #[case::write(WRITE)]
    #[case::hour(HOUR)]
    #[case::audio(Meter::Input(InputUnit::Token(TokenKind::Audio)))]
    #[case::image(Meter::Output(OutputUnit::Token(TokenKind::Image)))]
    fn an_explicit_zero_is_a_priced_line_not_a_fallback(when: When, #[case] meter: Meter) {
        let row = prices([(INPUT, 1.0), (OUTPUT, 2.0), (meter.clone(), 0.0)]);
        let usage = Usage {
            quantities: [(meter.clone(), 100)].into(),
            ..Usage::default()
        };
        let result = known(calculate(&row, &usage, &when));
        assert_eq!(result.total, Usd::ZERO);
        assert_eq!(
            result.lines,
            [Line {
                meter,
                quantity: 100,
                unit_price: UnitPrice::new(0.0).unwrap(),
                amount: Usd::ZERO,
                fallback_from: None,
            }]
        );
    }

    #[rstest]
    fn every_missing_meter_is_reported_and_priced_lines_survive(when: When) {
        let audio = Meter::Input(InputUnit::Token(TokenKind::Audio));
        let usage = Usage {
            quantities: [(INPUT, 3), (OUTPUT, 4), (audio.clone(), 2)].into(),
            ..Usage::default()
        };
        let Priced::Unpriced { missing, partial } =
            calculate(&prices([(OUTPUT, 2.0)]), &usage, &when)
        else {
            panic!("unpriced input must not be free");
        };
        assert_eq!(missing, [INPUT, audio]);
        assert_eq!(partial.total.get(), 8.0);
        assert_eq!(partial.lines.len(), 1);
        assert_eq!(partial.lines[0].meter, OUTPUT);
        assert_eq!(partial.lines[0].quantity, 4);
        assert_eq!(partial.output().get(), 8.0);
    }

    #[rstest]
    fn an_unpriced_row_cannot_fall_back_to_zero(when: When) {
        let usage = Usage {
            quantities: [(INPUT, 1000), (OUTPUT, 1000)].into(),
            ..Usage::default()
        };
        assert_eq!(
            calculate(&Prices::default(), &usage, &when),
            Priced::Unpriced {
                missing: vec![INPUT, OUTPUT],
                partial: Breakdown {
                    lines: vec![],
                    total: Usd::ZERO
                },
            }
        );
    }

    #[rstest]
    fn unused_meters_need_no_price(when: When) {
        let usage = Usage {
            quantities: [(INPUT, 0), (OUTPUT, 0)].into(),
            ..Usage::default()
        };
        assert_eq!(
            calculate(&Prices::default(), &usage, &when),
            Priced::Known(Breakdown {
                lines: vec![],
                total: Usd::ZERO
            })
        );
    }

    #[rstest]
    fn one_hour_writes_fall_back_to_the_default_write_price(when: When) {
        let usage = Usage {
            quantities: [(WRITE, 4), (HOUR, 6)].into(),
            ..Usage::default()
        };
        let result = known(calculate(
            &prices([(INPUT, 1.0), (WRITE, 3.0)]),
            &usage,
            &when,
        ));
        assert_eq!(result.total.get(), 30.0);
        assert_eq!(result.lines[1].meter, WRITE);
        assert_eq!(result.lines[1].fallback_from, Some(HOUR));
    }

    #[rstest]
    #[case::tier_band(ServiceTier::Priority, Band::over_prompt_tokens(100), 2.0)]
    #[case::standard_band(ServiceTier::Standard, Band::over_prompt_tokens(100), 2.0)]
    #[case::tier_base(ServiceTier::Priority, Band::Base, 2.0)]
    #[case::standard_base(ServiceTier::Standard, Band::Base, 1.0)]
    fn calculator_uses_every_lookup_fallback_step(
        when: When,
        #[case] tier: ServiceTier,
        #[case] band: Band,
        #[case] expected: f64,
    ) {
        let row = Prices::new(
            [
                (
                    Price::online(INPUT, PriceWhen { tier, band }),
                    UnitPrice::new(2.0).unwrap(),
                ),
                (
                    Price::online(INPUT, PriceWhen::default()),
                    UnitPrice::new(1.0).unwrap(),
                ),
                (
                    Price::online(
                        OUTPUT,
                        PriceWhen {
                            tier: ServiceTier::Standard,
                            band: Band::over_prompt_tokens(100),
                        },
                    ),
                    UnitPrice::new(3.0).unwrap(),
                ),
            ]
            .into(),
            None,
        );
        let usage = Usage {
            quantities: [(INPUT, 101)].into(),
            prompt_tokens: 101,
            ..Usage::default()
        };
        let result = known(calculate(
            &row,
            &usage,
            &When {
                tier: ServiceTier::Priority,
                ..when
            },
        ));
        assert_eq!(result.total.get(), 101.0 * expected);
        assert_eq!(result.lines[0].fallback_from, None);
    }

    #[rstest]
    #[case::input_characters(Meter::Input(InputUnit::Character))]
    #[case::output_characters(Meter::Output(OutputUnit::Character))]
    #[case::audio_seconds(Meter::Input(InputUnit::AudioSecond))]
    #[case::wall_clock(Meter::WallClockSecond)]
    #[case::image_size(Meter::Output(OutputUnit::Image(ImageSize::new("1K"))))]
    #[case::search(Meter::WebSearch(SearchContextSize::High))]
    #[case::ocr(Meter::Feature(Feature::OcrPage))]
    fn non_token_meters_use_their_own_units_and_never_fall_back_to_text(
        when: When,
        #[case] meter: Meter,
    ) {
        let usage = Usage {
            quantities: [(meter.clone(), 7)].into(),
            ..Usage::default()
        };
        let result = known(calculate(&prices([(meter.clone(), 0.25)]), &usage, &when));
        assert_eq!(result.total.get(), 1.75);
        assert_eq!(result.lines[0].meter, meter);
        assert_eq!(result.lines[0].fallback_from, None);
        assert!(
            matches!(calculate(&prices([(INPUT, 1.0), (OUTPUT, 2.0)]), &usage, &when), Priced::Unpriced { missing, .. } if missing == [meter])
        );
    }

    #[rstest]
    #[case::before(9, 1.0)]
    #[case::at(10, 1.0)]
    #[case::after(11, 3.0)]
    fn query_count_and_result_band_are_independent(
        when: When,
        #[case] results: u64,
        #[case] rate: f64,
    ) {
        let query = Meter::Input(InputUnit::Query { over_results: None });
        let upper = Meter::Input(InputUnit::Query {
            over_results: NonZeroU64::new(10),
        });
        let usage = Usage {
            quantities: [(query.clone(), 2)].into(),
            search_results: results,
            ..Usage::default()
        };
        let result = known(calculate(
            &prices([(query, 1.0), (upper, 3.0)]),
            &usage,
            &when,
        ));
        assert_eq!(result.total.get(), 2.0 * rate);
        assert_eq!(result.lines[0].fallback_from, None);
    }

    #[rstest]
    #[case::at(10, 10.0)]
    #[case::after(11, 33.0)]
    fn video_duration_selects_the_interval(
        when: When,
        #[case] seconds: u64,
        #[case] expected: f64,
    ) {
        let base = Meter::Input(InputUnit::VideoSecond { over_seconds: None });
        let upper = Meter::Input(InputUnit::VideoSecond {
            over_seconds: NonZeroU64::new(10),
        });
        let usage = Usage {
            quantities: [(base.clone(), seconds)].into(),
            ..Usage::default()
        };
        assert_eq!(
            known(calculate(
                &prices([(base, 1.0), (upper, 3.0)]),
                &usage,
                &when
            ))
            .total
            .get(),
            expected
        );
    }

    #[rstest]
    #[case::inside("2026-01-05T23:00:00Z", 20.0)]
    #[case::end_excluded("2026-01-05T01:00:00Z", 38.0)]
    #[case::start_included("2026-01-05T22:00:00Z", 20.0)]
    fn off_peak_overlays_only_declared_rates_and_propagates_fallbacks(
        when: When,
        #[case] at: &str,
        #[case] expected: f64,
    ) {
        let base = prices([(INPUT, 2.0), (OUTPUT, 4.0), (READ, 3.0)]);
        let row = Prices::new(
            base.iter().map(|(key, rate)| (key.clone(), rate)).collect(),
            OffPeak::new(
                Schedule {
                    windows: vec![Window {
                        hours: vec!["22:00-01:00".parse().unwrap()],
                        weekdays: None,
                    }],
                    weekday_timezone: None,
                },
                OffPeakRates {
                    input: UnitPrice::new(0.0),
                    ..OffPeakRates::default()
                },
            ),
        );
        let usage = Usage {
            quantities: [(INPUT, 2), (READ, 4), (HOUR, 7), (OUTPUT, 2)].into(),
            ..Usage::default()
        };
        let result = known(calculate(
            &row,
            &usage,
            &When {
                at: OffsetDateTime::parse(at, &Rfc3339).unwrap(),
                ..when
            },
        ));
        assert_eq!(result.total.get(), expected);
        assert_eq!(
            result
                .lines
                .iter()
                .find(|line| line.fallback_from == Some(HOUR))
                .unwrap()
                .meter,
            INPUT
        );
    }

    #[rstest]
    #[case::explicit(0.5, 0.0, 2.0)]
    #[case::free(0.0, 0.0, 0.0)]
    fn batch_rates_preserve_explicit_zero(
        when: When,
        #[case] input: f64,
        #[case] output: f64,
        #[case] expected: f64,
    ) {
        let row = Prices::new(
            [
                (
                    Price {
                        meter: INPUT,
                        channel: Channel::Batch,
                        when: PriceWhen::default(),
                    },
                    UnitPrice::new(input).unwrap(),
                ),
                (
                    Price {
                        meter: OUTPUT,
                        channel: Channel::Batch,
                        when: PriceWhen::default(),
                    },
                    UnitPrice::new(output).unwrap(),
                ),
                (
                    Price::online(INPUT, PriceWhen::default()),
                    UnitPrice::new(20.0).unwrap(),
                ),
            ]
            .into(),
            None,
        );
        let usage = Usage {
            quantities: [(INPUT, 4), (OUTPUT, 2)].into(),
            ..Usage::default()
        };
        assert_eq!(
            known(calculate(
                &row,
                &usage,
                &When {
                    channel: Channel::Batch,
                    ..when
                }
            ))
            .total
            .get(),
            expected
        );
    }

    #[rstest]
    fn absent_batch_rates_use_half_online_base_rates(when: When) {
        let usage = Usage {
            quantities: [(INPUT, 4), (OUTPUT, 2), (READ, 6)].into(),
            ..Usage::default()
        };
        let result = known(calculate(
            &prices([(INPUT, 2.0), (OUTPUT, 4.0), (READ, 1.0)]),
            &usage,
            &When {
                channel: Channel::Batch,
                ..when
            },
        ));
        assert_eq!(result.total.get(), 11.0);
        assert_eq!(result.input().get(), 7.0);
        assert_eq!(result.output().get(), 4.0);
    }

    #[rstest]
    #[case::exclusive(100, false, 10.0)]
    #[case::inclusive(100, true, 30.0)]
    #[case::above(101, false, 30.0)]
    fn batch_bands_are_per_meter_and_do_not_borrow_online_bands(
        when: When,
        #[case] prompt_tokens: u64,
        #[case] inclusive_threshold: bool,
        #[case] expected: f64,
    ) {
        let row = Prices::new(
            [
                (INPUT, Channel::Batch, Band::Base, 2.0),
                (INPUT, Channel::Batch, Band::over_prompt_tokens(100), 6.0),
                (OUTPUT, Channel::Batch, Band::Base, 3.0),
                (OUTPUT, Channel::Batch, Band::over_prompt_tokens(50), 5.0),
                (INPUT, Channel::Online, Band::over_prompt_tokens(75), 100.0),
                (READ, Channel::Online, Band::Base, 50.0),
            ]
            .into_iter()
            .map(|(meter, channel, band, rate)| {
                (
                    Price {
                        meter,
                        channel,
                        when: PriceWhen {
                            band,
                            ..PriceWhen::default()
                        },
                    },
                    UnitPrice::new(rate).unwrap(),
                )
            })
            .collect(),
            None,
        );
        let usage = Usage {
            quantities: [(INPUT, 2), (READ, 3), (OUTPUT, 4)].into(),
            prompt_tokens,
            inclusive_threshold,
            ..Usage::default()
        };
        let result = known(calculate(
            &row,
            &usage,
            &When {
                channel: Channel::Batch,
                ..when
            },
        ));
        assert_eq!(result.input().get(), expected);
        assert_eq!(result.output().get(), 20.0);
        assert_eq!(result.lines.last().unwrap().fallback_from, Some(READ));
    }

    #[rstest]
    fn dedicated_modality_and_long_ttl_rates_win(when: When) {
        let audio = Meter::Input(InputUnit::Token(TokenKind::Audio));
        let usage = Usage {
            quantities: [(INPUT, 10), (audio.clone(), 5), (WRITE, 3), (HOUR, 2)].into(),
            ..Usage::default()
        };
        let result = known(calculate(
            &prices([(INPUT, 1.0), (audio, 2.0), (WRITE, 4.0), (HOUR, 6.0)]),
            &usage,
            &when,
        ));
        assert_eq!(result.total.get(), 44.0);
        assert_eq!(
            result
                .lines
                .iter()
                .filter(|line| line.fallback_from.is_some())
                .count(),
            0
        );
    }

    #[rstest]
    fn decisions_prices_are_isolated_from_online_and_batch(when: When) {
        let usage = Usage {
            quantities: [(INPUT, 4)].into(),
            ..Usage::default()
        };
        assert!(matches!(
            calculate(
                &prices([(INPUT, 2.0)]),
                &usage,
                &When {
                    channel: Channel::Decisions,
                    ..when
                }
            ),
            Priced::Unpriced { .. }
        ));
        let row = Prices::new(
            [(
                Price {
                    meter: INPUT,
                    channel: Channel::Decisions,
                    when: PriceWhen::default(),
                },
                UnitPrice::new(3.0).unwrap(),
            )]
            .into(),
            None,
        );
        assert_eq!(
            known(calculate(
                &row,
                &usage,
                &When {
                    channel: Channel::Decisions,
                    ..when
                }
            ))
            .total
            .get(),
            12.0
        );
    }

    #[rstest]
    fn existing_python_reference_still_agrees(when: When) {
        for line in include_str!("../tests/python_reference.tsv")
            .lines()
            .filter(|line| !line.starts_with('#'))
        {
            let fields: Vec<_> = line.split('\t').collect();
            let count = |i: usize| fields[i].parse::<u64>().unwrap();
            let optional = |i: usize| fields[i].parse::<f64>().ok().and_then(UnitPrice::new);
            let band = fields[9].parse::<u64>().ok().map(Band::over_prompt_tokens);
            let row = Prices::new(
                [
                    (INPUT, Band::Base, optional(5)),
                    (OUTPUT, Band::Base, optional(6)),
                    (READ, Band::Base, optional(7)),
                    (WRITE, Band::Base, optional(8)),
                    (INPUT, band.unwrap_or_default(), optional(10)),
                    (OUTPUT, band.unwrap_or_default(), optional(11)),
                ]
                .into_iter()
                .filter_map(|(meter, band, rate)| {
                    Some((
                        Price::online(
                            meter,
                            PriceWhen {
                                band,
                                ..PriceWhen::default()
                            },
                        ),
                        rate?,
                    ))
                })
                .collect(),
                None,
            );
            let usage = Usage {
                quantities: [
                    (INPUT, count(1) - count(3) - count(4)),
                    (OUTPUT, count(2)),
                    (READ, count(3)),
                    (WRITE, count(4)),
                ]
                .into(),
                prompt_tokens: count(1),
                ..Usage::default()
            };
            let result = known(calculate(&row, &usage, &when));
            assert_eq!(
                result.input().get(),
                fields[12].parse::<f64>().unwrap(),
                "{}",
                fields[0]
            );
            assert_eq!(
                result.output().get(),
                fields[13].parse::<f64>().unwrap(),
                "{}",
                fields[0]
            );
        }
    }
}
