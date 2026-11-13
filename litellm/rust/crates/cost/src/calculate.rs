use std::time::Duration;

use crate::call::CallContext;
use crate::money::{Multiplier, Usd};
use crate::price::{Adjustments, InputUnit, Meter, OutputUnit, Prices, TokenKind};
use crate::priced::{Line, Outcome, Priced, Quantity, Side};
use crate::quote::Quote;
use crate::usage::{DurationMeter, Usage};

/// Prices the token, character, image, second and query meters of one online call. Each side is
/// summed in the order main sums it, so the two totals equal main's to the last bit
pub fn calculate(
    prices: &Prices,
    adjustments: &Adjustments,
    usage: &Usage,
    call: &CallContext,
) -> Outcome {
    let quote = prices.quote(call, usage.prompt_tokens);
    let uplifts: Vec<Multiplier> = adjustments.for_call(call).collect();
    let prompt = Billed::new(
        &quote,
        Side::Prompt,
        prompt_charges(&quote, usage),
        &uplifts,
    );
    let completion = Billed::new(
        &quote,
        Side::Completion,
        completion_charges(usage),
        &uplifts,
    );
    let lines: Vec<Line> = prompt.lines.into_iter().chain(completion.lines).collect();
    let missing: Vec<Meter> = prompt
        .missing
        .into_iter()
        .chain(completion.missing)
        .collect();
    if lines.is_empty() && !missing.is_empty() {
        return Outcome::Unpriced { missing };
    }
    Outcome::Priced(Priced {
        lines,
        missing,
        prompt: prompt.total,
        completion: completion.total,
    })
}

type Charge = (Meter, Quantity);

/// The prompt-side charges, grouped the way main adds them: the cache reads and the cache writes
/// are each summed on their own before joining the total
fn prompt_charges(quote: &Quote<'_>, usage: &Usage) -> Vec<Vec<Charge>> {
    let input = |kind: TokenKind| usage.tokens.input.get(&kind).copied().unwrap_or(0);
    let lasted = |meter: DurationMeter| usage.durations.get(&meter).copied().unwrap_or_default();
    let unit = |unit: InputUnit| Meter::Input(unit);
    let tokens_unless = |kind: TokenKind, billed_by: InputUnit, used: bool| {
        let replaced = used && quote.rate(&unit(billed_by)).is_some();
        let tokens = if replaced { 0 } else { input(kind) };
        vec![(unit(InputUnit::Token(kind)), Quantity::Count(tokens))]
    };
    let video = lasted(DurationMeter::InputVideo);
    let audio = lasted(DurationMeter::InputAudio);
    let video_second = InputUnit::VideoSecond { over_seconds: None };
    vec![
        tokens_unless(
            TokenKind::Text,
            InputUnit::Character,
            usage.characters.prompt > 0,
        ),
        usage
            .tokens
            .cache_read
            .iter()
            .map(|(&(kind, mode), &tokens)| {
                (Meter::CacheRead { kind, mode }, Quantity::Count(tokens))
            })
            .collect(),
        tokens_unless(TokenKind::Audio, InputUnit::AudioSecond, !audio.is_zero()),
        tokens_unless(TokenKind::Image, InputUnit::Image, usage.images.input > 0),
        tokens_unless(TokenKind::Video, video_second, !video.is_zero()),
        usage
            .tokens
            .cache_write
            .iter()
            .map(|(&(kind, ttl), &tokens)| {
                (Meter::CacheWrite { kind, ttl }, Quantity::Count(tokens))
            })
            .collect(),
        vec![(
            unit(InputUnit::Character),
            Quantity::Count(usage.characters.prompt),
        )],
        vec![(unit(InputUnit::Image), Quantity::Count(usage.images.input))],
        vec![(unit(video_second), Quantity::Duration(video))],
        vec![(unit(InputUnit::AudioSecond), Quantity::Duration(audio))],
        vec![(
            unit(InputUnit::Query { over_results: None }),
            Quantity::Count(usage.search.queries),
        )],
    ]
}

fn completion_charges(usage: &Usage) -> Vec<Vec<Charge>> {
    let output = |kind: TokenKind| {
        let tokens = usage.tokens.output.get(&kind).copied().unwrap_or(0);
        vec![(
            Meter::Output(OutputUnit::Token(kind)),
            Quantity::Count(tokens),
        )]
    };
    vec![
        output(TokenKind::Text),
        output(TokenKind::Audio),
        vec![(Meter::REASONING, Quantity::Count(usage.tokens.reasoning))],
        output(TokenKind::Image),
        output(TokenKind::Video),
    ]
}

struct Billed {
    lines: Vec<Line>,
    missing: Vec<Meter>,
    total: Usd,
}

impl Billed {
    fn new(
        quote: &Quote<'_>,
        side: Side,
        groups: Vec<Vec<Charge>>,
        uplifts: &[Multiplier],
    ) -> Self {
        let uplifted = |amount: Usd| {
            uplifts
                .iter()
                .fold(amount, |amount, uplift| amount * *uplift)
        };
        let priced: Vec<Vec<Result<Line, Meter>>> = groups
            .into_iter()
            .map(|group| {
                group
                    .into_iter()
                    .filter(|(_, quantity)| !is_zero(*quantity))
                    .map(|(meter, quantity)| match quote.rate(&meter) {
                        Some(rate) => Ok(Line {
                            meter,
                            quantity,
                            unit_price: rate.unit_price,
                            amount: rate.unit_price * quantity,
                            fallback_from: rate.fallback_from,
                            side,
                        }),
                        None => Err(meter),
                    })
                    .collect()
            })
            .collect();
        let total = priced
            .iter()
            .map(|group| group.iter().flatten().map(|line| line.amount).sum::<Usd>())
            .sum();
        let (lines, missing): (Vec<_>, Vec<_>) =
            priced.into_iter().flatten().partition(Result::is_ok);
        Self {
            lines: lines
                .into_iter()
                .flatten()
                .map(|line| Line {
                    amount: uplifted(line.amount),
                    ..line
                })
                .collect(),
            missing: missing.into_iter().filter_map(Result::err).collect(),
            total: uplifted(total),
        }
    }
}

fn is_zero(quantity: Quantity) -> bool {
    match quantity {
        Quantity::Count(units) => units == 0,
        Quantity::Duration(duration) => duration == Duration::ZERO,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use rstest::rstest;

    use super::*;
    use crate::money::UnitPrice;
    use crate::price::{
        Billing, CacheReadMode, CacheTtl, OffPeakRates, Residency, ServiceTier::Standard, Uplift,
    };
    use crate::testing::{
        AUDIO_INPUT, BASE, CACHE_WRITE_HOUR, call, off_peak_all_day, over, prices, prices_with,
    };
    use crate::usage::{Characters, Images, Search, Tokens};

    const INPUT: Meter = Meter::TEXT_INPUT;
    const OUTPUT: Meter = Meter::TEXT_OUTPUT;
    const CHARACTER: Meter = Meter::Input(InputUnit::Character);

    fn text(prompt_tokens: u64, input: u64, output: u64) -> Usage {
        Usage {
            prompt_tokens,
            tokens: Tokens {
                input: BTreeMap::from([(TokenKind::Text, input)]),
                output: BTreeMap::from([(TokenKind::Text, output)]),
                ..Tokens::default()
            },
            ..Usage::default()
        }
    }

    fn priced(prices: &Prices, usage: &Usage) -> Priced {
        priced_with(prices, &Adjustments::default(), usage, &call())
    }

    fn priced_with(
        prices: &Prices,
        adjustments: &Adjustments,
        usage: &Usage,
        call: &CallContext,
    ) -> Priced {
        match calculate(prices, adjustments, usage, call) {
            Outcome::Priced(priced) => priced,
            Outcome::Unpriced { missing } => panic!("unpriced, missing {missing:?}"),
        }
    }

    fn sides(priced: &Priced) -> (f64, f64) {
        (priced.prompt.get(), priced.completion.get())
    }

    fn billed(priced: &Priced) -> Vec<(Meter, f64)> {
        priced
            .lines
            .iter()
            .map(|line| (line.meter.clone(), line.amount.get()))
            .collect()
    }

    #[rstest]
    fn each_side_is_its_tokens_times_its_rate() {
        let prices = prices(&[(INPUT, Standard, BASE, 0.25), (OUTPUT, Standard, BASE, 2.0)]);
        let priced = priced(&prices, &text(100, 100, 10));

        assert_eq!(sides(&priced), (25.0, 20.0));
        assert_eq!(priced.total().get(), 45.0);
        assert_eq!(billed(&priced), [(INPUT, 25.0), (OUTPUT, 20.0)]);
        assert!(priced.missing.is_empty());
    }

    #[rstest]
    fn the_prompt_size_picks_the_band_for_every_token_of_the_call() {
        let prices = prices(&[
            (INPUT, Standard, BASE, 1.0),
            (INPUT, Standard, over(200), 2.0),
            (OUTPUT, Standard, BASE, 4.0),
            (OUTPUT, Standard, over(200), 8.0),
        ]);
        assert_eq!(sides(&priced(&prices, &text(200, 200, 10))), (200.0, 40.0));
        assert_eq!(sides(&priced(&prices, &text(201, 201, 10))), (402.0, 80.0));
    }

    #[rstest]
    fn cache_reads_and_writes_bill_by_modality_mode_and_lifetime() {
        let cached_audio = Meter::CacheRead {
            kind: TokenKind::Audio,
            mode: CacheReadMode::Implicit,
        };
        let prices = prices(&[
            (INPUT, Standard, BASE, 1.0),
            (Meter::CACHE_READ, Standard, BASE, 0.125),
            (cached_audio.clone(), Standard, BASE, 0.5),
            (Meter::CACHE_WRITE, Standard, BASE, 1.25),
            (CACHE_WRITE_HOUR, Standard, BASE, 2.0),
        ]);
        let usage = Usage {
            prompt_tokens: 1_000,
            tokens: Tokens {
                input: BTreeMap::from([(TokenKind::Text, 100)]),
                cache_read: BTreeMap::from([
                    ((TokenKind::Text, CacheReadMode::Implicit), 800),
                    ((TokenKind::Audio, CacheReadMode::Implicit), 16),
                    ((TokenKind::Image, CacheReadMode::Implicit), 8),
                ]),
                cache_write: BTreeMap::from([
                    ((TokenKind::Text, CacheTtl::Default), 40),
                    ((TokenKind::Text, CacheTtl::OneHour), 20),
                    ((TokenKind::Text, CacheTtl::TwelveHours), 4),
                ]),
                ..Tokens::default()
            },
            ..Usage::default()
        };
        let priced = priced(&prices, &usage);

        assert_eq!(
            billed(&priced),
            [
                (INPUT, 100.0),
                (Meter::CACHE_READ, 100.0),
                (cached_audio, 8.0),
                (
                    Meter::CacheRead {
                        kind: TokenKind::Image,
                        mode: CacheReadMode::Implicit
                    },
                    1.0
                ),
                (Meter::CACHE_WRITE, 50.0),
                (CACHE_WRITE_HOUR, 40.0),
                (
                    Meter::CacheWrite {
                        kind: TokenKind::Text,
                        ttl: CacheTtl::TwelveHours
                    },
                    5.0
                ),
            ]
        );
        assert_eq!(sides(&priced), (304.0, 0.0));
    }

    #[rstest]
    fn an_explicit_cache_read_bills_at_the_explicit_price() {
        let prices = prices(&[
            (INPUT, Standard, BASE, 1.0),
            (Meter::CACHE_READ, Standard, BASE, 0.5),
            (Meter::EXPLICIT_CACHE_READ, Standard, BASE, 0.25),
        ]);
        let read = |mode| Usage {
            prompt_tokens: 100,
            tokens: Tokens {
                cache_read: BTreeMap::from([((TokenKind::Text, mode), 100)]),
                ..Tokens::default()
            },
            ..Usage::default()
        };
        assert_eq!(
            sides(&priced(&prices, &read(CacheReadMode::Explicit))),
            (25.0, 0.0)
        );
        assert_eq!(
            sides(&priced(&prices, &read(CacheReadMode::Implicit))),
            (50.0, 0.0)
        );
    }

    #[rstest]
    #[case::characters_replace_text_tokens(4.0, 400, (1600.0, vec![(CHARACTER, 1600.0)]))]
    #[case::free_characters_still_replace_text_tokens(0.0, 400, (0.0, vec![(CHARACTER, 0.0)]))]
    #[case::no_characters_reported(4.0, 0, (100.0, vec![(INPUT, 100.0)]))]
    fn characters_replace_text_tokens_when_the_row_prices_them(
        #[case] character_price: f64,
        #[case] characters: u64,
        #[case] expected: (f64, Vec<(Meter, f64)>),
    ) {
        let prices = prices(&[
            (INPUT, Standard, BASE, 1.0),
            (CHARACTER, Standard, BASE, character_price),
        ]);
        let usage = Usage {
            characters: Characters {
                prompt: characters,
                completion: 0,
            },
            ..text(100, 100, 0)
        };
        let priced = priced(&prices, &usage);

        assert_eq!((priced.prompt.get(), billed(&priced)), expected);
    }

    #[rstest]
    fn characters_without_a_price_leave_the_text_tokens_billed() {
        let prices = prices(&[(INPUT, Standard, BASE, 1.0)]);
        let usage = Usage {
            characters: Characters {
                prompt: 400,
                completion: 0,
            },
            ..text(100, 100, 0)
        };
        let priced = priced(&prices, &usage);

        assert_eq!(billed(&priced), [(INPUT, 100.0)]);
        assert_eq!(priced.missing, [CHARACTER]);
    }

    fn multimodal() -> Usage {
        Usage {
            prompt_tokens: 60,
            tokens: Tokens {
                input: BTreeMap::from([
                    (TokenKind::Audio, 10),
                    (TokenKind::Image, 20),
                    (TokenKind::Video, 30),
                ]),
                ..Tokens::default()
            },
            durations: BTreeMap::from([
                (DurationMeter::InputAudio, Duration::from_millis(2_500)),
                (DurationMeter::InputVideo, Duration::from_secs(4)),
            ]),
            images: Images {
                input: 3,
                ..Images::default()
            },
            search: Search {
                queries: 2,
                ..Search::default()
            },
            ..Usage::default()
        }
    }

    #[rstest]
    fn a_per_second_or_per_image_price_replaces_that_modalitys_tokens() {
        let image = Meter::Input(InputUnit::Image);
        let audio_second = Meter::Input(InputUnit::AudioSecond);
        let video_second = Meter::Input(InputUnit::VideoSecond { over_seconds: None });
        let query = Meter::Input(InputUnit::Query { over_results: None });
        let prices = prices(&[
            (INPUT, Standard, BASE, 1.0),
            (image.clone(), Standard, BASE, 100.0),
            (audio_second.clone(), Standard, BASE, 8.0),
            (video_second.clone(), Standard, BASE, 16.0),
            (query.clone(), Standard, BASE, 0.5),
        ]);
        let priced = priced(&prices, &multimodal());

        assert_eq!(
            billed(&priced),
            [
                (image, 300.0),
                (video_second, 64.0),
                (audio_second, 20.0),
                (query, 1.0)
            ]
        );
        assert_eq!(sides(&priced), (385.0, 0.0));
    }

    #[rstest]
    fn modality_tokens_bill_at_their_rate_or_the_text_rate() {
        let prices = prices(&[
            (INPUT, Standard, BASE, 1.0),
            (AUDIO_INPUT, Standard, BASE, 4.0),
        ]);
        let priced = priced(&prices, &multimodal());

        assert_eq!(
            billed(&priced),
            [
                (AUDIO_INPUT, 40.0),
                (Meter::Input(InputUnit::Token(TokenKind::Image)), 20.0),
                (Meter::Input(InputUnit::Token(TokenKind::Video)), 30.0),
            ]
        );
        assert_eq!(
            priced
                .lines
                .iter()
                .map(|line| line.fallback_from.clone())
                .collect::<Vec<_>>(),
            [None, Some(INPUT), Some(INPUT)]
        );
        assert_eq!(priced.missing.len(), 4);
    }

    #[rstest]
    fn the_completion_side_bills_each_modality_and_reasoning() {
        let audio = Meter::Output(OutputUnit::Token(TokenKind::Audio));
        let prices = prices(&[
            (OUTPUT, Standard, BASE, 2.0),
            (audio.clone(), Standard, BASE, 16.0),
            (Meter::REASONING, Standard, BASE, 4.0),
        ]);
        let usage = Usage {
            tokens: Tokens {
                output: BTreeMap::from([
                    (TokenKind::Text, 10),
                    (TokenKind::Audio, 2),
                    (TokenKind::Image, 5),
                ]),
                reasoning: 3,
                ..Tokens::default()
            },
            ..Usage::default()
        };
        let priced = priced(&prices, &usage);

        assert_eq!(
            billed(&priced),
            [
                (OUTPUT, 20.0),
                (audio, 32.0),
                (Meter::REASONING, 12.0),
                (Meter::Output(OutputUnit::Token(TokenKind::Image)), 10.0),
            ]
        );
        assert_eq!(sides(&priced), (0.0, 74.0));
    }

    #[rstest]
    #[case::eu_residency(
        CallContext { residency: Some(Residency::Eu), ..call() },
        (150.0, 30.0)
    )]
    #[case::us_residency_without_a_multiplier(
        CallContext { residency: Some(Residency::Us), ..call() },
        (100.0, 20.0)
    )]
    #[case::fedramp(CallContext { fedramp_endpoint: true, ..call() }, (200.0, 40.0))]
    #[case::regional_endpoint(CallContext { non_global_endpoint: true, ..call() }, (125.0, 25.0))]
    #[case::residency_and_regional_endpoint_compound(
        CallContext { residency: Some(Residency::Eu), non_global_endpoint: true, ..call() },
        (187.5, 37.5)
    )]
    #[case::no_call_fact(call(), (100.0, 20.0))]
    fn an_uplift_applies_only_to_the_call_it_names(
        #[case] call: CallContext,
        #[case] expected: (f64, f64),
    ) {
        let prices = prices(&[(INPUT, Standard, BASE, 1.0), (OUTPUT, Standard, BASE, 2.0)]);
        let adjustments = Adjustments::new([
            (
                Uplift::Residency(Residency::Eu),
                Multiplier::new(1.5).unwrap(),
            ),
            (Uplift::FedrampEndpoint, Multiplier::new(2.0).unwrap()),
            (Uplift::NonGlobalEndpoint, Multiplier::new(1.25).unwrap()),
        ]);
        let priced = priced_with(&prices, &adjustments, &text(100, 100, 10), &call);

        assert_eq!(sides(&priced), expected);
        assert_eq!(
            priced.lines.iter().map(|line| line.amount).sum::<Usd>(),
            priced.total()
        );
    }

    #[rstest]
    fn an_open_off_peak_window_bills_the_off_peak_rates() {
        let prices = prices_with(
            &[(INPUT, Standard, BASE, 1.0), (OUTPUT, Standard, BASE, 2.0)],
            off_peak_all_day(OffPeakRates {
                input: UnitPrice::new(0.5),
                ..OffPeakRates::default()
            }),
            Billing::default(),
        );
        assert_eq!(sides(&priced(&prices, &text(100, 100, 10))), (50.0, 20.0));
    }

    #[rstest]
    fn usage_no_price_covers_is_unpriced() {
        let prices = prices(&[(CHARACTER, Standard, BASE, 1.0)]);
        assert_eq!(
            calculate(
                &prices,
                &Adjustments::default(),
                &text(100, 100, 10),
                &call()
            ),
            Outcome::Unpriced {
                missing: vec![INPUT, OUTPUT]
            }
        );
    }

    #[rstest]
    fn no_usage_is_priced_at_nothing() {
        let prices = prices(&[(INPUT, Standard, BASE, 1.0)]);
        let priced = priced(&prices, &Usage::default());

        assert_eq!(sides(&priced), (0.0, 0.0));
        assert!(priced.lines.is_empty());
    }
}
