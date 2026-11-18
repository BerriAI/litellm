//! The `speech` kind, a port of the speech branch of `cost_per_token` (`cost_calculator.py`): a Vertex AI Lyria call
//! bills each generation at the first table's `output_cost_per_image`, any other call bills the prompt's characters
//! at `input_cost_per_character`. A row priced per token instead bills the call's token usage through
//! `generic_cost_per_token`, `NotPorted` when the caller sent none

use crate::price::{Channel, InputUnit, Meter, OutputUnit};
use crate::priced::{Quantity, Side};
use crate::query::{Header, Kind};
use crate::receipt::{PriceOutcome, Receipt, Unpriced};
use crate::row::RowSource;
use crate::units::{self, Billed, TokenBilled};
use crate::usage::{Characters, Usage};

const CHARACTER: Meter = Meter::Input(InputUnit::Character);
const GENERATION: Meter = Meter::Output(OutputUnit::Image(None));

/// One speech call
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct SpeechQuery {
    #[cfg_attr(feature = "serde", serde(flatten))]
    pub header: Header,
    /// Characters billed, `None` when the caller counted none
    #[cfg_attr(feature = "serde", serde(default))]
    pub characters: Option<u64>,
    /// Lyria generations, each billed whole. The header's `Rows` are the Lyria rows in the order main reads them
    #[cfg_attr(feature = "serde", serde(default))]
    pub generations: Option<u64>,
    /// The token usage the response reported, billed when the row prices tokens rather than characters
    #[cfg_attr(feature = "serde", serde(default))]
    pub tokens: Option<TokenBilled>,
}

pub(crate) fn price<S: RowSource>(source: &S, query: &SpeechQuery) -> PriceOutcome {
    match query.generations {
        Some(generations) => generated(source, query, generations),
        None => spoken(source, query),
    }
}

fn generated<S: RowSource>(source: &S, query: &SpeechQuery, generations: u64) -> PriceOutcome {
    let layers = units::layers(source, &query.header);
    let priced = layers.iter().find_map(|(at, row)| {
        units::listed(*row, &GENERATION, Channel::Online).map(|rate| (at, rate))
    });
    let Some((at, rate)) = priced else {
        return match layers.first() {
            Some((at, _)) => Unpriced::MissingRates {
                row: at.clone(),
                missing: vec![GENERATION],
            }
            .into(),
            None => units::no_row(&query.header),
        };
    };
    let line = units::line(
        GENERATION,
        Quantity::Count(generations),
        rate,
        Side::Completion,
    );
    let receipt = units::receipt(
        source,
        &query.header,
        None,
        Billed::of(vec![line], Vec::new()),
    );
    Receipt {
        row: Some(at.clone()),
        candidate: query.header.candidates.first().cloned(),
        ..receipt
    }
    .into()
}

fn spoken<S: RowSource>(source: &S, query: &SpeechQuery) -> PriceOutcome {
    let Some(named) = units::named(source, &query.header) else {
        return units::no_row(&query.header);
    };
    let listed = |meter: &Meter| units::listed(named.row, meter, Channel::Online).is_some();
    if !listed(&CHARACTER) {
        return match (listed(&Meter::TEXT_INPUT), &query.tokens) {
            (true, Some(tokens)) => units::generic_tokens(source, &query.header, tokens),
            (true, None) => Unpriced::NotPorted(Kind::Speech).into(),
            (false, _) => Unpriced::MissingRates {
                row: named.at,
                missing: vec![CHARACTER],
            }
            .into(),
        };
    }
    let Some(characters) = query.characters else {
        return Unpriced::MissingUsage {
            row: Some(named.at),
            meter: CHARACTER,
        }
        .into();
    };
    let usage = Usage {
        characters: Characters {
            prompt: characters,
            completion: 0,
        },
        ..Usage::default()
    };
    units::bill(
        source,
        &query.header,
        &named,
        &usage,
        &units::call(&query.header.basis),
    )
}

#[cfg(test)]
mod parity {
    //! Parity with the speech branch of main's `cost_per_token`. `fixtures/speech_parity.json` was captured once from
    //! main's Python: each case holds the cost fields `get_model_info` returned (or the Lyria row), the billed characters
    //! and main's `(prompt, completion)`, or that it raised. The adapter raises on any unpriced speech call, as main did

    use rstest::rstest;
    use serde_json::Value;

    use super::{SpeechQuery, price};
    use crate::query::{Header, PriceSource};
    use crate::receipt::PriceOutcome;
    use crate::testing::{TestSource, expected_sides, fixture_case, header, row_ref, units_row};

    const FIXTURE: &str = include_str!("../fixtures/speech_parity.json");
    const LYRIA: &str = "lyria";

    fn case(name: &str) -> Value {
        fixture_case(FIXTURE, name)
    }

    #[rstest]
    #[case::tts_1("tts_1")]
    #[case::azure_tts("azure_tts")]
    #[case::elevenlabs("elevenlabs")]
    #[case::dashscope_characters("dashscope_characters")]
    #[case::uncounted_characters("uncounted_characters")]
    #[case::lyria_generation("lyria_generation")]
    fn a_speech_call_costs_what_main_charged(#[case] name: &str) {
        let case = case(name);
        let model = case["model"].as_str().unwrap();
        let base = header(case["provider"].as_str().unwrap(), model);
        let query = match case["lyria"].is_null() {
            true => SpeechQuery {
                header: base,
                characters: case["characters"].as_u64(),
                generations: None,
                tokens: None,
            },
            false => SpeechQuery {
                header: Header {
                    source: PriceSource::Rows {
                        rows: vec![row_ref(LYRIA)],
                    },
                    ..base
                },
                characters: None,
                generations: Some(1),
                tokens: None,
            },
        };
        let source = TestSource::with_row(model, units_row(&case["row"]))
            .and_row(LYRIA, units_row(&case["lyria"]));
        let sides = match price(&source, &query) {
            PriceOutcome::Priced(receipt) => Some((receipt.prompt.get(), receipt.completion.get())),
            PriceOutcome::Unpriced(_) => None,
        };
        assert_eq!(sides, expected_sides(&case));
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::price::ServiceTier;
    use crate::query::PriceSource;
    use crate::testing::{BASE, TestRow, TestSource, header, priced, prices, row_ref, usage};

    fn row(rates: &[(Meter, f64)]) -> TestRow {
        TestRow {
            prices: prices(
                &rates
                    .iter()
                    .map(|(meter, rate)| (meter.clone(), ServiceTier::Standard, BASE, *rate))
                    .collect::<Vec<_>>(),
            ),
            ..TestRow::default()
        }
    }

    fn query(characters: Option<u64>, generations: Option<u64>) -> SpeechQuery {
        SpeechQuery {
            header: header("openai", "tts-1"),
            characters,
            generations,
            tokens: None,
        }
    }

    #[rstest]
    fn characters_bill_the_prompt_at_the_character_rate() {
        let source = TestSource::with_row("tts-1", row(&[(CHARACTER, 0.000015)]));
        let receipt = priced(price(&source, &query(Some(120), None)));
        assert_eq!(
            (receipt.prompt.get(), receipt.completion.get()),
            (0.000015 * 120.0, 0.0)
        );
    }

    #[rstest]
    #[case::token_priced(row(&[(Meter::TEXT_INPUT, 1.0)]), Unpriced::NotPorted(Kind::Speech))]
    #[case::unpriced(
        row(&[]),
        Unpriced::MissingRates { row: row_ref("tts-1"), missing: vec![CHARACTER] }
    )]
    fn a_row_without_a_character_rate_is_not_billed_here(
        #[case] speech_row: TestRow,
        #[case] expected: Unpriced,
    ) {
        let source = TestSource::with_row("tts-1", speech_row);
        assert_eq!(price(&source, &query(Some(5), None)), expected.into());
    }

    #[rstest]
    fn a_token_priced_row_bills_the_reported_tokens_on_the_row_main_read_for_them() {
        let source = TestSource::with_row("tts-1", row(&[(Meter::TEXT_INPUT, 9.0)])).and_row(
            "tokens-row",
            row(&[(Meter::TEXT_INPUT, 2e-6), (Meter::TEXT_OUTPUT, 8e-6)]),
        );
        let spoken = SpeechQuery {
            tokens: Some(TokenBilled {
                row: row_ref("tokens-row"),
                usage: usage(300, 50),
            }),
            ..query(Some(5), None)
        };
        let receipt = priced(price(&source, &spoken));
        assert_eq!(
            (receipt.prompt.get(), receipt.completion.get()),
            (2e-6 * 300.0, 8e-6 * 50.0)
        );
    }

    #[rstest]
    fn uncounted_characters_are_missing_usage() {
        let source = TestSource::with_row("tts-1", row(&[(CHARACTER, 1.0)]));
        assert_eq!(
            price(&source, &query(None, None)),
            Unpriced::MissingUsage {
                row: Some(row_ref("tts-1")),
                meter: CHARACTER
            }
            .into()
        );
    }

    #[rstest]
    #[case::runtime_row_prices(vec!["runtime", "bundled"], 0.06 * 3.0)]
    #[case::bundled_row_when_the_runtime_one_does_not(vec!["unpriced", "bundled"], 0.08 * 3.0)]
    fn each_lyria_generation_bills_the_first_rows_generation_rate(
        #[case] rows: Vec<&str>,
        #[case] expected: f64,
    ) {
        let source = TestSource::default()
            .and_row("runtime", row(&[(GENERATION, 0.06)]))
            .and_row("unpriced", row(&[]))
            .and_row("bundled", row(&[(GENERATION, 0.08)]));
        let lyria = SpeechQuery {
            header: Header {
                source: PriceSource::Rows {
                    rows: rows.into_iter().map(row_ref).collect(),
                },
                ..header("vertex_ai", "lyria-002")
            },
            characters: Some(10),
            generations: Some(3),
            tokens: None,
        };
        let receipt = priced(price(&source, &lyria));
        assert_eq!(
            (receipt.prompt.get(), receipt.completion.get()),
            (0.0, expected)
        );
    }
}
