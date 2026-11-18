//! The `speech` kind, a port of the speech branch of `cost_per_token` (`cost_calculator.py`): a Vertex AI Lyria call
//! bills each generation at the first table's `output_cost_per_image`, any other call bills the prompt's characters
//! at `input_cost_per_character`. A row priced per token instead is `NotPorted`, since main bills it through the
//! generic token path

use crate::price::{Channel, InputUnit, Meter, OutputUnit};
use crate::priced::{Quantity, Side};
use crate::query::{Header, Kind};
use crate::receipt::{PriceOutcome, Receipt, Unpriced};
use crate::row::RowSource;
use crate::units::{self, Billed};
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
        return match listed(&Meter::TEXT_INPUT) {
            true => Unpriced::NotPorted(Kind::Speech).into(),
            false => Unpriced::MissingRates {
                row: named.at,
                missing: vec![CHARACTER],
            }
            .into(),
        };
    }
    let Some(characters) = query.characters else {
        return Unpriced::MissingUsage {
            row: named.at,
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
mod parity;

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::price::ServiceTier;
    use crate::query::PriceSource;
    use crate::testing::{BASE, TestRow, TestSource, header, priced, prices, row_ref};

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
    fn uncounted_characters_are_missing_usage() {
        let source = TestSource::with_row("tts-1", row(&[(CHARACTER, 1.0)]));
        assert_eq!(
            price(&source, &query(None, None)),
            Unpriced::MissingUsage {
                row: row_ref("tts-1"),
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
        };
        let receipt = priced(price(&source, &lyria));
        assert_eq!(
            (receipt.prompt.get(), receipt.completion.get()),
            (0.0, expected)
        );
    }
}
